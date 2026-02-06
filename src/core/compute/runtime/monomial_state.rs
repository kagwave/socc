use std::collections::HashMap;

use crate::core::bits::{Blade, Rotor, Sector};
use crate::core::compute::local::PackedBlockTerm;
use crate::core::compute::mem::gp_simd_x4_auto;
use crate::core::compute::multivector_packed::{PackedMultivector, PackedTermCoeff};
use crate::core::forms::{diagonal::DiagonalPacked, monomial::MonomialPacked};
use crate::core::ir::{Multivector, Term};

const EPS: f64 = 1e-12;

#[inline(always)]
fn is_zero(x: f64) -> bool {
    x.abs() < EPS
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct StateKey {
    sector: u64,
    blade_x: u64,
    blade_z: u64,
    rotor_q1: u64,
    rotor_q2: u64,
    rotor_q3: u64,
    sign: bool,
    n: u8,
}

/// Sparse monomial-state representation.
///
/// This is the first true Schrödinger fast path:
///
///     ψ = Σ_i coeffs[i] * payload[i] * Π_{sectors[i]}
///
/// The state is not stored as a full Multivector unless fallback/export is needed.
///
/// Hot-loop design:
/// - sectors: u64 lane
/// - coeffs: f64 lane
/// - payload: PackedBlockTerm lane
/// - scratch buffers are reused on every apply
#[derive(Clone, Debug, PartialEq)]
pub struct MonomialState {
    pub sectors: Vec<u64>,
    pub coeffs: Vec<f64>,
    pub payload: Vec<PackedBlockTerm>,
    pub n: u8,

    scratch_sectors: Vec<u64>,
    scratch_coeffs: Vec<f64>,
    scratch_payload: Vec<PackedBlockTerm>,
}

impl MonomialState {
    /// Vacuum sparse state:
    ///
    ///     ψ₀ = Π_vacuum
    #[inline]
    pub fn from_vacuum(n: u8, vacuum: u64) -> Self {
        Self {
            sectors: vec![vacuum],
            coeffs: vec![1.0],
            payload: vec![PackedBlockTerm::identity(n)],
            n,

            scratch_sectors: Vec::with_capacity(1),
            scratch_coeffs: Vec::with_capacity(1),
            scratch_payload: Vec::with_capacity(1),
        }
    }

    /// Useful for tests and synthetic benchmarks.
    pub fn from_sparse(
        n: u8,
        sectors: Vec<u64>,
        coeffs: Vec<f64>,
        payload: Vec<PackedBlockTerm>,
    ) -> Self {
        assert_eq!(sectors.len(), coeffs.len(), "sector/coeff length mismatch");
        assert_eq!(sectors.len(), payload.len(), "sector/payload length mismatch");

        let cap = sectors.len();

        Self {
            sectors,
            coeffs,
            payload,
            n,
            scratch_sectors: Vec::with_capacity(cap),
            scratch_coeffs: Vec::with_capacity(cap),
            scratch_payload: Vec::with_capacity(cap),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.sectors.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.sectors.is_empty()
    }

    /// Apply a monomial operator with SIMD-accelerated payload composition.
    ///
    ///     x ↦ op.perm[x]
    ///     c ↦ c * op.coeffs[x]
    ///     K ↦ op.payload[x] * K
    ///
    /// Uses fixed-size 4-element chunks with the portable SIMD kernel.
    /// Gather/compute/scatter pattern for CPU cache efficiency.
    ///
    /// This reuses scratch buffers and swaps them into place.
    pub fn apply_monomial(&mut self, op: &MonomialPacked) {
        assert_eq!(self.n, op.n, "MonomialState/op width mismatch");

        self.scratch_sectors.clear();
        self.scratch_coeffs.clear();
        self.scratch_payload.clear();

        self.scratch_sectors.reserve(self.sectors.len());
        self.scratch_coeffs.reserve(self.coeffs.len());
        self.scratch_payload.reserve(self.payload.len());

        let len = self.sectors.len();

        // Process full 4-element chunks with SIMD kernel
        for chunk_start in (0..len).step_by(4) {
            let chunk_end = std::cmp::min(chunk_start + 4, len);
            let chunk_size = chunk_end - chunk_start;

            if chunk_size == 4 {
                // Full chunk: use SIMD kernel
                let mut op_payloads = [PackedBlockTerm::identity(self.n); 4];
                let mut state_payloads = [PackedBlockTerm::identity(self.n); 4];

                for (j, i) in (chunk_start..chunk_end).enumerate() {
                    let x = self.sectors[i] as usize;
                    op_payloads[j] = op.payload[x];
                    state_payloads[j] = self.payload[i];
                }

                // Compute phase: use SIMD kernel for payload composition
                let composed = gp_simd_x4_auto(&op_payloads, &state_payloads);

                // Scatter phase: accumulate results
                for (j, i) in (chunk_start..chunk_end).enumerate() {
                    let x = self.sectors[i] as usize;
                    let y = op.perm[x];
                    let coeff = self.coeffs[i] * op.coeffs[x];

                    if is_zero(coeff) {
                        continue;
                    }

                    if let Some(p) = composed[j] {
                        self.scratch_sectors.push(y);
                        self.scratch_coeffs.push(coeff);
                        self.scratch_payload.push(p);
                    }
                }
            } else {
                // Remainder: process individually without SIMD
                for i in chunk_start..chunk_end {
                    let x = self.sectors[i] as usize;
                    let y = op.perm[x];
                    let coeff = self.coeffs[i] * op.coeffs[x];

                    if is_zero(coeff) {
                        continue;
                    }

                    let next_payload = compose_payloads(op.payload[x], self.payload[i], self.n);

                    if let Some(p) = next_payload {
                        self.scratch_sectors.push(y);
                        self.scratch_coeffs.push(coeff);
                        self.scratch_payload.push(p);
                    }
                }
            }
        }

        std::mem::swap(&mut self.sectors, &mut self.scratch_sectors);
        std::mem::swap(&mut self.coeffs, &mut self.scratch_coeffs);
        std::mem::swap(&mut self.payload, &mut self.scratch_payload);
    }

    /// Apply a diagonal operator:
    ///
    ///     c_x ↦ d_x c_x
    ///
    /// This is in-place because sectors and payloads do not move.
    pub fn apply_diagonal(&mut self, op: &DiagonalPacked) {
        assert_eq!(self.n, op.n, "MonomialState/op width mismatch");

        let mut write = 0usize;

        for read in 0..self.sectors.len() {
            let x = self.sectors[read];
            let coeff = self.coeffs[read] * op.coeff_of(x);

            if is_zero(coeff) {
                continue;
            }

            self.sectors[write] = self.sectors[read];
            self.coeffs[write] = coeff;
            self.payload[write] = self.payload[read];
            write += 1;
        }

        self.sectors.truncate(write);
        self.coeffs.truncate(write);
        self.payload.truncate(write);
    }

    /// Optional cleanup if external/manual construction introduces duplicate sectors.
    ///
    /// Monomial permutations should preserve uniqueness if the state already had
    /// unique sectors, so this should not be called in the hot path.
    pub fn canonicalize(&mut self) {
        let mut acc: HashMap<StateKey, f64> = HashMap::new();

        for i in 0..self.sectors.len() {
            if is_zero(self.coeffs[i]) {
                continue;
            }

            let p = local_payload(self.payload[i], self.n);
            let key = StateKey {
                sector: self.sectors[i],
                blade_x: p.blade_x,
                blade_z: p.blade_z,
                rotor_q1: p.rotor_q1,
                rotor_q2: p.rotor_q2,
                rotor_q3: p.rotor_q3,
                sign: p.sign,
                n: p.n,
            };

            *acc.entry(key).or_insert(0.0) += self.coeffs[i];
        }

        self.sectors.clear();
        self.coeffs.clear();
        self.payload.clear();

        for (key, coeff) in acc {
            if is_zero(coeff) {
                continue;
            }

            self.sectors.push(key.sector);
            self.coeffs.push(coeff);
            self.payload.push(PackedBlockTerm::new(
                0,
                key.blade_x,
                key.blade_z,
                0,
                key.rotor_q1,
                key.rotor_q2,
                key.rotor_q3,
                key.sign,
                key.n,
            ));
        }
    }

    /// Convert the sparse monomial state to a Multivector.
    ///
    /// This is export/fallback/debug only. Do not call this in the fast path.
    pub fn to_mv(&self) -> Multivector {
        let mut terms = Vec::with_capacity(self.sectors.len());

        for i in 0..self.sectors.len() {
            if is_zero(self.coeffs[i]) {
                continue;
            }

            let mut t = packed_payload_to_term(self.payload[i], self.n);

            // State convention:
            //
            //     payload * Π_sector
            //
            // Therefore this is a right-sector state term, not an operator block.
            t.left = None;
            t.right = Some(Sector::new(self.sectors[i], self.n));
            t.coeff *= self.coeffs[i];

            terms.push(t);
        }

        Multivector::from_terms(self.n, terms)
    }

    /// Convert the sparse monomial state to PackedMultivector.
    ///
    /// Preferred generic-runtime boundary.
    pub fn to_packed(&self) -> PackedMultivector {
        let mut terms = Vec::with_capacity(self.sectors.len());

        for i in 0..self.sectors.len() {
            let coeff = self.coeffs[i];
            if is_zero(coeff) {
                continue;
            }

            let p = local_payload(self.payload[i], self.n);

            terms.push(PackedTermCoeff {
                term: PackedBlockTerm::new(
                    0,
                    p.blade_x,
                    p.blade_z,
                    self.sectors[i],
                    p.rotor_q1,
                    p.rotor_q2,
                    p.rotor_q3,
                    p.sign,
                    self.n,
                ),
                coeff,
            });
        }

        PackedMultivector::new(self.n, terms)
    }
}

/// Compose two local payloads.
///
/// The payload lane should represent local blade/rotor action, not the global
/// input/output sector map. To avoid accidental Peirce-block incompatibility,
/// we strip left/right sector bits before multiplying.
#[inline(always)]
fn compose_payloads(
    op_payload: PackedBlockTerm,
    state_payload: PackedBlockTerm,
    n: u8,
) -> Option<PackedBlockTerm> {
    if op_payload.is_identity() {
        return Some(local_payload(state_payload, n));
    }

    if state_payload.is_identity() {
        return Some(local_payload(op_payload, n));
    }

    let a = local_payload(op_payload, n);
    let b = local_payload(state_payload, n);

    a.gp(b)
}

/// Strip sector boundaries from a packed term, preserving only local blade/rotor/sign payload.
#[inline(always)]
fn local_payload(p: PackedBlockTerm, n: u8) -> PackedBlockTerm {
    PackedBlockTerm {
        left_bits: 0,
        right_bits: 0,
        blade_x: p.blade_x,
        blade_z: p.blade_z,
        rotor_q1: p.rotor_q1,
        rotor_q2: p.rotor_q2,
        rotor_q3: p.rotor_q3,
        sign: p.sign,
        n,
    }
}

/// Convert a local payload into an IR term without sectors.
#[inline(always)]
fn packed_payload_to_term(p: PackedBlockTerm, n: u8) -> Term {
    let p = local_payload(p, n);

    let has_rotor = (p.rotor_q1 | p.rotor_q2 | p.rotor_q3) != 0;

    Term {
        left: None,
        blade: Blade::new(p.blade_x, p.blade_z, p.sign),
        right: None,
        rotor: if has_rotor {
            Some(crate::core::bits::Rotor {
                q1_mask: p.rotor_q1,
                q2_mask: p.rotor_q2,
                q3_mask: p.rotor_q3,
                sign: false,
            })
        } else {
            None
        },
        coeff: 1.0,
    }
}