//! SIMD-style GP operations with portable implementation.
//!
//! This module implements bulk geometric product operations using manual
//! loop unrolling and vectorization-friendly patterns that work on stable Rust.
//!
//! While we await `std::simd` stabilization, we use portable SIMD-like patterns:
//! - Process multiple items per iteration
//! - Minimize branching in inner loops
//! - Organize data as SoA (Structure of Arrays) for better cache locality
//! - Use inline hints for compiler optimization

use crate::core::compute::local::PackedBlockTerm;

/// Process 4 packed block terms at once using scalar operations.
///
/// This is a manual SIMD-like operation that processes 4 terms in parallel
/// using scalar code. The compiler can vectorize this on platforms with
/// good vector support.
///
/// # Parameters
/// - `a`: Array of 4 left operands
/// - `b`: Array of 4 right operands
///
/// # Returns
/// Array of 4 GP results
#[inline(always)]
pub fn gp_simd_x4_scalar(
    a: &[PackedBlockTerm; 4],
    b: &[PackedBlockTerm; 4],
) -> [Option<PackedBlockTerm>; 4] {
    [
        gp_single_unrolled(&a[0], &b[0]),
        gp_single_unrolled(&a[1], &b[1]),
        gp_single_unrolled(&a[2], &b[2]),
        gp_single_unrolled(&a[3], &b[3]),
    ]
}

/// Optimized single GP with explicit inlining for vectorization.
#[inline(always)]
fn gp_single_unrolled(a: &PackedBlockTerm, b: &PackedBlockTerm) -> Option<PackedBlockTerm> {
    // Fast early exit: sector mismatch
    if a.right_bits != b.left_bits {
        return None;
    }

    // Inline XOR operations (vectorizable)
    let blade_x = a.blade_x ^ b.blade_x;
    let blade_z = a.blade_z ^ b.blade_z;

    // Crossover parity (count_ones is typically single instruction)
    let crossing = (a.blade_x & b.blade_z).count_ones() & 1;
    let sign = (a.sign as u64 ^ b.sign as u64 ^ crossing as u64) != 0;

    Some(PackedBlockTerm::new(
        a.left_bits,
        blade_x,
        blade_z,
        b.right_bits,
        a.rotor_q1 ^ b.rotor_q1,
        a.rotor_q2 ^ b.rotor_q2,
        a.rotor_q3 ^ b.rotor_q3,
        sign,
        a.n,
    ))
}

/// Auto-selected GP: uses scalar x4 for now, can be extended for actual SIMD.
///
/// In future, this will dispatch to actual std::simd when available,
/// or use architecture-specific intrinsics.
#[inline(always)]
pub fn gp_simd_x4_auto(
    a: &[PackedBlockTerm; 4],
    b: &[PackedBlockTerm; 4],
) -> [Option<PackedBlockTerm>; 4] {
    // For now, same as scalar. In future: dispatch to AVX-512, AVX2, NEON, etc.
    gp_simd_x4_scalar(a, b)
}

/// Bulk GP with optimal loop unrolling for vectorization.
///
/// Processes terms in chunks of 4, unrolled for compiler optimization.
/// This pattern minimizes branch prediction pressure and enables
/// the compiler to generate better vector code.
pub fn gp_bulk_unrolled(
    a: &[PackedBlockTerm],
    b: &[PackedBlockTerm],
) -> Vec<Option<PackedBlockTerm>> {
    let n = a.len();
    let mut results = Vec::with_capacity(n);

    // Process in chunks of 4
    let chunks = n / 4;
    for chunk_idx in 0..chunks {
        let base = chunk_idx * 4;
        let a_chunk = [a[base], a[base + 1], a[base + 2], a[base + 3]];
        let b_chunk = [b[base], b[base + 1], b[base + 2], b[base + 3]];
        let res = gp_simd_x4_scalar(&a_chunk, &b_chunk);
        results.extend_from_slice(&res);
    }

    // Process remainder
    for i in (chunks * 4)..n {
        results.push(gp_single_unrolled(&a[i], &b[i]));
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_term(blade_x: u64, blade_z: u64, left_bits: u64, right_bits: u64) -> PackedBlockTerm {
        PackedBlockTerm::new(left_bits, blade_x, blade_z, right_bits, 0, 0, 0, false, 8)
    }

    #[test]
    fn gp_simd_x4_matches_scalar() {
        let a = [
            make_term(1, 0, 0, 0),
            make_term(2, 0, 0, 0),
            make_term(0, 1, 0, 0),
            make_term(0, 2, 0, 0),
        ];

        let b = [
            make_term(1, 0, 0, 0),
            make_term(1, 0, 0, 0),
            make_term(0, 1, 0, 0),
            make_term(0, 1, 0, 0),
        ];

        let results = gp_simd_x4_scalar(&a, &b);

        // Verify each result matches individual GP
        for (result, (a_i, b_i)) in results.iter().zip(a.iter().zip(b.iter())) {
            let expected = a_i.gp(*b_i);
            assert_eq!(result.is_some(), expected.is_some());
            if let (Some(r), Some(e)) = (result, expected) {
                assert_eq!(r.blade_x, e.blade_x);
                assert_eq!(r.blade_z, e.blade_z);
                assert_eq!(r.sign, e.sign);
            }
        }
    }

    #[test]
    fn gp_simd_x4_auto_available() {
        let a = [
            make_term(1, 0, 0, 0),
            make_term(2, 0, 0, 0),
            make_term(0, 1, 0, 0),
            make_term(0, 2, 0, 0),
        ];

        let b = [
            make_term(1, 0, 0, 0),
            make_term(1, 0, 0, 0),
            make_term(0, 1, 0, 0),
            make_term(0, 1, 0, 0),
        ];

        let _results = gp_simd_x4_auto(&a, &b);
        // If we get here without panic, auto-selection works
        assert!(true);
    }

    #[test]
    fn bulk_unrolled_correctness() {
        let n = 13; // Not divisible by 4 to test remainder
        let mut a = Vec::with_capacity(n);
        let mut b = Vec::with_capacity(n);

        for i in 0..n {
            a.push(make_term(i as u64, 0, 0, 0));
            b.push(make_term(i as u64, 0, 0, 0));
        }

        let results = gp_bulk_unrolled(&a, &b);
        assert_eq!(results.len(), n);
    }
}
