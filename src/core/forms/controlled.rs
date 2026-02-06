use crate::core::bits::{Blade, Rotor, Sector};
use crate::core::compute::local::PackedBlockTerm;
use crate::core::ir::{Multivector, Term};

const EPS: f64 = 1e-12;

#[inline(always)]
fn is_zero(x: f64) -> bool {
    x.abs() < EPS
}

/// Two-branch exact-sector controlled form:
///
///     off_coeff * Π_off + on_coeff * (Π_on K Π_on)
///
/// This is a narrow structured form. General controlled gates should eventually
/// lower to MonomialPacked when they permute sectors.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ControlledPacked {
    pub off_bits: u64,
    pub on_bits: u64,
    pub off_coeff: f64,
    pub on_coeff: f64,
    pub k: PackedBlockTerm,
    pub n: u8,
}

impl ControlledPacked {
    #[inline(always)]
    pub fn identity_on(on_bits: u64, n: u8) -> PackedBlockTerm {
        PackedBlockTerm::new(
            on_bits,
            0,
            0,
            on_bits,
            0,
            0,
            0,
            false,
            n,
        )
    }

    #[inline(always)]
    pub fn k_is_on_local(k: PackedBlockTerm, on_bits: u64, n: u8) -> bool {
        k.n == n && k.left_bits == on_bits && k.right_bits == on_bits
    }

    pub fn new(
        off_bits: u64,
        on_bits: u64,
        off_coeff: f64,
        on_coeff: f64,
        k: PackedBlockTerm,
        n: u8,
    ) -> Option<Self> {
        if off_bits == on_bits {
            return None;
        }

        if !Self::k_is_on_local(k, on_bits, n) {
            return None;
        }

        Some(Self {
            off_bits,
            on_bits,
            off_coeff,
            on_coeff,
            k,
            n,
        })
    }

    pub fn unit_branches(
        off_bits: u64,
        on_bits: u64,
        k: PackedBlockTerm,
        n: u8,
    ) -> Option<Self> {
        Self::new(off_bits, on_bits, 1.0, 1.0, k, n)
    }

    #[inline(always)]
    pub fn off_sector(&self) -> Sector {
        Sector::new(self.off_bits, self.n)
    }

    #[inline(always)]
    pub fn on_sector(&self) -> Sector {
        Sector::new(self.on_bits, self.n)
    }

    #[inline(always)]
    pub fn same_split(&self, rhs: &Self) -> bool {
        self.n == rhs.n && self.off_bits == rhs.off_bits && self.on_bits == rhs.on_bits
    }

    pub fn gp(self, rhs: Self) -> Option<Self> {
        if !self.same_split(&rhs) {
            return None;
        }

        let off_coeff = self.off_coeff * rhs.off_coeff;
        let on_coeff = self.on_coeff * rhs.on_coeff;

        let k = if is_zero(on_coeff) {
            Self::identity_on(self.on_bits, self.n)
        } else {
            self.k.gp(rhs.k)?
        };

        Some(Self {
            off_bits: self.off_bits,
            on_bits: self.on_bits,
            off_coeff,
            on_coeff,
            k,
            n: self.n,
        })
    }

    pub fn scale(mut self, scalar: f64) -> Self {
        self.off_coeff *= scalar;
        self.on_coeff *= scalar;
        self
    }

    pub fn to_mv(self) -> Multivector {
        let off = Sector::new(self.off_bits, self.n);
        let on = Sector::new(self.on_bits, self.n);

        let off_term = Term {
            left: Some(off),
            blade: Blade::identity(),
            right: Some(off),
            rotor: None,
            coeff: self.off_coeff,
        };

        let (left, blade, right, rotor, sign) = self.k.into_parts();

        let on_term = Term {
            left: Some(left),
            blade: Blade::new(blade.x, blade.z, sign),
            right: Some(right),
            rotor: if (rotor.q1_mask | rotor.q2_mask | rotor.q3_mask) == 0 {
                None
            } else {
                Some(rotor)
            },
            coeff: self.on_coeff,
        };

        let mut terms = Vec::new();

        if !is_zero(self.off_coeff) {
            terms.push(off_term);
        }

        if !is_zero(self.on_coeff) {
            terms.push(on_term);
        }

        // Keep `on` referenced to make the intended branch explicit.
        let _ = on;

        Multivector::from_terms(self.n, terms)
    }

    pub fn try_from_mv(mv: &Multivector) -> Option<Self> {
        if mv.terms.len() != 2 {
            return None;
        }

        let n = mv.n;
        let t0 = &mv.terms[0];
        let t1 = &mv.terms[1];

        let classify = |t: &Term| -> Option<(bool, u64)> {
            let left = t.left?;
            let right = t.right?;

            if left.n != n || right.n != n {
                return None;
            }

            if left != right {
                return None;
            }

            let is_identity_payload = t.blade == Blade::identity() && t.rotor.is_none();

            Some((is_identity_payload, left.bits))
        };

        let c0 = classify(t0)?;
        let c1 = classify(t1)?;

        let (off_term, on_term) = match (c0.0, c1.0) {
            (true, false) => (t0, t1),
            (false, true) => (t1, t0),
            _ => return None,
        };

        let off = off_term.left?;
        let on = on_term.left?;

        if off == on || off.n != n || on.n != n {
            return None;
        }

        let rotor = on_term.rotor.unwrap_or(Rotor {
            q1_mask: 0,
            q2_mask: 0,
            q3_mask: 0,
            sign: false,
        });

        let k = PackedBlockTerm::new(
            on.bits,
            on_term.blade.x,
            on_term.blade.z,
            on.bits,
            rotor.q1_mask,
            rotor.q2_mask,
            rotor.q3_mask,
            on_term.blade.sign ^ rotor.sign,
            n,
        );

        Self::new(off.bits, on.bits, off_term.coeff, on_term.coeff, k, n)
    }
}