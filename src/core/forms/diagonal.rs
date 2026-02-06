use crate::core::bits::{Blade, Sector};
use crate::core::ir::{Multivector, Term};

const EPS: f64 = 1e-12;

#[inline(always)]
fn is_zero(x: f64) -> bool {
    x.abs() < EPS
}

/// Diagonal Peirce-sector normal form:
///
///     D = Σ_x coeff[x] Π_x
///
/// Current version is scalar-diagonal only.
/// Rotor/phase-polynomial support can be added later.
#[derive(Clone, Debug, PartialEq)]
pub struct DiagonalPacked {
    pub coeffs: Vec<f64>,
    pub n: u8,
}

impl DiagonalPacked {
    #[inline(always)]
    pub fn dim(n: u8) -> usize {
        1usize << n
    }

    #[inline]
    pub fn identity(n: u8) -> Self {
        Self {
            coeffs: vec![1.0; Self::dim(n)],
            n,
        }
    }

    #[inline]
    pub fn zero(n: u8) -> Self {
        Self {
            coeffs: vec![0.0; Self::dim(n)],
            n,
        }
    }

    #[inline(always)]
    pub fn coeff_of(&self, bits: u64) -> f64 {
        self.coeffs[bits as usize]
    }

    #[inline(always)]
    pub fn set_coeff(&mut self, bits: u64, coeff: f64) {
        self.coeffs[bits as usize] = coeff;
    }

    #[inline]
    pub fn scale(mut self, scalar: f64) -> Self {
        for c in &mut self.coeffs {
            *c *= scalar;
        }
        self
    }

    #[inline]
    pub fn gp(&self, rhs: &Self) -> Option<Self> {
        if self.n != rhs.n {
            return None;
        }

        let coeffs = self
            .coeffs
            .iter()
            .zip(rhs.coeffs.iter())
            .map(|(a, b)| a * b)
            .collect();

        Some(Self {
            coeffs,
            n: self.n,
        })
    }

    #[inline]
    pub fn add(&self, rhs: &Self) -> Option<Self> {
        if self.n != rhs.n {
            return None;
        }

        let coeffs = self
            .coeffs
            .iter()
            .zip(rhs.coeffs.iter())
            .map(|(a, b)| a + b)
            .collect();

        Some(Self {
            coeffs,
            n: self.n,
        })
    }

    #[inline]
    pub fn trace(&self) -> f64 {
        self.coeffs.iter().sum()
    }

    #[inline]
    pub fn z(n: u8, q: usize) -> Self {
        let dim = Self::dim(n);
        let mut coeffs = vec![1.0; dim];

        for x in 0..dim {
            if ((x >> q) & 1) == 1 {
                coeffs[x] = -1.0;
            }
        }

        Self { coeffs, n }
    }

    /// Scalar placeholder for T until rotor/complex phase diagonal is added.
    ///
    /// This is structurally useful, but not yet a physically complete T phase.
    #[inline]
    pub fn t(n: u8, q: usize) -> Self {
        let dim = Self::dim(n);
        let mut coeffs = vec![1.0; dim];

        for x in 0..dim {
            if ((x >> q) & 1) == 1 {
                coeffs[x] = std::f64::consts::FRAC_1_SQRT_2;
            }
        }

        Self { coeffs, n }
    }

    #[inline]
    pub fn s(n: u8, q: usize) -> Self {
        let dim = Self::dim(n);
        let mut coeffs = vec![1.0; dim];

        for x in 0..dim {
            if ((x >> q) & 1) == 1 {
                // Placeholder until rotor-aware diagonal phase.
                coeffs[x] = 0.0;
            }
        }

        Self { coeffs, n }
    }

    pub fn to_mv(&self) -> Multivector {
        let mut terms = Vec::new();

        for (bits, coeff) in self.coeffs.iter().enumerate() {
            if is_zero(*coeff) {
                continue;
            }

            let s = Sector::new(bits as u64, self.n);

            terms.push(Term {
                left: Some(s),
                blade: Blade::identity(),
                right: Some(s),
                rotor: None,
                coeff: *coeff,
            });
        }

        Multivector::from_terms(self.n, terms)
    }

    pub fn try_from_mv(mv: &Multivector) -> Option<Self> {
        let n = mv.n;
        let mut out = Self::zero(n);

        for t in &mv.terms {
            if is_zero(t.coeff) {
                continue;
            }

            let left = t.left.unwrap_or_else(|| Sector::new(0, n));
            let right = t.right.unwrap_or_else(|| Sector::new(0, n));

            if left.n != n || right.n != n {
                return None;
            }

            if left != right {
                return None;
            }

            if t.blade != Blade::identity() {
                return None;
            }

            if t.rotor.is_some() {
                return None;
            }

            out.coeffs[left.bits as usize] += t.coeff;
        }

        Some(out)
    }
}