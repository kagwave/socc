use crate::core::bits::{Blade, Rotor, Sector};

/// High-level IR term.
///
/// Represents:
///
///     coeff * Π_left * B * Π_right * R
///
/// where any component may be absent.
///
/// This is the flexible, user-facing representation.
/// It is NOT guaranteed to be canonical.
#[derive(Clone, Debug, PartialEq)]
pub struct Term {
    pub left: Option<Sector>,
    pub blade: Blade,
    pub right: Option<Sector>,
    pub rotor: Option<Rotor>,
    pub coeff: f64,
}

impl Term {
    /// Create a pure blade term (no sectors, no rotor)
    #[inline]
    pub fn blade(blade: Blade, coeff: f64) -> Self {
        Self {
            left: None,
            blade,
            right: None,
            rotor: None,
            coeff,
        }
    }

    /// Create a Peirce block term
    #[inline]
    pub fn peirce(
        left: Sector,
        blade: Blade,
        right: Sector,
        rotor: Option<Rotor>,
        coeff: f64,
    ) -> Self {
        Self {
            left: Some(left),
            blade,
            right: Some(right),
            rotor,
            coeff,
        }
    }
}