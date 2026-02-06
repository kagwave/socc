use crate::core::bits::{Blade, Sector};
use crate::core::ir::Term;

/// High-level IR multivector.
///
/// Represents a finite sum of SOCC terms in:
///
///     Cl_{2,0}^{⊗ n}
///
/// The field `n` is part of the algebraic context. It is not just metadata:
/// identity, trace, sectors, measurement, and packing all depend on knowing
/// the ambient number of qubits.
#[derive(Clone, Debug, PartialEq)]
pub struct Multivector {
    pub terms: Vec<Term>,
    pub n: u8,
}

impl Multivector {
    /// Empty multivector in an explicit n-qubit ambient algebra.
    #[inline(always)]
    pub fn new(n: u8) -> Self {
        Self {
            terms: Vec::new(),
            n,
        }
    }

    /// Build from terms in an explicit n-qubit ambient algebra.
    ///
    /// This is the preferred constructor. Do not infer `n` in hot paths.
    #[inline(always)]
    pub fn from_terms(n: u8, terms: Vec<Term>) -> Self {
        Self { terms, n }
    }

    /// Build a one-term multivector.
    #[inline(always)]
    pub fn singleton(n: u8, term: Term) -> Self {
        Self {
            terms: vec![term],
            n,
        }
    }

    /// Pure blade multivector:
    ///
    ///     coeff * B
    ///
    /// No Peirce sectors are attached.
    #[inline(always)]
    pub fn from_blade(n: u8, blade: Blade, coeff: f64) -> Self {
        Self::singleton(n, Term::blade(blade, coeff))
    }

    /// Create the identity operator (coefficient 1.0) for n qubits.
    #[inline]
    pub fn identity(n: u8) -> Self {
        Self::from_blade(n, Blade::identity(), 1.0)
    }

    /// Peirce sector-map term:
    ///
    ///     coeff * Π_left * B * Π_right
    ///
    /// This is one sparse matrix block over Peirce sectors.
    #[inline(always)]
    pub fn from_sector_map(
        n: u8,
        left: Sector,
        blade: Blade,
        right: Sector,
        coeff: f64,
    ) -> Self {
        debug_assert_eq!(left.n, n);
        debug_assert_eq!(right.n, n);

        Self::singleton(n, Term::peirce(left, blade, right, None, coeff))
    }

    /// Push a term into the multivector.
    ///
    /// This does not simplify or canonicalize. That belongs either in IR
    /// simplification or compute lowering.
    #[inline(always)]
    pub fn push(&mut self, term: Term) {
        self.terms.push(term);
    }

    /// Number of stored terms.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.terms.len()
    }

    /// Whether no terms are stored.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }
}

impl Default for Multivector {
    /// Default is the zero multivector in the zero-qubit algebra.
    ///
    /// Prefer `Multivector::new(n)` in real code.
    #[inline(always)]
    fn default() -> Self {
        Self {
            terms: Vec::new(),
            n: 0,
        }
    }
}