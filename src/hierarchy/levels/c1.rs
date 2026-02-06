use crate::core::bits::{Blade, Rotor};
use crate::core::compute::blade::{anticommutes, commutes, grade, support_size};
use crate::core::ir::{Multivector, Term};

/// Level-C1 utilities: Pauli / blade-level recognition and invariants.
///
/// In this engine, a packed `Blade` is the natural Pauli-word representation:
///
/// - z-bit  = e1 / Z-like factor
/// - x-bit  = e2 / X-like factor
/// - xz-bit = J / local bivector factor
///
/// The Y gate is represented as a biaction:
///
///     Y(ψ) = -J ψ J
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum C1Gate {
    X { qubit: usize },
    Y { qubit: usize },
    Z { qubit: usize },
}

pub struct C1;

impl C1 {
    // ============================================================
    // Gate descriptors
    // ============================================================

    #[inline(always)]
    pub fn x(qubit: usize) -> C1Gate {
        C1Gate::X { qubit }
    }

    #[inline(always)]
    pub fn y(qubit: usize) -> C1Gate {
        C1Gate::Y { qubit }
    }

    #[inline(always)]
    pub fn z(qubit: usize) -> C1Gate {
        C1Gate::Z { qubit }
    }

    // ============================================================
    // Operator constructors
    // ============================================================

    /// Bare X operator as a multivector.
    pub fn x_gate(n: u8, qubit: usize) -> Multivector {
        debug_assert!(qubit < n as usize);

        Multivector::from_terms(
            n,
            vec![Self::pauli_term(Blade::x(qubit), 1.0)],
        )
    }

    /// Y as a biaction:
    ///
    ///     Y(ψ) = -J ψ J
    pub fn y_gate(n: u8, qubit: usize) -> Multivector {
        debug_assert!(qubit < n as usize);

        Multivector::from_terms(
            n,
            vec![Term {
                left: None,
                blade: Blade::j(qubit),
                right: None,
                rotor: Some(Rotor::j_at(qubit)),
                coeff: -1.0,
            }],
        )
    }

    /// Bare Z operator as a multivector.
    pub fn z_gate(n: u8, qubit: usize) -> Multivector {
        debug_assert!(qubit < n as usize);

        Multivector::from_terms(
            n,
            vec![Self::pauli_term(Blade::z(qubit), 1.0)],
        )
    }

    /// Bare J bivector operator as a multivector.
    pub fn j_gate(n: u8, qubit: usize) -> Multivector {
        debug_assert!(qubit < n as usize);

        Multivector::from_terms(
            n,
            vec![Self::pauli_term(Blade::j(qubit), 1.0)],
        )
    }

    // ============================================================
    // Blade-level recognition
    // ============================================================

    /// Convert a packed single-qubit Pauli blade into a C1Gate, if possible.
    ///
    /// Recognizes only single-site Pauli words X_i, Y_i, Z_i.
    pub fn get_gate(blade: Blade) -> Option<C1Gate> {
        if support_size(blade) != 1 {
            return None;
        }

        let q = (blade.x | blade.z).trailing_zeros() as usize;
        let bit = 1u64 << q;

        let xi = (blade.x & bit) != 0;
        let zi = (blade.z & bit) != 0;

        match (xi, zi) {
            (true, false) => Some(C1Gate::X { qubit: q }),
            (false, true) => Some(C1Gate::Z { qubit: q }),
            (true, true) => Some(C1Gate::Y { qubit: q }),
            _ => None,
        }
    }

    #[inline(always)]
    pub fn is_pauli_blade(_b: Blade) -> bool {
        true
    }

    // ============================================================
    // Multivector-level predicates
    // ============================================================

    /// Predicate: is this a single bare Pauli-like term?
    ///
    /// Conditions:
    /// - exactly one term
    /// - no explicit Peirce sectors
    /// - trivial or absent right rotor
    /// - unit coefficient magnitude
    pub fn is_atomic_pauli(mv: &Multivector) -> bool {
        if mv.terms.len() != 1 {
            return false;
        }

        let t = &mv.terms[0];

        t.left.is_none()
            && t.right.is_none()
            && t.rotor.map(|r| r.is_identity()).unwrap_or(true)
            && (t.coeff.abs() - 1.0).abs() < 1e-9
    }

    pub fn as_pauli_blade(mv: &Multivector) -> Option<Blade> {
        if Self::is_atomic_pauli(mv) {
            Some(mv.terms[0].blade)
        } else {
            None
        }
    }

    pub fn all_terms_pairwise_commuting(mv: &Multivector) -> bool {
        for i in 0..mv.terms.len() {
            for j in (i + 1)..mv.terms.len() {
                if !commutes(mv.terms[i].blade, mv.terms[j].blade) {
                    return false;
                }
            }
        }

        true
    }

    pub fn blades(mv: &Multivector) -> Vec<Blade> {
        mv.terms.iter().map(|t| t.blade).collect()
    }

    // ============================================================
    // Blade properties
    // ============================================================

    #[inline(always)]
    pub fn pauli_weight(b: Blade) -> u32 {
        support_size(b)
    }

    #[inline(always)]
    pub fn blade_grade(b: Blade) -> u32 {
        grade(b)
    }

    #[inline(always)]
    pub fn commutes(a: Blade, b: Blade) -> bool {
        commutes(a, b)
    }

    #[inline(always)]
    pub fn anticommutes(a: Blade, b: Blade) -> bool {
        anticommutes(a, b)
    }

    // ============================================================
    // Term-level helpers
    // ============================================================

    /// Create a bare Pauli term from a blade.
    #[inline(always)]
    pub fn pauli_term(blade: Blade, coeff: f64) -> Term {
        Term {
            left: None,
            blade,
            right: None,
            rotor: Some(Rotor::identity()),
            coeff,
        }
    }

    /// Predicate: is this term bare, i.e. no sectors and identity rotor?
    #[inline(always)]
    pub fn is_bare_pauli_term(term: &Term) -> bool {
        term.left.is_none()
            && term.right.is_none()
            && term.rotor.map(|r| r.is_identity()).unwrap_or(true)
    }
}