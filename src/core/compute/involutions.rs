use crate::core::bits::Blade;
use crate::core::compute::blade::grade;
use crate::core::compute::multivector::simplify_mv;
use crate::core::compute::term::canonicalize_term;
use crate::core::ir::{Multivector, Term};

/// Computes the sign parity change from reversion on a blade of grade `k`.
///
/// Reversion is the main involution in geometric algebra. When reversing the order of
/// basis vectors in a k-vector (blade), the sign changes by (-1)^{k(k-1)/2}.
///
/// This formula comes from the number of transpositions needed to reverse a sequence:
/// - k=0: 0 transpositions → sign: +1
/// - k=1: 0 transpositions → sign: +1  
/// - k=2: 1 transposition → sign: -1
/// - k=3: 3 transpositions → sign: -1
/// - k=4: 6 transpositions → sign: +1
#[inline(always)]
fn reverse_sign_parity(k: u32) -> bool {
    // Reversion contributes (-1)^{k(k-1)/2}.
    // For k = 0 or 1 this is always +1, so avoid underflow on (k - 1).
    if k < 2 {
        false
    } else {
        (((k * (k - 1)) / 2) & 1) != 0
    }
}

/// Applies reversion automorphism to a single blade.
///
/// Reversion reverses the order of all basis vectors in the blade. In a bitwise representation,
/// the basis elements (stored in `x` and `z` registers) are not reordered, but the sign bit
/// is updated to account for the commutation parity.
///
/// For a blade with k basis vectors, the sign changes by (-1)^{k(k-1)/2}.
#[inline(always)]
pub fn reverse_blade(b: Blade) -> Blade {
    let k = grade(b);
    let reverse_parity = reverse_sign_parity(k);

    Blade {
        x: b.x,
        z: b.z,
        sign: b.sign ^ reverse_parity,
    }
}

/// Applies grade involution automorphism to a single blade.
///
/// Grade involution negates all odd-grade blades while leaving even-grade blades unchanged.
/// It is the automorphism defined by negating all basis generators (eᵢ → -eᵢ).
/// 
/// Sign change rule:
/// - Even-grade blades (k=0,2,4,...): No sign change, (-1)^k = +1
/// - Odd-grade blades (k=1,3,5,...): Sign flips, (-1)^k = -1
#[inline(always)]
pub fn grade_involution_blade(b: Blade) -> Blade {
    // Check if grade is odd by testing the least significant bit
    let odd = (grade(b) & 1) != 0;

    Blade {
        x: b.x,
        z: b.z,
        sign: b.sign ^ odd,
    }
}

/// Applies reversion automorphism to a multivector.
///
/// Reversion is applied term-by-term: each blade within the multivector is reversed while
/// preserving the term's left/right gate structure and coefficient.
///
/// The process:
/// 1. Allocate output vector with same capacity as input
/// 2. For each term, reverse its blade using `reverse_blade()`
/// 3. Canonicalize and simplify the resulting term (combines like terms)
/// 4. Simplify the overall multivector to merge any duplicate terms
pub fn reverse_mv(mv: &Multivector) -> Multivector {
    let mut out = Vec::with_capacity(mv.terms.len());

    for t in &mv.terms {
        // Apply reverse_blade to the blade component of each term
        if let Some(term) = canonicalize_term(Term {
            left: t.left,
            blade: reverse_blade(t.blade),
            right: t.right,
            rotor: t.rotor,
            coeff: t.coeff,
        }) {
            out.push(term);
        }
    }

    // Combine and simplify all terms (handles repeated basis combinations)
    simplify_mv(Multivector::from_terms(mv.n, out))
}

/// Applies grade involution automorphism to a multivector.
///
/// Grade involution is applied term-by-term: each blade within the multivector has its
/// grade involution computed while preserving the term's left/right gate structure and coefficient.
///
/// The process:
/// 1. Allocate output vector with same capacity as input
/// 2. For each term, apply grade involution to its blade using `grade_involution_blade()`
/// 3. Canonicalize and simplify the resulting term
/// 4. Simplify the overall multivector to merge any duplicate terms
///
/// Grade involution negates odd-grade blades, leaving even-grade blades unchanged.
/// Used in quantum state transformations and Clifford operator calculations.
pub fn grade_involution_mv(mv: &Multivector) -> Multivector {
    let mut out = Vec::with_capacity(mv.terms.len());

    for t in &mv.terms {
        // Apply grade_involution_blade to the blade component of each term
        if let Some(term) = canonicalize_term(Term {
            left: t.left,
            blade: grade_involution_blade(t.blade),
            right: t.right,
            rotor: t.rotor,
            coeff: t.coeff,
        }) {
            out.push(term);
        }
    }

    // Combine and simplify all terms (handles repeated basis combinations)
    simplify_mv(Multivector::from_terms(mv.n, out))
}