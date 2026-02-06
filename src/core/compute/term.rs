use crate::core::bits::{Blade, Rotor, Sector};
use crate::core::compute::local::PackedBlockTerm;
use crate::core::ir::Term;

/// Convert IR term → packed compute term.
///
/// Rules:
/// - missing sectors → all-P sector
/// - missing rotor → identity
/// - blade + rotor sign combined
#[inline(always)]
pub fn term_to_packed(term: &Term, n: u8) -> PackedBlockTerm {
    let left = term.left.unwrap_or_else(|| Sector::new(0, n));
    let right = term.right.unwrap_or_else(|| Sector::new(0, n));

    let rotor = term.rotor.unwrap_or(Rotor {
        q1_mask: 0,
        q2_mask: 0,
        q3_mask: 0,
        sign: false,
    });

    PackedBlockTerm::new(
        left.bits,
        term.blade.x,
        term.blade.z,
        right.bits,
        rotor.q1_mask,
        rotor.q2_mask,
        rotor.q3_mask,
        term.blade.sign ^ rotor.sign,
        n,
    )
}

/// Convert packed term → IR term.
///
/// Note:
/// - does NOT restore coefficient (handled externally)
/// - sectors == all-P become None
#[inline(always)]
pub fn packed_to_term(p: PackedBlockTerm) -> Term {
    let left = if p.left_bits == 0 {
        None
    } else {
        Some(Sector::new(p.left_bits, p.n))
    };

    let right = if p.right_bits == 0 {
        None
    } else {
        Some(Sector::new(p.right_bits, p.n))
    };

    let blade = Blade::new(p.blade_x, p.blade_z, p.sign);

    let has_rotor = (p.rotor_q1 | p.rotor_q2 | p.rotor_q3) != 0;

    let rotor = if has_rotor {
        Some(Rotor {
            q1_mask: p.rotor_q1,
            q2_mask: p.rotor_q2,
            q3_mask: p.rotor_q3,
            sign: false,
        })
    } else {
        None
    };

    Term {
        left,
        blade,
        right,
        rotor,
        coeff: 1.0,
    }
}

/// Fallback geometric product at the Term level.
///
/// This is NOT used in fast paths.
/// Production code should route through:
/// - ComputeOp (structured)
/// - PackedMultivector (generic)
///
/// This exists for:
/// - debugging
/// - unit tests
/// - correctness reference
///
/// Takes references to avoid ownership issues, requires explicit n parameter.
/// Preserves coefficients from both input terms.
#[inline]
pub fn gp_term(a: &Term, b: &Term, n: u8) -> Option<Term> {
    // Convert to packed representation
    let pa = term_to_packed(a, n);
    let pb = term_to_packed(b, n);

    // Perform packed GP
    let out = pa.gp(pb)?;

    // Convert back to IR
    let mut t = packed_to_term(out);

    // Combine coefficients
    t.coeff = a.coeff * b.coeff;

    Some(t)
}

/// Canonicalize a term by normalizing its representation.
/// Currently returns the term as-is (always Some); can be extended to filter out zero terms.
#[inline(always)]
pub fn canonicalize_term(term: Term) -> Option<Term> {
    Some(term)
}

/// Normalize a term in Peirce block form (with explicit left/right sectors).
/// This is an alias for canonicalize_term; can be extended to add Peirce-specific normalizations.
#[inline(always)]
pub fn normalize_term_peirce(term: Term) -> Option<Term> {
    canonicalize_term(term)
}