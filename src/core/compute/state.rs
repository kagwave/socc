use crate::core::bits::Sector;
use crate::core::compute::multivector::simplify_mv;
use crate::core::compute::term::normalize_term_peirce;
use crate::core::ir::{Multivector, Term};

/// Apply a multivector/operator to a right exact sector:
///
///     A Π_x
///
/// This keeps the sector exact and uses packed Peirce normalization
/// instead of expanding Π_x as (1 ± e1)/2.
///
/// Semantics:
/// - if a term has no explicit right sector, attach `sector`
/// - if a term already has a right sector, it must match `sector`
///   or the term vanishes
/// - after attaching the right sector, normalize using the
///   exact-sector/blade rewrite rules
pub fn apply_mv_to_right_sector(op: &Multivector, sector: Sector) -> Multivector {
    let mut out = Vec::with_capacity(op.terms.len());

    for t in &op.terms {
        let right = match t.right {
            None => Some(sector),
            Some(r) if r == sector => Some(r),
            Some(_) => continue,
        };

        let term = Term {
            left: t.left,
            blade: t.blade,
            right,
            rotor: t.rotor,
            coeff: t.coeff,
        };

        if let Some(norm) = normalize_term_peirce(term) {
            out.push(norm);
        }
    }

    simplify_mv(Multivector::from_terms(op.n, out))
}

/// Apply a multivector/operator to a left exact sector:
///
///     Π_x A
///
/// This is the left-handed analogue of `apply_mv_to_right_sector`.
pub fn apply_mv_to_left_sector(sector: Sector, op: &Multivector) -> Multivector {
    let mut out = Vec::with_capacity(op.terms.len());

    for t in &op.terms {
        let left = match t.left {
            None => Some(sector),
            Some(l) if l == sector => Some(l),
            Some(_) => continue,
        };

        let term = Term {
            left,
            blade: t.blade,
            right: t.right,
            rotor: t.rotor,
            coeff: t.coeff,
        };

        if let Some(norm) = normalize_term_peirce(term) {
            out.push(norm);
        }
    }

    simplify_mv(Multivector::from_terms(op.n, out))
}

/// Extract the exact Peirce block:
///
///     Π_y A Π_x
pub fn peirce_block(op: &Multivector, left: Sector, right: Sector) -> Multivector {
    let tmp = apply_mv_to_right_sector(op, right);
    let out = apply_mv_to_left_sector(left, &tmp);
    simplify_mv(out)
}

/// Convenience alias for the common SOCC pattern:
///
///     A P_n
pub fn apply_to_vacuum(op: &Multivector, vacuum: Sector) -> Multivector {
    apply_mv_to_right_sector(op, vacuum)
}

/// Check whether every surviving term is diagonal on the exact sector,
/// i.e. both left and right labels are that same sector.
pub fn diagonal_on_sector(op: &Multivector, sector: Sector) -> bool {
    op.terms
        .iter()
        .all(|t| t.left == Some(sector) && t.right == Some(sector))
}