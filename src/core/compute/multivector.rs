use crate::core::bits::Blade;
use crate::core::compute::structured::{
    add_mv_structured,
    gp_mv_structured,
    scale_mv_structured,
    sub_mv_structured,
};
use crate::core::ir::{Multivector, Term};

/// ============================================================
/// COMPUTE MULTIVECTOR API
/// ============================================================
///
/// This is the public execution layer.
/// - route operations through structured dispatcher
/// - preserve specialized representations when possible
/// - remain independent of packed layout
///
/// ============================================================

#[inline]
pub fn gp_mv(a: &Multivector, b: &Multivector) -> Multivector {
    gp_mv_structured(a, b)
}

#[inline]
pub fn scale_mv(mv: &Multivector, scalar: f64) -> Multivector {
    scale_mv_structured(mv, scalar)
}

#[inline]
pub fn add_mv(a: &Multivector, b: &Multivector) -> Multivector {
    add_mv_structured(a, b)
}

#[inline]
pub fn sub_mv(a: &Multivector, b: &Multivector) -> Multivector {
    sub_mv_structured(a, b)
}

#[inline]
pub fn gp_mv_scaled(a: &Multivector, b: &Multivector, scalar: f64) -> Multivector {
    scale_mv(&gp_mv(a, b), scalar)
}

/// ============================================================
/// LIGHTWEIGHT IR UTILITIES
/// ============================================================

/// Extract scalar coefficient (NOT trace)
///
/// Returns coefficient of identity operator only.
#[inline]
pub fn scalar_component(mv: &Multivector) -> f64 {
    mv.terms
        .iter()
        .filter(|t| {
            t.left.is_none()
                && t.right.is_none()
                && t.blade == Blade::identity()
                && t.rotor.is_none()
        })
        .map(|t| t.coeff)
        .sum()
}

/// Alias for scalar_component: extract the scalar (identity) part of a multivector.
#[inline]
pub fn scalar_part(mv: &Multivector) -> f64 {
    scalar_component(mv)
}

/// Trace of a multivector in Cl_{2,0}^{⊗ n}.
///
/// Contributions come from:
/// - diagonal Peirce blocks (left == right)
/// - scalar blade only
/// - rotor scalar contribution
///
/// NOTE:
/// This is NOT optimized yet.
/// Later, structured compute will override this.
pub fn trace_mv(mv: &Multivector) -> f64 {
    let n = mv.n;
    let dim = 1u64 << n;

    mv.terms
        .iter()
        .filter(|t| t.left == t.right)
        .filter_map(|t| {
            if t.blade != Blade::identity() {
                return None;
            }

            let phase = match &t.rotor {
                None => 1.0,
                Some(_r) => 1.0, // TODO: rotor scalar extraction
            };

            Some(t.coeff * phase)
        })
        .sum::<f64>()
        * (dim as f64)
}

/// Combine identical IR terms (non-hot path)
pub fn combine_like_terms(terms: Vec<Term>) -> Vec<Term> {
    use std::collections::HashMap;

    let mut map: HashMap<(Option<_>, _, Option<_>, Option<_>), f64> = HashMap::new();

    for t in terms {
        let key = (t.left, t.blade, t.right, t.rotor);
        *map.entry(key).or_insert(0.0) += t.coeff;
    }

    map.into_iter()
        .filter(|(_, coeff)| *coeff != 0.0)
        .map(|((left, blade, right, rotor), coeff)| Term {
            left,
            blade,
            right,
            rotor,
            coeff,
        })
        .collect()
}

/// Simplify a multivector by combining like terms.
///
/// This removes zero-coefficient terms and merges terms with identical
/// (left, blade, right, rotor) signatures.
#[inline]
pub fn simplify_mv(mv: Multivector) -> Multivector {
    let combined = combine_like_terms(mv.terms);
    Multivector::from_terms(mv.n, combined)
}