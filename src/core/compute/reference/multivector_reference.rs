use crate::core::compute::multivector::combine_like_terms;
use crate::core::compute::reference::term_reference::gp_term_reference;
use crate::core::ir::{Multivector, Term};

const EPS: f64 = 1e-12;

#[inline(always)]
fn is_effectively_zero(x: f64) -> bool {
    x.abs() < EPS
}

/// Slow reference geometric product of two multivectors.
///
/// This uses the slow/reference term kernel and then combines like terms.
/// It is intended for:
/// - correctness checks
/// - regression tests
/// - benchmarks against the fast packed kernel
pub fn gp_mv_reference(a: &Multivector, b: &Multivector) -> Multivector {
    if a.terms.is_empty() || b.terms.is_empty() {
        return Multivector::from_terms(a.n, vec![]);
    }

    let mut out_terms: Vec<Term> = Vec::new();

    for ta in &a.terms {
        if is_effectively_zero(ta.coeff) {
            continue;
        }

        for tb in &b.terms {
            if is_effectively_zero(tb.coeff) {
                continue;
            }

            if let Some(mut prod) = gp_term_reference(ta, tb) {
                prod.coeff = ta.coeff * tb.coeff;
                if !is_effectively_zero(prod.coeff) {
                    out_terms.push(prod);
                }
            }
        }
    }

    Multivector::from_terms(a.n, combine_like_terms(out_terms))
}