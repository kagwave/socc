use std::collections::HashMap;

use crate::core::compute::local::PackedBlockTerm;
use crate::core::ir::Multivector;

const EPS: f64 = 1e-12;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PackedTermCoeff {
    pub term: PackedBlockTerm,
    pub coeff: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PackedMultivector {
    pub terms: Vec<PackedTermCoeff>,
    pub n: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct PackedTermKey {
    left_bits: u64,
    right_bits: u64,
    blade_x: u64,
    blade_z: u64,
    rotor_q1: u64,
    rotor_q2: u64,
    rotor_q3: u64,
    sign: bool,
    n: u8,
}

#[inline(always)]
fn is_zero(x: f64) -> bool {
    x.abs() < EPS
}

impl PackedMultivector {
    #[inline]
    pub fn new(n: u8, terms: Vec<PackedTermCoeff>) -> Self {
        Self { terms, n }
    }

    #[inline]
    pub fn empty(n: u8) -> Self {
        Self { terms: vec![], n }
    }

    #[inline]
    pub fn from_mv(mv: &Multivector) -> Self {
        let n = mv.n;

        let terms = mv
            .terms
            .iter()
            .filter(|t| !is_zero(t.coeff))
            .map(|t| PackedTermCoeff {
                term: super::lower::term_to_packed(t, n),
                coeff: t.coeff,
            })
            .collect();

        Self { terms, n }
    }

    #[inline]
    pub fn to_mv(&self) -> Multivector {
        let terms = self
            .terms
            .iter()
            .filter(|tc| !is_zero(tc.coeff))
            .map(|tc| {
                let mut t = super::lower::packed_to_term(tc.term);
                t.coeff = tc.coeff;
                t
            })
            .collect();

        Multivector::from_terms(self.n, terms)
    }

    #[inline]
    pub fn simplify(self) -> Self {
        let mut acc: HashMap<PackedTermKey, f64> = HashMap::new();

        for tc in self.terms {
            if is_zero(tc.coeff) {
                continue;
            }

            let key = PackedTermKey {
                left_bits: tc.term.left_bits,
                right_bits: tc.term.right_bits,
                blade_x: tc.term.blade_x,
                blade_z: tc.term.blade_z,
                rotor_q1: tc.term.rotor_q1,
                rotor_q2: tc.term.rotor_q2,
                rotor_q3: tc.term.rotor_q3,
                sign: tc.term.sign,
                n: tc.term.n,
            };

            *acc.entry(key).or_insert(0.0) += tc.coeff;
        }

        let terms = acc
            .into_iter()
            .filter_map(|(k, coeff)| {
                if is_zero(coeff) {
                    None
                } else {
                    Some(PackedTermCoeff {
                        term: PackedBlockTerm {
                            left_bits: k.left_bits,
                            right_bits: k.right_bits,
                            blade_x: k.blade_x,
                            blade_z: k.blade_z,
                            rotor_q1: k.rotor_q1,
                            rotor_q2: k.rotor_q2,
                            rotor_q3: k.rotor_q3,
                            sign: k.sign,
                            n: k.n,
                        },
                        coeff,
                    })
                }
            })
            .collect();

        Self { terms, n: self.n }
    }

    #[inline]
    pub fn gp(a: &Self, b: &Self) -> Self {
        if a.terms.is_empty() || b.terms.is_empty() {
            return Self::empty(a.n.max(b.n));
        }

        // If both multivectors are effectively unconstrained (all terms have left_bits=0 and right_bits=0),
        // use the reference implementation to avoid destroying blade structure via locally_reduce().
        // This handles cases like Pauli*Identity composition correctly.
        let a_unconstrained = a.terms.iter().all(|t| t.term.left_bits == 0 && t.term.right_bits == 0);
        let b_unconstrained = b.terms.iter().all(|t| t.term.left_bits == 0 && t.term.right_bits == 0);
        
        if a_unconstrained && b_unconstrained {
            // Both unconstrained: use reference path to preserve structure
            let a_mv = a.to_mv();
            let b_mv = b.to_mv();
            let result_mv = crate::core::compute::reference::multivector_reference::gp_mv_reference(&a_mv, &b_mv);
            return PackedMultivector::from_mv(&result_mv);
        }

        let mut out = Vec::new();

        for ta in &a.terms {
            if is_zero(ta.coeff) {
                continue;
            }

            for tb in &b.terms {
                if is_zero(tb.coeff) {
                    continue;
                }

                if let Some(prod) = ta.term.gp(tb.term) {
                    let coeff = ta.coeff * tb.coeff;
                    if !is_zero(coeff) {
                        out.push(PackedTermCoeff { term: prod, coeff });
                    }
                }
            }
        }

        Self { terms: out, n: a.n.max(b.n) }.simplify()
    }

    /// Scale a packed multivector by a scalar coefficient.
    #[inline]
    pub fn scale(pm: &Self, scalar: f64) -> Self {
        if is_zero(scalar) {
            return Self::empty(pm.n);
        }

        let terms = pm
            .terms
            .iter()
            .map(|tc| PackedTermCoeff {
                term: tc.term,
                coeff: tc.coeff * scalar,
            })
            .collect();

        Self { terms, n: pm.n }
    }

    /// Add two packed multivectors.
    #[inline]
    pub fn add(a: &Self, b: &Self) -> Self {
        let mut combined = a.terms.clone();
        combined.extend(b.terms.iter().cloned());

        Self {
            terms: combined,
            n: a.n.max(b.n),
        }
        .simplify()
    }
}