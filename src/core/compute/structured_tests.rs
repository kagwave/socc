#[cfg(test)]
mod tests {
    use crate::core::bits::{Blade, Rotor, Sector};
    use crate::core::compute::multivector_packed::PackedMultivector;
    use crate::core::compute::structured::{ComputeOp, gp_compute, gp_mv_structured};
    use crate::core::ir::{Multivector, Term};

    #[test]
    fn lower_prefers_controlled_when_recognizable() {
        let n = 1;
        let off = Sector::new(0, n);
        let on = Sector::new(1, n);

        let mv = Multivector::from_terms(
            n,
            vec![
                Term {
                    left: Some(off),
                    blade: Blade::identity(),
                    right: Some(off),
                    rotor: None,
                    coeff: 1.0,
                },
                Term {
                    left: Some(on),
                    blade: Blade::x(0),
                    right: Some(on),
                    rotor: None,
                    coeff: 1.0,
                },
            ],
        );

        match ComputeOp::lower_mv(&mv) {
            ComputeOp::Controlled(_) => {}
            ComputeOp::Generic(_) => panic!("expected controlled lowering"),
            _ => panic!("expected controlled lowering"),
        }
    }

    #[test]
    fn lower_falls_back_to_generic_when_not_recognizable() {
        let n = 1;

        let mv = Multivector::from_terms(
            n,
            vec![
                Term {
                    left: None,
                    blade: Blade::x(0),
                    right: None,
                    rotor: None,
                    coeff: 1.0,
                },
                Term {
                    left: None,
                    blade: Blade::z(0),
                    right: None,
                    rotor: None,
                    coeff: 1.0,
                },
                Term {
                    left: None,
                    blade: Blade::identity(),
                    right: None,
                    rotor: Some(Rotor {
                        q1_mask: 1,
                        q2_mask: 0,
                        q3_mask: 0,
                        sign: false,
                    }),
                    coeff: 1.0,
                },
            ],
        );

        match ComputeOp::lower_mv(&mv) {
            ComputeOp::Controlled(_) => panic!("expected generic lowering"),
            ComputeOp::Generic(_) => {}
            _ => panic!("expected generic lowering"),
        }
    }

    #[test]
    fn structured_multiply_matches_generic_term_count() {
        let n = 1;
        let off = Sector::new(0, n);
        let on = Sector::new(1, n);

        let a = Multivector::from_terms(
            n,
            vec![
                Term {
                    left: Some(off),
                    blade: Blade::identity(),
                    right: Some(off),
                    rotor: None,
                    coeff: 1.0,
                },
                Term {
                    left: Some(on),
                    blade: Blade::x(0),
                    right: Some(on),
                    rotor: None,
                    coeff: 1.0,
                },
            ],
        );

        let b = Multivector::from_terms(
            n,
            vec![
                Term {
                    left: Some(off),
                    blade: Blade::identity(),
                    right: Some(off),
                    rotor: None,
                    coeff: 1.0,
                },
                Term {
                    left: Some(on),
                    blade: Blade::identity(),
                    right: Some(on),
                    rotor: Some(Rotor {
                        q1_mask: 1,
                        q2_mask: 0,
                        q3_mask: 0,
                        sign: false,
                    }),
                    coeff: 1.0,
                },
            ],
        );

        let structured = gp_mv_structured(&a, &b);

        let ga = PackedMultivector::from_mv(&a);
        let gb = PackedMultivector::from_mv(&b);
        let generic = PackedMultivector::gp(&ga, &gb).to_mv();

        assert_eq!(structured.terms.len(), generic.terms.len());
    }

    #[test]
    fn diagonal_left_monomial_preserves_perm_and_scales_output_sector() {
        let n = 2;
        let d = crate::core::forms::diagonal::DiagonalPacked::z(n, 0);
        let m = crate::core::forms::monomial::MonomialPacked::cnot(n, 0, 1);

        let out = match crate::core::compute::structured::gp_compute(
            &crate::core::compute::structured::ComputeOp::Diagonal(d.clone()),
            &crate::core::compute::structured::ComputeOp::Monomial(m.clone()),
        ) {
            crate::core::compute::structured::ComputeOp::Monomial(out) => out,
            _ => panic!("expected monomial result"),
        };

        assert_eq!(out.perm, m.perm);

        for x in 0..out.coeffs.len() {
            assert_eq!(out.coeffs[x], m.coeffs[x] * d.coeff_of(out.perm[x]));
        }
    }

    #[test]
    fn monomial_right_diagonal_preserves_perm_and_scales_input_sector() {
        let n = 2;
        let m = crate::core::forms::monomial::MonomialPacked::cnot(n, 0, 1);
        let d = crate::core::forms::diagonal::DiagonalPacked::s(n, 1);

        let out = match crate::core::compute::structured::gp_compute(
            &crate::core::compute::structured::ComputeOp::Monomial(m.clone()),
            &crate::core::compute::structured::ComputeOp::Diagonal(d.clone()),
        ) {
            crate::core::compute::structured::ComputeOp::Monomial(out) => out,
            _ => panic!("expected monomial result"),
        };

        assert_eq!(out.perm, m.perm);

        for x in 0..out.coeffs.len() {
            assert_eq!(out.coeffs[x], m.coeffs[x] * d.coeff_of(x as u64));
        }
    }
}