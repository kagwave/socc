#[cfg(test)]
mod tests {
    use crate::core::bits::{Blade, Rotor, Sector};
    use crate::core::compute::multivector::gp_mv;
    use crate::core::compute::reference::multivector_reference::gp_mv_reference;
    use crate::core::ir::{Multivector, Term};

    
// Kernel tests
    fn same_mv(a: &Multivector, b: &Multivector) -> bool {
        if a.terms.len() != b.terms.len() {
            return false;
        }

        a.terms.iter().all(|ta| {
            b.terms.iter().any(|tb| {
                ta.left == tb.left
                    && ta.blade == tb.blade
                    && ta.right == tb.right
                    && ta.rotor == tb.rotor
                    && (ta.coeff - tb.coeff).abs() < 1e-12
            })
        })
    }

    #[test]
    fn fast_and_reference_agree_on_simple_blade_mv() {
        let a = Multivector::from_terms(vec![Term {
            left: None,
            blade: Blade::x(0),
            right: None,
            rotor: None,
            coeff: 2.0,
        }]);

        let b = Multivector::from_terms(vec![Term {
            left: None,
            blade: Blade::z(0),
            right: None,
            rotor: None,
            coeff: 3.0,
        }]);

        let fast = gp_mv(&a, &b);
        let slow = gp_mv_reference(&a, &b);

        assert!(same_mv(&fast, &slow));
    }

    #[test]
    fn fast_and_reference_agree_on_sector_pruning() {
        let a = Multivector::from_terms(vec![Term {
            left: Some(Sector::new(0, 1)),
            blade: Blade::identity(),
            right: Some(Sector::new(0, 1)),
            rotor: None,
            coeff: 1.0,
        }]);

        let b = Multivector::from_terms(vec![Term {
            left: Some(Sector::new(1, 1)),
            blade: Blade::identity(),
            right: Some(Sector::new(0, 1)),
            rotor: None,
            coeff: 1.0,
        }]);

        let fast = gp_mv(&a, &b);
        let slow = gp_mv_reference(&a, &b);

        assert!(same_mv(&fast, &slow));
    }

    #[test]
    fn fast_and_reference_agree_on_rotor_payloads() {
        let a = Multivector::from_terms(vec![Term {
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
        }]);

        let b = Multivector::from_terms(vec![Term {
            left: None,
            blade: Blade::identity(),
            right: None,
            rotor: Some(Rotor {
                q1_mask: 0,
                q2_mask: 1,
                q3_mask: 0,
                sign: false,
            }),
            coeff: 1.0,
        }]);

        let fast = gp_mv(&a, &b);
        let slow = gp_mv_reference(&a, &b);

        assert!(same_mv(&fast, &slow));
    }

    #[test]
    fn fast_and_reference_agree_on_mixed_sparse_case() {
        let a = Multivector::from_terms(vec![
            Term {
                left: Some(Sector::new(0, 2)),
                blade: Blade::x(0),
                right: Some(Sector::new(1, 2)),
                rotor: Some(Rotor {
                    q1_mask: 1,
                    q2_mask: 0,
                    q3_mask: 0,
                    sign: false,
                }),
                coeff: 2.0,
            },
            Term {
                left: None,
                blade: Blade::identity(),
                right: None,
                rotor: None,
                coeff: 1.0,
            },
        ]);

        let b = Multivector::from_terms(vec![
            Term {
                left: Some(Sector::new(1, 2)),
                blade: Blade::z(1),
                right: Some(Sector::new(2, 2)),
                rotor: Some(Rotor {
                    q1_mask: 0,
                    q2_mask: 1,
                    q3_mask: 0,
                    sign: false,
                }),
                coeff: 3.0,
            },
            Term {
                left: None,
                blade: Blade::x(1),
                right: None,
                rotor: None,
                coeff: -1.0,
            },
        ]);

        let fast = gp_mv(&a, &b);
        let slow = gp_mv_reference(&a, &b);

        assert!(same_mv(&fast, &slow));
    }
}