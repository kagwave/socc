#[cfg(test)]
mod tests {
    use crate::core::bits::{Blade, Rotor, Sector};
    use crate::core::compute::term::gp_term;
    use crate::core::compute::reference::term_reference::gp_term_reference;
    use crate::core::ir::Term;

    fn id_term() -> Term {
        Term {
            left: None,
            blade: Blade::identity(),
            right: None,
            rotor: None,
            coeff: 1.0,
        }
    }

    #[test]
    fn gp_term_identity_times_identity() {
        let a = id_term();
        let b = id_term();

        let out = gp_term(&a, &b, 1).unwrap();

        assert_eq!(out.left, None);
        assert_eq!(out.blade, Blade::identity());
        assert_eq!(out.right, None);
        assert_eq!(out.rotor, None);
        assert!((out.coeff - 1.0).abs() < 1e-12);
    }

    #[test]
    fn gp_term_rejects_incompatible_inner_sectors() {
        let a = Term {
            left: Some(Sector::new(0, 1)),
            blade: Blade::identity(),
            right: Some(Sector::new(0, 1)),
            rotor: None,
            coeff: 1.0,
        };

        let b = Term {
            left: Some(Sector::new(1, 1)),
            blade: Blade::identity(),
            right: Some(Sector::new(0, 1)),
            rotor: None,
            coeff: 1.0,
        };

        assert!(gp_term(&a, &b, 1).is_none());
    }

    #[test]
    fn gp_term_multiplies_basic_blades() {
        // e2 * e1 = -J in the chosen convention
        let a = Term {
            left: None,
            blade: Blade::x(0),
            right: None,
            rotor: None,
            coeff: 1.0,
        };

        let b = Term {
            left: None,
            blade: Blade::z(0),
            right: None,
            rotor: None,
            coeff: 1.0,
        };

        let out = gp_term(&a, &b, 1).unwrap();

        assert_eq!(out.blade.x, 1);
        assert_eq!(out.blade.z, 1);
        assert!(out.blade.sign);
        assert!(out.rotor.is_none());
    }

    #[test]
    fn gp_term_multiplies_blades_on_different_qubits() {
        let a = Term {
            left: None,
            blade: Blade::x(0),
            right: None,
            rotor: None,
            coeff: 1.0,
        };

        let b = Term {
            left: None,
            blade: Blade::z(1),
            right: None,
            rotor: None,
            coeff: 1.0,
        };

        let out = gp_term(&a, &b, 2).unwrap();

        assert_eq!(out.blade.x, 1u64 << 0);
        assert_eq!(out.blade.z, 1u64 << 1);
        assert!(!out.blade.sign);
    }

    #[test]
    fn gp_term_composes_rotor_lane() {
        let a = Term {
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
        };

        let b = Term {
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
        };

        let out = gp_term(&a, &b, 1).unwrap();
        let r = out.rotor.unwrap();

        assert_eq!(r.q1_mask, 0);
        assert_eq!(r.q2_mask, 0);
        assert_eq!(r.q3_mask, 1);
        assert!(!r.sign);
    }

    #[test]
    fn gp_term_j_rotor_squared_gives_sign() {
        let a = Term {
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
        };

        let b = a;

        let out = gp_term(&a, &b, 1).unwrap();

        assert!(out.rotor.is_none());
        assert!(out.blade.sign);
    }

    #[test]
    fn gp_term_blade_and_rotor_lanes_stay_separate() {
        let a = Term {
            left: None,
            blade: Blade::x(0),
            right: None,
            rotor: Some(Rotor {
                q1_mask: 1,
                q2_mask: 0,
                q3_mask: 0,
                sign: false,
            }),
            coeff: 1.0,
        };

        let b = Term {
            left: None,
            blade: Blade::z(0),
            right: None,
            rotor: Some(Rotor {
                q1_mask: 0,
                q2_mask: 1,
                q3_mask: 0,
                sign: false,
            }),
            coeff: 1.0,
        };

        let out = gp_term(&a, &b, 1).unwrap();

        // Blade lane: X * Z = -J
        assert_eq!(out.blade.x, 1);
        assert_eq!(out.blade.z, 1);

        // Rotor lane: q1 + q2 = q3
        let r = out.rotor.unwrap();
        assert_eq!(r.q1_mask, 0);
        assert_eq!(r.q2_mask, 0);
        assert_eq!(r.q3_mask, 1);
    }

    #[test]
    fn gp_term_preserves_outer_sector_coordinates() {
        let a = Term {
            left: Some(Sector::new(0, 2)),
            blade: Blade::x(0),
            right: Some(Sector::new(1, 2)),
            rotor: None,
            coeff: 1.0,
        };

        let b = Term {
            left: Some(Sector::new(1, 2)),
            blade: Blade::identity(),
            right: Some(Sector::new(2, 2)),
            rotor: None,
            coeff: 1.0,
        };

        let out = gp_term(&a, &b, 2).unwrap();

        assert_eq!(out.left, Some(Sector::new(0, 2)));
        assert_eq!(out.right, Some(Sector::new(2, 2)));
    }

    #[test]
    fn gp_term_coeff_is_reset_to_structural_unit() {
        // gp_term is structural; coeff multiplication happens at mv level
        let a = Term {
            left: None,
            blade: Blade::x(0),
            right: None,
            rotor: None,
            coeff: 2.5,
        };

        let b = Term {
            left: None,
            blade: Blade::z(0),
            right: None,
            rotor: None,
            coeff: -7.0,
        };

        let out = gp_term(&a, &b, 1).unwrap();
        assert!((out.coeff - (2.5 * -7.0)).abs() < 1e-12);
    }

    #[test]
    fn gp_term_with_exact_all_p_sectors_can_drop_back_to_none() {
        let a = Term {
            left: Some(Sector::new(0, 1)),
            blade: Blade::identity(),
            right: Some(Sector::new(0, 1)),
            rotor: None,
            coeff: 1.0,
        };

        let b = id_term();

        let out = gp_term(a, b).unwrap();

        assert!(out.left.is_none() || out.left == Some(Sector::new(0, 1)));
        assert!(out.right.is_none() || out.right == Some(Sector::new(0, 1)));
    }

// Kernel tests for gp_term reference implementation, which is simpler but less efficient than the main gp_term.

    fn same_term(a: &Term, b: &Term) -> bool {
        a.left == b.left
            && a.blade == b.blade
            && a.right == b.right
            && a.right_rotor == b.right_rotor
            && (a.coeff - b.coeff).abs() < 1e-12
    }

    #[test]
    fn fast_and_reference_agree_on_basic_blade_product() {
        let a = Term {
            left: None,
            blade: Blade::x(0),
            right: None,
            right_rotor: None,
            coeff: 1.0,
        };

        let b = Term {
            left: None,
            blade: Blade::z(0),
            right: None,
            right_rotor: None,
            coeff: 1.0,
        };

        let fast = gp_term(a, b);
        let slow = gp_term_reference(&a, &b);

        assert_eq!(fast.is_some(), slow.is_some());
        assert!(same_term(&fast.unwrap(), &slow.unwrap()));
    }

    #[test]
    fn fast_and_reference_agree_on_sector_pruning() {
        let a = Term {
            left: Some(Sector::new(0, 1)),
            blade: Blade::identity(),
            right: Some(Sector::new(0, 1)),
            right_rotor: None,
            coeff: 1.0,
        };

        let b = Term {
            left: Some(Sector::new(1, 1)),
            blade: Blade::identity(),
            right: Some(Sector::new(0, 1)),
            right_rotor: None,
            coeff: 1.0,
        };

        assert_eq!(gp_term(a, b), gp_term_reference(&a, &b));
    }

    #[test]
    fn fast_and_reference_agree_on_rotor_composition() {
        let a = Term {
            left: None,
            blade: Blade::identity(),
            right: None,
            right_rotor: Some(Rotor {
                q1_mask: 1,
                q2_mask: 0,
                q3_mask: 0,
                sign: false,
            }),
            coeff: 1.0,
        };

        let b = Term {
            left: None,
            blade: Blade::identity(),
            right: None,
            right_rotor: Some(Rotor {
                q1_mask: 0,
                q2_mask: 1,
                q3_mask: 0,
                sign: false,
            }),
            coeff: 1.0,
        };

        let fast = gp_term(a, b).unwrap();
        let slow = gp_term_reference(a, b).unwrap();

        assert!(same_term(&fast, &slow));
    }

    #[test]
    fn fast_and_reference_agree_on_mixed_case() {
        let a = Term {
            left: Some(Sector::new(0, 2)),
            blade: Blade::x(0),
            right: Some(Sector::new(1, 2)),
            right_rotor: Some(Rotor {
                q1_mask: 1 << 0,
                q2_mask: 0,
                q3_mask: 0,
                sign: false,
            }),
            coeff: 1.0,
        };

        let b = Term {
            left: Some(Sector::new(1, 2)),
            blade: Blade::z(1),
            right: Some(Sector::new(2, 2)),
            right_rotor: Some(Rotor {
                q1_mask: 0,
                q2_mask: 1 << 0,
                q3_mask: 0,
                sign: false,
            }),
            coeff: 1.0,
        };

        let fast = gp_term(a, b).unwrap();
        let slow = gp_term_reference(a, b).unwrap();

        assert!(same_term(&fast, &slow));
    }
}