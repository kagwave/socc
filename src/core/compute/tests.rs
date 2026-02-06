#[cfg(test)]
mod tests {
    use crate::core::bits::{Blade, Rotor, Sector};
    use crate::core::compute::blade::{anticommutes, gp_blade};
    use crate::core::compute::involutions::{grade_involution_blade, reverse_blade};
    use crate::core::compute::multivector::{add_mv, gp_mv, scale_mv, sub_mv};
    use crate::core::compute::term::gp_term;
    use crate::core::ir::{Multivector, Term};

    const EPS: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    fn has_term(
        mv: &Multivector,
        left: Option<Sector>,
        blade: Blade,
        right: Option<Sector>,
        rotor: Option<Rotor>,
        coeff: f64,
    ) -> bool {
        mv.terms.iter().any(|t| {
            t.left == left
                && t.blade == blade
                && t.right == right
                && t.rotor == rotor
                && approx_eq(t.coeff, coeff)
        })
    }

    #[test]
    fn blade_identity_mul_left_and_right() {
        let a = Blade::x(2);
        assert_eq!(gp_blade(a, Blade::identity()), a);
        assert_eq!(gp_blade(Blade::identity(), a), a);
    }

    #[test]
    fn blade_x_square_is_identity() {
        let x0 = Blade::x(0);
        let out = gp_blade(x0, x0);
        assert_eq!(out, Blade::identity());
    }

    #[test]
    fn blade_z_square_is_identity() {
        let z0 = Blade::z(0);
        let out = gp_blade(z0, z0);
        assert_eq!(out, Blade::identity());
    }

    #[test]
    fn x_and_z_anticommute_same_qubit() {
        let x0 = Blade::x(0);
        let z0 = Blade::z(0);
        assert!(anticommutes(x0, z0));
    }

    #[test]
    fn x_and_z_commute_different_qubits() {
        let x0 = Blade::x(0);
        let z1 = Blade::z(1);
        assert!(!anticommutes(x0, z1));
    }

    #[test]
    fn reverse_is_involution_on_blades() {
        let b = gp_blade(Blade::x(0), Blade::z(1));
        let rr = reverse_blade(reverse_blade(b));
        assert_eq!(rr, b);
    }

    #[test]
    fn grade_involution_is_involution_on_blades() {
        let b = gp_blade(Blade::x(0), Blade::z(0));
        let gg = grade_involution_blade(grade_involution_blade(b));
        assert_eq!(gg, b);
    }

    #[test]
    fn multivector_x_plus_z_squared_is_2i() {
        let x = Multivector::from_blade(1, Blade::x(0), 1.0);
        let z = Multivector::from_blade(1, Blade::z(0), 1.0);

        let x_plus_z = add_mv(&x, &z);
        let prod = gp_mv(&x_plus_z, &x_plus_z);

        assert_eq!(prod.terms.len(), 1);
        assert!(has_term(&prod, None, Blade::identity(), None, None, 2.0));
    }

    #[test]
    fn projector_p_squared_is_p() {
        let i = Multivector::from_blade(1, Blade::identity(), 1.0);
        let z = Multivector::from_blade(1, Blade::z(0), 1.0);
        let p = scale_mv(&add_mv(&i, &z), 0.5);

        let pp = gp_mv(&p, &p);

        assert_eq!(pp.terms.len(), 2);
        assert!(has_term(&pp, None, Blade::identity(), None, None, 0.5));
        assert!(has_term(&pp, None, Blade::z(0), None, None, 0.5));
    }

    #[test]
    fn projector_q_squared_is_q() {
        let i = Multivector::from_blade(1, Blade::identity(), 1.0);
        let z = Multivector::from_blade(1, Blade::z(0), 1.0);
        let q = scale_mv(&sub_mv(&i, &z), 0.5);

        let qq = gp_mv(&q, &q);

        assert_eq!(qq.terms.len(), 2);
        assert!(has_term(&qq, None, Blade::identity(), None, None, 0.5));
        assert!(has_term(&qq, None, Blade::z(0), None, None, -0.5));
    }

    #[test]
    fn projector_pq_is_zero() {
        let i = Multivector::from_blade(1, Blade::identity(), 1.0);
        let z = Multivector::from_blade(1, Blade::z(0), 1.0);

        let p = scale_mv(&add_mv(&i, &z), 0.5);
        let q = scale_mv(&sub_mv(&i, &z), 0.5);

        let pq = gp_mv(&p, &q);
        assert!(pq.terms.is_empty());
    }

    #[test]
    fn peirce_term_composes_when_middle_matches() {
        let s0 = Sector::from_bits(0, 1);
        let s1 = Sector::from_bits(1, 1);

        let a = Term::sector_map(s1, Blade::identity(), s0, 2.0);
        let b = Term::sector_map(s0, Blade::identity(), s1, 3.0);

        let out = gp_term(a, b).unwrap();

        assert_eq!(out.left, Some(s1));
        assert_eq!(out.right, Some(s1));
        assert!(approx_eq(out.coeff, 1.0)); // structural kernel
    }

    #[test]
    fn peirce_term_composes_with_blade_sign() {
        let s0 = Sector::from_bits(0, 1);
        let s1 = Sector::from_bits(1, 1);

        let a = Term::sector_map(s1, Blade::x(0), s0, 2.0);
        let b = Term::sector_map(s0, Blade::z(0), s1, 3.0);

        let out = gp_term(a, b).unwrap();

        assert_eq!(out.left, Some(s1));
        assert_eq!(out.right, Some(s1));
        assert_eq!(out.blade.x, 1);
        assert_eq!(out.blade.z, 1);
        assert!(out.blade.sign);
        assert!(approx_eq(out.coeff, 1.0)); // structural kernel
    }

    #[test]
    fn structural_nilpotent_example_off_diagonal_sector_map() {
        let s0 = Sector::from_bits(0, 1);
        let s1 = Sector::from_bits(1, 1);

        let a = Term::sector_map(s1, Blade::identity(), s0, 1.0);
        let aa = gp_term(a, a);

        assert!(aa.is_none());
    }

    #[test]
    fn rotor_payload_survives_multivector_add_and_simplify() {
        let rotor = Some(Rotor {
            q1_mask: 1u64 << 0,
            q2_mask: 0,
            q3_mask: 0,
            sign: false,
        });

        let t1 = Term {
            left: None,
            blade: Blade::z(0),
            right: None,
            coeff: 1.0,
            rotor: rotor,
        };

        let t2 = Term {
            left: None,
            blade: Blade::z(0),
            right: None,
            coeff: 2.0,
            rotor: rotor,
        };

        let mv1 = Multivector { terms: vec![t1] };
        let mv2 = Multivector { terms: vec![t2] };

        let sum = add_mv(&mv1, &mv2);

        assert_eq!(sum.terms.len(), 1);
        let t = &sum.terms[0];
        assert_eq!(t.rotor, rotor);
        assert_eq!(t.blade, Blade::z(0));
        assert!(approx_eq(t.coeff, 3.0));
    }

    #[test]
    fn rotor_lane_is_composed_in_gp_term() {
        let s0 = Sector::from_bits(0, 1);
        let s1 = Sector::from_bits(1, 1);

        let rotor_a = Some(Rotor {
            q1_mask: 1u64 << 0,
            q2_mask: 0,
            q3_mask: 0,
            sign: false,
        });

        let rotor_b = Some(Rotor {
            q1_mask: 0,
            q2_mask: 1u64 << 0,
            q3_mask: 0,
            sign: false,
        });

        let a = Term {
            left: Some(s1),
            blade: Blade::x(0),
            right: Some(s0),
            coeff: 2.0,
            rotor: rotor_a,
        };

        let b = Term {
            left: Some(s0),
            blade: Blade::z(0),
            right: Some(s1),
            coeff: 3.0,
            rotor: rotor_b,
        };

        let out = gp_term(a, b).unwrap();

        assert_eq!(out.left, Some(s1));
        assert_eq!(out.right, Some(s1));

        let r = out.rotor.unwrap();
        assert_eq!(r.q1_mask, 0);
        assert_eq!(r.q2_mask, 0);
        assert_eq!(r.q3_mask, 1u64 << 0);
    }

    #[test]
    fn gp_mv_multiplies_coefficients_even_though_gp_term_is_structural() {
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

        let out = gp_mv(&a, &b);
        assert_eq!(out.terms.len(), 1);
        assert!(approx_eq(out.terms[0].coeff, 6.0));
        assert_eq!(out.terms[0].blade.x, 1);
        assert_eq!(out.terms[0].blade.z, 1);
        assert!(out.terms[0].blade.sign);
    }
}