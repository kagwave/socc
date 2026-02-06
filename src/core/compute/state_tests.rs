#[cfg(test)]
mod tests {
    use crate::core::bits::{Blade, Rotor, Sector};
    use crate::core::compute::state::{
        apply_mv_to_right_sector,
        apply_to_vacuum,
        peirce_block,
    };
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
        right_rotor: Option<Rotor>,
        coeff: f64,
    ) -> bool {
        mv.terms.iter().any(|t| {
            t.left == left
                && t.blade == blade
                && t.right == right
                && t.right_rotor == right_rotor
                && approx_eq(t.coeff, coeff)
        })
    }

    #[test]
    fn apply_x_to_p_sector_gives_qe2() {
        let p = Sector::from_bits(0, 1);
        let q = Sector::from_bits(1, 1);

        let op = Multivector::from_blade(1, Blade::x(0), 1.0);
        let out = apply_mv_to_right_sector(&op, p);

        assert_eq!(out.terms.len(), 1);
        assert!(has_term(
            &out,
            None,
            Blade::x(0),
            Some(q),
            None,
            1.0
        ));
    }

    #[test]
    fn apply_z_to_q_sector_gives_minus_q() {
        let q = Sector::from_bits(1, 1);

        let op = Multivector::from_blade(1, Blade::z(0), 2.0);
        let out = apply_mv_to_right_sector(&op, q);

        assert_eq!(out.terms.len(), 1);
        assert!(has_term(
            &out,
            None,
            Blade::identity(),
            Some(q),
            None,
            -2.0
        ));
    }

    #[test]
    fn apply_j_to_p_sector_flips_sector_and_keeps_j() {
        let p = Sector::from_bits(0, 1);
        let q = Sector::from_bits(1, 1);

        let op = Multivector::from_blade(1, Blade::j(0), 1.0);
        let out = apply_mv_to_right_sector(&op, p);

        assert_eq!(out.terms.len(), 1);
        assert!(has_term(
            &out,
            None,
            Blade::j(0),
            Some(q),
            None,
            1.0
        ));
    }

    #[test]
    fn peirce_block_extracts_matching_block() {
        let s0 = Sector::from_bits(0, 1);
        let s1 = Sector::from_bits(1, 1);

        let op = Multivector::from_terms(vec![
            Term {
                left: Some(s1),
                blade: Blade::x(0),
                right: Some(s0),
                right_rotor: None,
                coeff: 3.0,
            },
            Term {
                left: Some(s0),
                blade: Blade::identity(),
                right: Some(s0),
                right_rotor: None,
                coeff: 1.0,
            },
        ]);

        let block = peirce_block(&op, s1, s0);

        assert_eq!(block.terms.len(), 1);
        assert!(has_term(
            &block,
            Some(s1),
            Blade::x(0),
            Some(s0),
            None,
            3.0
        ));
    }

    #[test]
    fn peirce_block_preserves_rotor_payload() {
        let s0 = Sector::from_bits(0, 1);

        let op = Multivector::from_terms(vec![
            Term {
                left: Some(s0),
                blade: Blade::identity(),
                right: Some(s0),
                right_rotor: Some(Rotor {
                    q1_mask: 0,
                    q2_mask: 1,
                    q3_mask: 0,
                    sign: false,
                }),
                coeff: 2.0,
            },
            Term {
                left: None,
                blade: Blade::x(0),
                right: None,
                right_rotor: None,
                coeff: 1.0,
            },
        ]);

        let block = peirce_block(&op, s0, s0);

        assert_eq!(block.terms.len(), 1);
        assert!(has_term(
            &block,
            Some(s0),
            Blade::identity(),
            Some(s0),
            Some(Rotor {
                q1_mask: 0,
                q2_mask: 1,
                q3_mask: 0,
                sign: false,
            }),
            2.0
        ));
    }

    #[test]
    fn apply_to_vacuum_is_alias_for_right_sector_application() {
        let vacuum = Sector::from_bits(0, 1);

        let op = Multivector::from_blade(1, Blade::x(0), 1.0);
        let out = apply_to_vacuum(&op, vacuum);

        assert_eq!(out.terms.len(), 1);
        assert!(has_term(
            &out,
            None,
            Blade::x(0),
            Some(Sector::from_bits(1, 1)),
            None,
            1.0
        ));
    }
}