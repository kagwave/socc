#[cfg(test)]
mod tests {
    use crate::core::compute::rotor::{
        compose_rotor,
        rotor_class_at,
        rotor_identity,
        rotor_is_identity,
        rotor_j_at,
        rotor_quarter_at,
        rotor_three_quarter_at,
    };

    #[test]
    fn identity_composes_trivially() {
        let id = rotor_identity();
        let r = rotor_j_at(2);

        assert_eq!(compose_rotor(id, r), r);
        assert_eq!(compose_rotor(r, id), r);
    }

    #[test]
    fn same_site_j_squared_gives_minus_identity() {
        let r = rotor_j_at(1);
        let out = compose_rotor(r, r);

        assert_eq!(out.q1_mask, 0);
        assert_eq!(out.q2_mask, 0);
        assert_eq!(out.q3_mask, 0);
        assert!(out.sign);
    }

    #[test]
    fn different_sites_j_accumulate_without_extra_sign() {
        let a = rotor_j_at(0);
        let b = rotor_j_at(3);

        let out = compose_rotor(a, b);

        assert_eq!(out.q1_mask, 0);
        assert_eq!(out.q2_mask, (1u64 << 0) | (1u64 << 3));
        assert_eq!(out.q3_mask, 0);
        assert!(!out.sign);
    }

    #[test]
    fn quarter_turn_squared_is_j() {
        let t = rotor_quarter_at(0);
        let out = compose_rotor(t, t);

        assert_eq!(out.q1_mask, 0);
        assert_eq!(out.q2_mask, 1u64 << 0);
        assert_eq!(out.q3_mask, 0);
        assert!(!out.sign);
    }

    #[test]
    fn quarter_turn_times_j_is_three_quarter_turn() {
        let t = rotor_quarter_at(0);
        let j = rotor_j_at(0);

        let out = compose_rotor(t, j);

        assert_eq!(out.q1_mask, 0);
        assert_eq!(out.q2_mask, 0);
        assert_eq!(out.q3_mask, 1u64 << 0);
        assert!(!out.sign);
    }

    #[test]
    fn three_quarter_turn_times_quarter_turn_is_minus_identity() {
        let a = rotor_three_quarter_at(0);
        let b = rotor_quarter_at(0);

        let out = compose_rotor(a, b);

        assert_eq!(out.q1_mask, 0);
        assert_eq!(out.q2_mask, 0);
        assert_eq!(out.q3_mask, 0);
        assert!(out.sign);
    }

    #[test]
    fn inverse_of_quarter_turn_is_sign_times_three_quarter_turn() {
        let t = rotor_quarter_at(2);
        let inv = rotor_three_quarter_at(2);
        let inv = crate::core::bits::Rotor {
            sign: true,
            ..inv
        };

        assert_eq!(inv.q1_mask, 0);
        assert_eq!(inv.q2_mask, 0);
        assert_eq!(inv.q3_mask, 1u64 << 2);
        assert!(inv.sign);

        let prod = compose_rotor(t, inv);
        assert!(rotor_is_identity(prod));
    }

    #[test]
    fn rotor_class_at_builds_expected_local_classes() {
        let q = 5usize;

        let c1 = rotor_quarter_at(q);
        assert_eq!(rotor_class_at(c1, q), 1);

        let c2 = rotor_j_at(q);
        assert_eq!(rotor_class_at(c2, q), 2);

        let c3 = rotor_three_quarter_at(q);
        assert_eq!(rotor_class_at(c3, q), 3);

        let id = rotor_identity();
        assert_eq!(rotor_class_at(id, q), 0);
    }
}