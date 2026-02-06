#[cfg(test)]
mod integration_tests {
    use crate::core::bits::{Blade, Sector};
    use crate::core::compute::local::{
        // Type layer
        LocalBlade,
        // Rewrite layer (reference)
        push_blade_through_right_sector,
        // Bitwise layer (optimized)
        parity_u64, pure_e1_mask, rewrite_bitwise,
        push_blade_through_right_sector_bitwise,
        // Tables layer
        local_h_action, local_s_action,
        // Packed block term
        PackedBlockTerm,
    };

    #[test]
    fn reference_and_bitwise_versions_agree_on_simple_rewrite() {
        let blade = Blade::new(1, 0, false); // e₂
        let sector = Sector::new(0, 1); // P

        let (ref_sector, ref_blade, ref_sign) =
            push_blade_through_right_sector(blade, sector);

        let (bw_sector, bw_blade, bw_sign) =
            push_blade_through_right_sector_bitwise(blade, sector);

        assert_eq!(ref_sector.bits, bw_sector.bits);
        assert_eq!(ref_sector.n, bw_sector.n);
        assert_eq!(ref_blade.x, bw_blade.x);
        assert_eq!(ref_blade.z, bw_blade.z);
        assert_eq!(ref_sign, bw_sign);
    }

    #[test]
    fn bitwise_rewrite_matches_local_table() {
        // e₁ on P: (0; 0,1) → (0; 0,0; false)
        let (new_s, new_x, new_z, new_sig) = rewrite_bitwise(0, 0, 1, false);
        assert_eq!((new_s, new_x, new_z, new_sig), (0, 0, 0, false));

        // e₁ on Q: (1; 0,1) → (1; 0,0; true)
        let (new_s, new_x, new_z, new_sig) = rewrite_bitwise(1, 0, 1, false);
        assert_eq!((new_s, new_x, new_z, new_sig), (1, 0, 0, true));

        // e₂ on P: (0; 1,0) → (1; 1,0; false)
        let (new_s, new_x, new_z, new_sig) = rewrite_bitwise(0, 1, 0, false);
        assert_eq!((new_s, new_x, new_z, new_sig), (1, 1, 0, false));

        // e₂ on Q: (1; 1,0) → (0; 1,0; false)
        let (new_s, new_x, new_z, new_sig) = rewrite_bitwise(1, 1, 0, false);
        assert_eq!((new_s, new_x, new_z, new_sig), (0, 1, 0, false));

        // J on P: (0; 1,1) → (1; 1,1; false)
        let (new_s, new_x, new_z, new_sig) = rewrite_bitwise(0, 1, 1, false);
        assert_eq!((new_s, new_x, new_z, new_sig), (1, 1, 1, false));
    }

    #[test]
    fn packed_block_term_locally_reduces() {
        // Create: Π_P · e₂ · Π_P with no rotor payload
        let term = PackedBlockTerm::new(
            0b0, // left = P
            1,   // blade_x = e₂
            0,   // blade_z
            0b0, // right = P
            0,   // rotor_q1
            0,   // rotor_q2
            0,   // rotor_q3
            false,
            1,
        );

        let after_right = term.push_blade_through_right();
        assert_eq!(after_right.right_bits, 0b1); // Q

        let final_reduced = after_right.push_blade_through_left();
        assert_eq!(final_reduced.left_bits, 0b1);  // Q
        assert_eq!(final_reduced.right_bits, 0b1); // Q
        assert_eq!(final_reduced.blade_x, 1);
        assert_eq!(final_reduced.blade_z, 0);
    }

    #[test]
    fn packed_block_term_gp_multiplies_rotor_lane_too() {
        let a = PackedBlockTerm::new(
            0, 0, 0, 0,
            1, 0, 0, // q1
            false,
            1,
        );

        let b = PackedBlockTerm::new(
            0, 0, 0, 0,
            0, 1, 0, // q2
            false,
            1,
        );

        let out = a.gp(b).unwrap();

        assert_eq!(out.rotor_q1, 0);
        assert_eq!(out.rotor_q2, 0);
        assert_eq!(out.rotor_q3, 1); // q1 + q2 = q3
        assert!(!out.sign);
    }

    #[test]
    fn parity_helper_computes_correctly() {
        assert!(!parity_u64(0b0000));
        assert!(parity_u64(0b0001));
        assert!(!parity_u64(0b0011));
        assert!(parity_u64(0b0111));
    }

    #[test]
    fn pure_e1_mask_identifies_pure_e1_positions() {
        let mask = pure_e1_mask(0b0000, 0b0001);
        assert_eq!(mask, 0b0001);

        let mask = pure_e1_mask(0b0000, 0b0101);
        assert_eq!(mask, 0b0101);

        let mask = pure_e1_mask(0b0010, 0b0000);
        assert_eq!(mask, 0b0000);

        let mask = pure_e1_mask(0b0001, 0b0001);
        assert_eq!(mask, 0b0000);
    }

    #[test]
    fn table_and_bitwise_agree_on_h_action() {
        let (new_blade, new_sign) = local_h_action(LocalBlade::E1);
        assert_eq!((new_blade, new_sign), (LocalBlade::E2, false));

        let (new_blade, new_sign) = local_h_action(LocalBlade::E2);
        assert_eq!((new_blade, new_sign), (LocalBlade::E1, false));

        let (new_blade, new_sign) = local_h_action(LocalBlade::J);
        assert_eq!((new_blade, new_sign), (LocalBlade::J, true));
    }

    #[test]
    fn table_and_bitwise_agree_on_s_action() {
        let (new_blade, new_sign) = local_s_action(LocalBlade::E1);
        assert_eq!((new_blade, new_sign), (LocalBlade::E1, false));

        let (new_blade, new_sign) = local_s_action(LocalBlade::E2);
        assert_eq!((new_blade, new_sign), (LocalBlade::J, false));

        let (new_blade, new_sign) = local_s_action(LocalBlade::J);
        assert_eq!((new_blade, new_sign), (LocalBlade::E2, true));
    }
}