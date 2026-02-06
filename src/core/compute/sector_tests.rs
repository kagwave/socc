#[cfg(test)]
mod tests {
    use crate::core::bits::{Blade, Sector};
    use crate::core::compute::local::{
        push_blade_through_left_sector,
        push_blade_through_right_sector,
    };
    use crate::core::compute::sector::{
        push_blade_through_left_exact_sector,
        push_blade_through_right_exact_sector,
    };

    #[test]
    fn packed_right_rewrite_matches_local_on_single_qubit_e2p() {
        let blade = Blade::x(0);
        let sector = Sector::from_bits(0, 1); // P

        let a = push_blade_through_right_sector(blade, sector);
        let b = push_blade_through_right_exact_sector(blade, sector);

        assert_eq!(a, b);
    }

    #[test]
    fn packed_right_rewrite_matches_local_on_single_qubit_e1q() {
        let blade = Blade::z(0);
        let sector = Sector::from_bits(1, 1); // Q

        let a = push_blade_through_right_sector(blade, sector);
        let b = push_blade_through_right_exact_sector(blade, sector);

        assert_eq!(a, b);
    }

    #[test]
    fn packed_left_rewrite_matches_local_on_single_qubit_qe2() {
        let blade = Blade::x(0);
        let sector = Sector::from_bits(1, 1); // Q

        let a = push_blade_through_left_sector(sector, blade);
        let b = push_blade_through_left_exact_sector(sector, blade);

        assert_eq!(a, b);
    }

    #[test]
    fn packed_right_rewrite_matches_local_on_multi_qubit_case() {
        let blade = Blade::new((1 << 0) | (1 << 2), (1 << 1) | (1 << 2), false);
        let sector = Sector::from_bits(0b101, 3);

        let a = push_blade_through_right_sector(blade, sector);
        let b = push_blade_through_right_exact_sector(blade, sector);

        assert_eq!(a, b);
    }

    #[test]
    fn packed_left_rewrite_matches_local_on_multi_qubit_case() {
        let blade = Blade::new((1 << 0) | (1 << 2), (1 << 1) | (1 << 2), false);
        let sector = Sector::from_bits(0b011, 3);

        let a = push_blade_through_left_sector(sector, blade);
        let b = push_blade_through_left_exact_sector(sector, blade);

        assert_eq!(a, b);
    }
}