#[cfg(test)]
mod tests {
    use crate::core::bits::{Blade, Sector};
    use crate::core::compute::local::{
        LocalBlade,
        LocalSector,
        local_blade_at,
        local_sector_at,
        rewrite_left,
        rewrite_right,
        push_blade_through_left_sector,
        push_blade_through_right_sector,
    };

    // ------------------------------------------------------------------------
    // Local rewrite rules: blade acting on sector from the right
    //
    // These encode the algebra: P = (1 + e1)/2, Q = (1 - e1)/2
    //
    // From this we get identities like:
    //
    //   e1 P = +P
    //   e1 Q = -Q
    //   e2 P = Q e2
    //   e2 Q = P e2

    #[test]
    fn rewrite_right_e1_on_p() {
        // Algebra:
        //
        //   e1 P = +P
        //
        // Because P is the +1 eigenprojector of e1.
        let (s, b, sign) = rewrite_right(LocalBlade::E1, LocalSector::P);

        assert_eq!(s, LocalSector::P);
        assert_eq!(b, LocalBlade::I); // e1 collapses to scalar
        assert!(!sign); // positive eigenvalue
    }

    #[test]
    fn rewrite_right_e1_on_q() {
        // Algebra:
        //
        //   e1 Q = -Q
        //
        // Because Q is the -1 eigenprojector of e1.
        let (s, b, sign) = rewrite_right(LocalBlade::E1, LocalSector::Q);

        assert_eq!(s, LocalSector::Q);
        assert_eq!(b, LocalBlade::I);
        assert!(sign); // negative eigenvalue
    }

    #[test]
    fn rewrite_right_e2_flips_sector() {
        // Algebra:
        //
        //   e2 P = Q e2
        //   e2 Q = P e2
        //
        // e2 swaps the two idempotent sectors.
        let (s, b, sign) = rewrite_right(LocalBlade::E2, LocalSector::P);

        assert_eq!(s, LocalSector::Q);
        assert_eq!(b, LocalBlade::E2);
        assert!(!sign);

        let (s2, b2, sign2) = rewrite_right(LocalBlade::E2, LocalSector::Q);

        assert_eq!(s2, LocalSector::P);
        assert_eq!(b2, LocalBlade::E2);
        assert!(!sign2);
    }

    #[test]
    fn rewrite_right_j_flips_sector() {
        // J = e1 e2
        //
        // In this convention J also flips the sector.
        //
        //   J P = Q J
        //   J Q = P J
        let (s, b, sign) = rewrite_right(LocalBlade::J, LocalSector::P);

        assert_eq!(s, LocalSector::Q);
        assert_eq!(b, LocalBlade::J);
        assert!(!sign);

        let (s2, b2, sign2) = rewrite_right(LocalBlade::J, LocalSector::Q);

        assert_eq!(s2, LocalSector::P);
        assert_eq!(b2, LocalBlade::J);
        assert!(!sign2);
    }

    // ------------------------------------------------------------------------
    // Local rewrite rules: sector acting on blade from the left
    //
    // These correspond to:
    //
    //   P e1 = P
    //   Q e1 = -Q
    //
    //   P e2 = e2 Q
    //   Q e2 = e2 P
    //
    // which are the mirror versions of the previous rules.
    // ------------------------------------------------------------------------

    #[test]
    fn rewrite_left_e1_on_q() {
        // Algebra:
        //
        //   Q e1 = -Q
        let (b, s, sign) = rewrite_left(LocalSector::Q, LocalBlade::E1);

        assert_eq!(b, LocalBlade::I);
        assert_eq!(s, LocalSector::Q);
        assert!(sign);
    }

    #[test]
    fn rewrite_left_e2_flips_sector() {
        // Algebra:
        //
        //   P e2 = e2 Q
        //   Q e2 = e2 P
        let (b, s, sign) = rewrite_left(LocalSector::P, LocalBlade::E2);

        assert_eq!(b, LocalBlade::E2);
        assert_eq!(s, LocalSector::Q);
        assert!(!sign);

        let (b2, s2, sign2) = rewrite_left(LocalSector::Q, LocalBlade::E2);

        assert_eq!(b2, LocalBlade::E2);
        assert_eq!(s2, LocalSector::P);
        assert!(!sign2);
    }

    // ------------------------------------------------------------------------
    // Packed rewrite tests
    //
    // These verify that the bitwise packed routines correctly apply
    // the local rewrite rules across qubits.
    //
    // These functions are the real performance-critical part of the engine.
    // ------------------------------------------------------------------------

    #[test]
    fn push_blade_through_right_sector_flips_sector_bit() {
        // Setup:
        //
        // blade = e2
        // sector = P
        //
        // Expected algebra:
        //
        //   e2 P = Q e2
        let blade = Blade::x(0); // e2
        let sector = Sector::from_bits(0, 1); // P

        let (new_sector, new_blade, sign) =
            push_blade_through_right_sector(blade, sector);

        // sector flipped P -> Q
        assert_eq!(new_sector.bits, 1);

        // blade unchanged
        assert_eq!(new_blade.x, blade.x);
        assert_eq!(new_blade.z, blade.z);

        // no sign change
        assert!(!sign);
    }

    #[test]
    fn push_blade_through_left_sector_flips_sector_bit() {
        // Setup:
        //
        // sector = Q
        // blade = e2
        //
        // Expected algebra:
        //
        //   Q e2 = e2 P
        let blade = Blade::x(0);
        let sector = Sector::from_bits(1, 1); // Q

        let (new_blade, new_sector, sign) =
            push_blade_through_left_sector(sector, blade);

        // sector flipped Q -> P
        assert_eq!(new_sector.bits, 0);

        // blade unchanged
        assert_eq!(new_blade.x, blade.x);
        assert_eq!(new_blade.z, blade.z);

        assert!(!sign);
    }

    // ------------------------------------------------------------------------
    // Packed detection helpers
    //
    // These confirm that the bit packing logic correctly extracts
    // local algebraic states from the packed blade/sector structures.
    // ------------------------------------------------------------------------

    #[test]
    fn packed_local_blade_detection() {
        // J = e1 e2
        let blade = Blade::j(3);

        // Confirm packed representation correctly identifies local blade.
        assert_eq!(local_blade_at(blade, 3), LocalBlade::J);
    }

    #[test]
    fn packed_local_sector_detection() {
        // sector with bit 2 set -> Q at qubit 2
        let sector = Sector::from_bits(1 << 2, 3);

        assert_eq!(local_sector_at(sector, 2), LocalSector::Q);
    }
}