use crate::core::bits::{Blade, Rotor, Sector};

use super::types::{LocalBlade, LocalRotor, LocalSector};

#[inline(always)]
pub fn local_blade_at(blade: Blade, i: usize) -> LocalBlade {
    let xb = ((blade.x >> i) & 1) != 0;
    let zb = ((blade.z >> i) & 1) != 0;

    match (xb, zb) {
        (false, false) => LocalBlade::I,
        (false, true) => LocalBlade::E1,
        (true, false) => LocalBlade::E2,
        (true, true) => LocalBlade::J,
    }
}

#[inline(always)]
pub fn set_local_blade(mut blade: Blade, i: usize, local: LocalBlade) -> Blade {
    let bit = 1u64 << i;

    blade.x &= !bit;
    blade.z &= !bit;

    match local {
        LocalBlade::I => {}
        LocalBlade::E1 => blade.z |= bit,
        LocalBlade::E2 => blade.x |= bit,
        LocalBlade::J => {
            blade.x |= bit;
            blade.z |= bit;
        }
    }

    blade
}

#[inline(always)]
pub fn local_sector_at(sector: Sector, i: usize) -> LocalSector {
    if sector.bit(i as u32) {
        LocalSector::Q
    } else {
        LocalSector::P
    }
}

#[inline(always)]
pub fn set_local_sector(mut sector: Sector, i: usize, local: LocalSector) -> Sector {
    sector = match local {
        LocalSector::P => sector.set_bit(i as u32, false),
        LocalSector::Q => sector.set_bit(i as u32, true),
    };
    sector
}

//////////////////////////////////////////////////////////////
// LOCAL ROTOR EXTRACTION / SETTING
//////////////////////////////////////////////////////////////

#[inline(always)]
pub fn local_rotor_at(rotor: Rotor, i: usize) -> LocalRotor {
    let bit = 1u64 << i;

    if (rotor.q1_mask & bit) != 0 {
        LocalRotor::Q1
    } else if (rotor.q2_mask & bit) != 0 {
        LocalRotor::Q2
    } else if (rotor.q3_mask & bit) != 0 {
        LocalRotor::Q3
    } else {
        LocalRotor::I
    }
}

#[inline(always)]
pub fn set_local_rotor(mut rotor: Rotor, i: usize, local: LocalRotor) -> Rotor {
    let bit = 1u64 << i;

    rotor.q1_mask &= !bit;
    rotor.q2_mask &= !bit;
    rotor.q3_mask &= !bit;

    match local {
        LocalRotor::I => {}
        LocalRotor::Q1 => rotor.q1_mask |= bit,
        LocalRotor::Q2 => rotor.q2_mask |= bit,
        LocalRotor::Q3 => rotor.q3_mask |= bit,
    }

    rotor
}

//////////////////////////////////////////////////////////////
// SECTOR REWRITE RULES FOR BLADES
//////////////////////////////////////////////////////////////

/// Rewrite a local right action:
///
///     B S  ->  S' B'
///
/// where `S` is `P` or `Q`.
///
/// Returns:
/// - new sector
/// - new blade
/// - sign bit (true = multiply by -1)
#[inline(always)]
pub fn rewrite_right(blade: LocalBlade, sector: LocalSector) -> (LocalSector, LocalBlade, bool) {
    match (blade, sector) {
        (LocalBlade::I, LocalSector::P) => (LocalSector::P, LocalBlade::I, false),
        (LocalBlade::I, LocalSector::Q) => (LocalSector::Q, LocalBlade::I, false),

        (LocalBlade::E1, LocalSector::P) => (LocalSector::P, LocalBlade::I, false),
        (LocalBlade::E1, LocalSector::Q) => (LocalSector::Q, LocalBlade::I, true),

        (LocalBlade::E2, LocalSector::P) => (LocalSector::Q, LocalBlade::E2, false),
        (LocalBlade::E2, LocalSector::Q) => (LocalSector::P, LocalBlade::E2, false),

        (LocalBlade::J, LocalSector::P) => (LocalSector::Q, LocalBlade::J, false),
        (LocalBlade::J, LocalSector::Q) => (LocalSector::P, LocalBlade::J, false),
    }
}

/// Rewrite a local left action:
///
///     S B  ->  B' S'
///
/// where `S` is `P` or `Q`.
///
/// Returns:
/// - new blade
/// - new sector
/// - sign bit (true = multiply by -1)
#[inline(always)]
pub fn rewrite_left(sector: LocalSector, blade: LocalBlade) -> (LocalBlade, LocalSector, bool) {
    match (sector, blade) {
        (LocalSector::P, LocalBlade::I) => (LocalBlade::I, LocalSector::P, false),
        (LocalSector::Q, LocalBlade::I) => (LocalBlade::I, LocalSector::Q, false),

        (LocalSector::P, LocalBlade::E1) => (LocalBlade::I, LocalSector::P, false),
        (LocalSector::Q, LocalBlade::E1) => (LocalBlade::I, LocalSector::Q, true),

        (LocalSector::P, LocalBlade::E2) => (LocalBlade::E2, LocalSector::Q, false),
        (LocalSector::Q, LocalBlade::E2) => (LocalBlade::E2, LocalSector::P, false),

        (LocalSector::P, LocalBlade::J) => (LocalBlade::J, LocalSector::Q, false),
        (LocalSector::Q, LocalBlade::J) => (LocalBlade::J, LocalSector::P, false),
    }
}

//////////////////////////////////////////////////////////////
// ROTOR "REWRITE" / TRANSPORT RULES
//////////////////////////////////////////////////////////////

/// Right rotors are a separate phase lane. In the paper's biaction view,
/// they live on the right and are controlled by sectors rather than
/// collapsing into the left blade algebra. So pushing a pure right rotor
/// through a sector does NOT change either object locally.
///
/// Returns:
/// - new sector
/// - new rotor
/// - sign bit
#[inline(always)]
pub fn rewrite_right_rotor(
    rotor: LocalRotor,
    sector: LocalSector,
) -> (LocalSector, LocalRotor, bool) {
    (sector, rotor, false)
}

/// Left-sector / right-rotor transport. Same semantics as above.
#[inline(always)]
pub fn rewrite_left_rotor(
    sector: LocalSector,
    rotor: LocalRotor,
) -> (LocalRotor, LocalSector, bool) {
    (rotor, sector, false)
}

//////////////////////////////////////////////////////////////
// PACKED REFERENCE PUSH OPERATIONS (BLADE / SECTOR)
//////////////////////////////////////////////////////////////

/// Push a packed blade through an exact right sector:
///
///     B Π_x = (±) Π_y B'
///
/// Returns:
/// - new sector Π_y
/// - rewritten blade B'
/// - sign bit (true = multiply by -1)
pub fn push_blade_through_right_sector(blade: Blade, sector: Sector) -> (Sector, Blade, bool) {
    let mut out_sector = sector;
    let mut out_blade = blade.unsigned();
    let mut sign = blade.sign;

    for i in 0..(sector.n as usize) {
        let lb = local_blade_at(out_blade, i);
        let ls = local_sector_at(out_sector, i);

        let (new_sector, new_blade, sgn) = rewrite_right(lb, ls);

        out_sector = set_local_sector(out_sector, i, new_sector);
        out_blade = set_local_blade(out_blade, i, new_blade);
        sign ^= sgn;
    }

    (out_sector, out_blade, sign)
}

/// Push an exact left sector through a packed blade:
///
///     Π_x B = (±) B' Π_y
///
/// Returns:
/// - rewritten blade B'
/// - new sector Π_y
/// - sign bit (true = multiply by -1)
pub fn push_blade_through_left_sector(sector: Sector, blade: Blade) -> (Blade, Sector, bool) {
    let mut out_sector = sector;
    let mut out_blade = blade.unsigned();
    let mut sign = blade.sign;

    for i in 0..(sector.n as usize) {
        let ls = local_sector_at(out_sector, i);
        let lb = local_blade_at(out_blade, i);

        let (new_blade, new_sector, sgn) = rewrite_left(ls, lb);

        out_blade = set_local_blade(out_blade, i, new_blade);
        out_sector = set_local_sector(out_sector, i, new_sector);
        sign ^= sgn;
    }

    (out_blade, out_sector, sign)
}

//////////////////////////////////////////////////////////////
// PACKED REFERENCE "PUSH" FOR ROTORS
//////////////////////////////////////////////////////////////

/// Push a packed right rotor through an exact right sector.
///
/// In the current SOCC design, the rotor lane is independent from the
/// blade transport lane, so this is identity on packed data.
pub fn push_rotor_through_right_sector(rotor: Rotor, sector: Sector) -> (Sector, Rotor, bool) {
    let mut out_sector = sector;
    let mut out_rotor = rotor;
    let mut sign = rotor.sign;

    for i in 0..(sector.n as usize) {
        let lr = local_rotor_at(out_rotor, i);
        let ls = local_sector_at(out_sector, i);

        let (new_sector, new_rotor, sgn) = rewrite_right_rotor(lr, ls);

        out_sector = set_local_sector(out_sector, i, new_sector);
        out_rotor = set_local_rotor(out_rotor, i, new_rotor);
        sign ^= sgn;
    }

    out_rotor.sign = false;
    (out_sector, out_rotor, sign)
}

/// Push an exact left sector through a packed right rotor.
///
/// Same current semantics as the right version.
pub fn push_rotor_through_left_sector(sector: Sector, rotor: Rotor) -> (Rotor, Sector, bool) {
    let mut out_sector = sector;
    let mut out_rotor = rotor;
    let mut sign = rotor.sign;

    for i in 0..(sector.n as usize) {
        let ls = local_sector_at(out_sector, i);
        let lr = local_rotor_at(out_rotor, i);

        let (new_rotor, new_sector, sgn) = rewrite_left_rotor(ls, lr);

        out_rotor = set_local_rotor(out_rotor, i, new_rotor);
        out_sector = set_local_sector(out_sector, i, new_sector);
        sign ^= sgn;
    }

    out_rotor.sign = false;
    (out_rotor, out_sector, sign)
}