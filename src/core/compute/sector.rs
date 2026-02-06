use crate::core::bits::{Blade, ControlMask, Sector};

/// Exact projector composition:
///
/// Π_x Π_y = δ_{x,y} Π_x
#[inline(always)]
pub fn compose_exact_sectors(a: Sector, b: Sector) -> Option<Sector> {
    if a == b {
        Some(a)
    } else {
        None
    }
}

/// Exact projector orthogonality:
///
/// Π_x Π_y = 0 for x != y
#[inline(always)]
pub fn sectors_orthogonal(a: Sector, b: Sector) -> bool {
    a != b
}

#[inline(always)]
pub fn same_sector(a: Sector, b: Sector) -> bool {
    a == b
}

#[inline(always)]
pub fn control_matches(control: ControlMask, sector: Sector) -> bool {
    (sector.bits & control.mask) == (control.bits & control.mask)
}

#[inline(always)]
fn active_mask(n: u8) -> u64 {
    if n == 64 {
        u64::MAX
    } else {
        (1u64 << n) - 1
    }
}

/// Packed exact-sector rewrite:
///
///     B Π_x = (±) Π_y B'
///
/// This is the fast no-loop version for exact primitive idempotents.
/// It uses whole-mask formulas instead of per-qubit local rewrites.
///
/// Semantics:
/// - `blade.x` marks local E2 or J factors, which flip sector bits
/// - pure E1 factors contribute eigen-signs on Q sectors and then disappear
/// - J factors remain in the blade
#[inline(always)]
pub fn push_blade_through_right_exact_sector(
    blade: Blade,
    sector: Sector,
) -> (Sector, Blade, bool) {
    let mask = active_mask(sector.n);

    let x = blade.x & mask;
    let z = blade.z & mask;
    let bits = sector.bits & mask;

    // E2 and J flip sector bits.
    let new_sector_bits = bits ^ x;

    // Pure E1 contributes eigen-sign on Q sectors.
    let e1_only = z & !x;
    let minus = ((e1_only & bits).count_ones() & 1) != 0;

    // Pure E1 collapses into the projector eigenvalue and disappears.
    // J survives because it has both x and z bits.
    let new_blade = Blade {
        x,
        z: z & x,
        sign: false,
    };

    let new_sector = Sector::from_bits(new_sector_bits, sector.n);

    (new_sector, new_blade, blade.sign ^ minus)
}

/// Packed exact-sector rewrite:
///
///     Π_x B = (±) B' Π_y
///
/// For the current exact-sector local rules, this has the same packed form
/// as the right rewrite.
#[inline(always)]
pub fn push_blade_through_left_exact_sector(
    sector: Sector,
    blade: Blade,
) -> (Blade, Sector, bool) {
    let mask = active_mask(sector.n);

    let x = blade.x & mask;
    let z = blade.z & mask;
    let bits = sector.bits & mask;

    let new_sector_bits = bits ^ x;

    let e1_only = z & !x;
    let minus = ((e1_only & bits).count_ones() & 1) != 0;

    let new_blade = Blade {
        x,
        z: z & x,
        sign: false,
    };

    let new_sector = Sector::from_bits(new_sector_bits, sector.n);

    (new_blade, new_sector, blade.sign ^ minus)
}

/// Backward-compatible aliases.
/// These now route to the packed exact-sector kernels.
#[inline(always)]
pub fn apply_blade_to_right_sector(blade: Blade, sector: Sector) -> (Sector, Blade, bool) {
    push_blade_through_right_exact_sector(blade, sector)
}

#[inline(always)]
pub fn apply_blade_to_left_sector(sector: Sector, blade: Blade) -> (Blade, Sector, bool) {
    push_blade_through_left_exact_sector(sector, blade)
}