use crate::core::bits::{Blade, Rotor, Sector};

//////////////////////////////////////////////////////////////
// BITWISE HELPERS
//////////////////////////////////////////////////////////////

/// Odd parity of a 64-bit mask.
/// 
/// Computes whether the number of 1-bits in `x` is odd.
/// This is crucial for tracking sign flips when anticommuting
/// Pauli operators: two anticommuting Paulis contribute 1 to the exponent
/// of the imaginary unit i (a -1 phase when two anticommutations occur).
///
/// count_ones() counts total 1-bits, & 1 extracts the low bit (odd/even),
/// and == 1 converts to bool (true if odd).
#[inline(always)]
pub fn parity_u64(x: u64) -> bool {
    // Count number of set bits, then check if count is odd (LSB = 1)
    (x.count_ones() & 1) == 1
}

/// Mask of positions where the local blade is exactly E1,
/// i.e. `(x_i, z_i) = (0, 1)`.
///
/// E1 represents the Pauli Z matrix, which anticommutes only with X.
/// This mask identifies all E1 basis elements in the blade, which are
/// the only parts that anticommute with identity sectors (S_i = 0).
/// Used in sign tracking: anticommutation contributes to phase.
///
/// z & !x selects bits where Z=1 AND X=0, which is exactly the E1 pattern.
#[inline(always)]
pub fn pure_e1_mask(x: u64, z: u64) -> u64 {
    // Bitwise AND of Z with bitwise NOT of X
    // Selects positions where Z=1 and X=0 (the E1 = Z basis element)
    z & !x
}

//////////////////////////////////////////////////////////////
// BLADE/SECTOR BITWISE REWRITE CORE
//////////////////////////////////////////////////////////////

/// Packed exact-sector rewrite law for the blade lane.
///
/// Given:
/// - sector bits `S`
/// - blade masks `(X, Z)`
/// - incoming sign bit `sign`
///
/// computes:
///
///     sector' = S ⊕ X
///     X'      = X
///     Z'      = Z ∧ X
///     sign'   = sign ⊕ parity((Z ∧ ¬X) ∧ S)
///
/// This law encodes pushing a blade B through a left-exact sector Π_S.
/// The sector projects onto subspace where exactly the bits in S are active.
/// 
/// - sector' = S ⊕ X: The sector's active bits flip by wherever the blade has X
///   (since X anticommutes with both X and Z, flipping the active subspace)
/// - X' = X: The X pattern of the blade is unchanged (invariant under Peirce)
/// - Z' = Z ∧ X: The Z pattern is reduced to overlap with X only.
///   Z alone (E1) anticommutes with active sectors, but Z∧X (Y) doesn't.
/// - sign': The phase accumulates from anticommutation of E1 positions
///   (Z ∧ ¬X) with the active sector bits (S). Each anticommutation
///   contributes a factor of i, and two anticommutations give -1.
#[inline(always)]
pub fn rewrite_bitwise(
    sector_bits: u64,
    blade_x: u64,
    blade_z: u64,
    sign: bool,
) -> (u64, u64, u64, bool) {
    // Compute mask of E1 (pure Z) positions: where Z=1 and X=0
    let e1_mask = pure_e1_mask(blade_x, blade_z);

    // Sector active bits flip by the X pattern of the incoming blade
    let new_sector_bits = sector_bits ^ blade_x;
    // X component of blade passes through unchanged
    let new_x = blade_x;
    // Z component reduces to the overlap with X (only Y = iZ remains, not pure Z)
    let new_z = blade_z & blade_x;
    // Sign flips if there's odd parity of anticommutations:
    // (pure E1 positions) ∧ (active sector bits)
    let new_sign = sign ^ parity_u64(e1_mask & sector_bits);

    (new_sector_bits, new_x, new_z, new_sign)
}

#[inline(always)]
pub fn rewrite_right_bitwise(
    sector_bits: u64,
    blade_x: u64,
    blade_z: u64,
    sign: bool,
) -> (u64, u64, u64, bool) {
    // Right sector rewrite: blade appears on the right, pushes through sector from right
    // Same rewrite law applies (sector algebra is symmetric in this context)
    rewrite_bitwise(sector_bits, blade_x, blade_z, sign)
}

#[inline(always)]
pub fn rewrite_left_bitwise(
    sector_bits: u64,
    blade_x: u64,
    blade_z: u64,
    sign: bool,
) -> (u64, u64, u64, bool) {
    // Left sector rewrite: blade appears on the left, pushes through sector from left
    // Same rewrite law applies (sector algebra is symmetric in this context)
    rewrite_bitwise(sector_bits, blade_x, blade_z, sign)
}

//////////////////////////////////////////////////////////////
// ROTOR/SECTOR BITWISE LANE
//////////////////////////////////////////////////////////////

/// Rotor lane is independent of blade transport.
///
/// For now, sector-controlled right rotors do not undergo packed
/// transport/collapse analogous to blades. They remain a separate
/// right-phase payload. So this is identity on packed rotor masks.
///
/// In standard Clifford theory, rotors (exponentials of bivectors, like e^(iπ/4 J_i))
/// commute with all Pauli operators, so they don't interact with sectors.
/// Their signs are tracked separately and combined at the end of the computation.
#[inline(always)]
pub fn rewrite_rotor_bitwise(
    sector_bits: u64,
    q1_mask: u64,
    q2_mask: u64,
    q3_mask: u64,
    sign: bool,
) -> (u64, u64, u64, u64, bool) {
    // Rotors pass through sectors unchanged (no anticommutation with Paulis)
    // Return all values as-is: identity transformation
    (sector_bits, q1_mask, q2_mask, q3_mask, sign)
}

#[inline(always)]
pub fn rewrite_right_rotor_bitwise(
    sector_bits: u64,
    q1_mask: u64,
    q2_mask: u64,
    q3_mask: u64,
    sign: bool,
) -> (u64, u64, u64, u64, bool) {
    // Right rotor rewrite: rotor on the right, passes through sector unchanged
    rewrite_rotor_bitwise(sector_bits, q1_mask, q2_mask, q3_mask, sign)
}

#[inline(always)]
pub fn rewrite_left_rotor_bitwise(
    sector_bits: u64,
    q1_mask: u64,
    q2_mask: u64,
    q3_mask: u64,
    sign: bool,
) -> (u64, u64, u64, u64, bool) {
    // Left rotor rewrite: rotor on the left, passes through sector unchanged
    rewrite_rotor_bitwise(sector_bits, q1_mask, q2_mask, q3_mask, sign)
}

//////////////////////////////////////////////////////////////
// BITWISE WRAPPERS FOR BLADE / SECTOR TYPES
//////////////////////////////////////////////////////////////

#[inline]
pub fn push_blade_through_right_sector_bitwise(
    blade: Blade,
    sector: Sector,
) -> (Sector, Blade, bool) {
    // Apply the bitwise rewrite law to push blade through right sector
    let (new_sector_bits, new_x, new_z, new_sign) =
        rewrite_right_bitwise(sector.bits, blade.x, blade.z, blade.sign);

    // Reconstruct the Sector and Blade types from the computed bitwise values
    // The reconstructed blade has sign=false because the sign is now tracked separately
    (
        Sector::new(new_sector_bits, sector.n),  // New sector with updated bits
        Blade::new(new_x, new_z, false),         // New blade with computed X, Z, and no sign
        new_sign,                                // Accumulated sign from anticommutation
    )
}

#[inline]
pub fn push_blade_through_left_sector_bitwise(
    sector: Sector,
    blade: Blade,
) -> (Blade, Sector, bool) {
    // Apply the bitwise rewrite law to push blade through left sector
    let (new_sector_bits, new_x, new_z, new_sign) =
        rewrite_left_bitwise(sector.bits, blade.x, blade.z, blade.sign);

    // Reconstruct the Blade and Sector types from the computed bitwise values
    // Note the different order of return compared to right version: (Blade, Sector, bool)
    (
        Blade::new(new_x, new_z, false),         // New blade with computed X, Z, and no sign
        Sector::new(new_sector_bits, sector.n),  // New sector with updated bits
        new_sign,                                // Accumulated sign from anticommutation
    )
}

//////////////////////////////////////////////////////////////
// BITWISE WRAPPERS FOR ROTOR / SECTOR TYPES
//////////////////////////////////////////////////////////////

#[inline]
pub fn push_rotor_through_right_sector_bitwise(
    rotor: Rotor,
    sector: Sector,
) -> (Sector, Rotor, bool) {
    // Apply the rotor rewrite law (which is identity since rotors commute with Paulis)
    let (new_sector_bits, q1, q2, q3, new_sign) = rewrite_right_rotor_bitwise(
        sector.bits,
        rotor.q1_mask,
        rotor.q2_mask,
        rotor.q3_mask,
        rotor.sign,
    );

    // Reconstruct the Sector and Rotor types
    // The rotor sign is set to false in the rotor object because we track the
    // combined sign separately in the output boolean (rotor commutes, so no additional phases)
    (
        Sector::new(new_sector_bits, sector.n),  // Sector passes through unchanged
        Rotor {
            q1_mask: q1,                         // q1 component unchanged
            q2_mask: q2,                         // q2 component unchanged
            q3_mask: q3,                         // q3 component unchanged
            sign: false,                         // Sign extracted and returned separately
        },
        new_sign,                                // Sign: no new phases from commuting rotor
    )
}

#[inline]
pub fn push_rotor_through_left_sector_bitwise(
    sector: Sector,
    rotor: Rotor,
) -> (Rotor, Sector, bool) {
    // Apply the rotor rewrite law (which is identity since rotors commute with Paulis)
    let (new_sector_bits, q1, q2, q3, new_sign) = rewrite_left_rotor_bitwise(
        sector.bits,
        rotor.q1_mask,
        rotor.q2_mask,
        rotor.q3_mask,
        rotor.sign,
    );

    // Reconstruct the Rotor and Sector types
    // Note the different order of return compared to right version: (Rotor, Sector, bool)
    (
        Rotor {
            q1_mask: q1,                         // q1 component unchanged
            q2_mask: q2,                         // q2 component unchanged
            q3_mask: q3,                         // q3 component unchanged
            sign: false,                         // Sign extracted and returned separately
        },
        Sector::new(new_sector_bits, sector.n),  // Sector passes through unchanged
        new_sign,                                // Sign: no new phases from commuting rotor
    )
}