use crate::core::bits::Rotor;

#[inline(always)]
fn mask_of_class(r: Rotor, class: u8) -> u64 {
    match class {
        1 => r.q1_mask,
        2 => r.q2_mask,
        3 => r.q3_mask,
        _ => 0,
    }
}

/// Identity rotor.
#[inline(always)]
pub fn rotor_identity() -> Rotor {
    Rotor {
        q1_mask: 0,
        q2_mask: 0,
        q3_mask: 0,
        sign: false,
    }
}

/// Local quarter-turn rotor on one site (class 1).
#[inline(always)]
pub fn rotor_quarter_at(i: usize) -> Rotor {
    Rotor {
        q1_mask: 1u64 << i,
        q2_mask: 0,
        q3_mask: 0,
        sign: false,
    }
}

/// Local J rotor on one site (class 2).
#[inline(always)]
pub fn rotor_j_at(i: usize) -> Rotor {
    Rotor {
        q1_mask: 0,
        q2_mask: 1u64 << i,
        q3_mask: 0,
        sign: false,
    }
}

/// Local three-quarter-turn rotor on one site (class 3).
#[inline(always)]
pub fn rotor_three_quarter_at(i: usize) -> Rotor {
    Rotor {
        q1_mask: 0,
        q2_mask: 0,
        q3_mask: 1u64 << i,
        sign: false,
    }
}

/// Number of non-identity local rotor sites.
#[inline(always)]
pub fn rotor_support_size(r: Rotor) -> u32 {
    (r.q1_mask | r.q2_mask | r.q3_mask).count_ones()
}

/// Whether the rotor is the identity.
#[inline(always)]
pub fn rotor_is_identity(r: Rotor) -> bool {
    r.q1_mask == 0 && r.q2_mask == 0 && r.q3_mask == 0 && !r.sign
}

/// Local rotor class at one site:
/// 0 = identity, 1 = quarter-turn, 2 = J, 3 = three-quarter-turn.
#[inline(always)]
pub fn rotor_class_at(r: Rotor, i: usize) -> u8 {
    let bit = 1u64 << i;
    if (r.q1_mask & bit) != 0 {
        1
    } else if (r.q2_mask & bit) != 0 {
        2
    } else if (r.q3_mask & bit) != 0 {
        3
    } else {
        0
    }
}

/// Clear the sign into a separate return value.
#[inline(always)]
pub fn unsigned_rotor(r: Rotor) -> (Rotor, bool) {
    (
        Rotor {
            q1_mask: r.q1_mask,
            q2_mask: r.q2_mask,
            q3_mask: r.q3_mask,
            sign: false,
        },
        r.sign,
    )
}

/// Compose packed discrete right rotors exactly.
///
/// For each qubit independently, local quarter-turn classes add mod 8.
/// Any local class in the upper half (4..7) is folded into:
/// - a canonical local class in {0,1,2,3}
/// - a global sign contribution
pub fn compose_rotor(a: Rotor, b: Rotor) -> Rotor {
    let used_a = a.q1_mask | a.q2_mask | a.q3_mask;
    let used_b = b.q1_mask | b.q2_mask | b.q3_mask;

    let mut out_q1 = 0u64;
    let mut out_q2 = 0u64;
    let mut out_q3 = 0u64;
    let mut sign = a.sign ^ b.sign;

    // Bits used only by a keep their local class.
    out_q1 |= a.q1_mask & !used_b;
    out_q2 |= a.q2_mask & !used_b;
    out_q3 |= a.q3_mask & !used_b;

    // Bits used only by b keep their local class.
    out_q1 |= b.q1_mask & !used_a;
    out_q2 |= b.q2_mask & !used_a;
    out_q3 |= b.q3_mask & !used_a;

    // Overlapping bits: add local classes.
    for ka in 1u8..=3 {
        for kb in 1u8..=3 {
            let overlap = mask_of_class(a, ka) & mask_of_class(b, kb);
            if overlap == 0 {
                continue;
            }

            let sum = ka + kb;

            // Fold upper half into global sign.
            if sum >= 4 && (overlap.count_ones() & 1) != 0 {
                sign = !sign;
            }

            match sum % 4 {
                0 => {}
                1 => out_q1 |= overlap,
                2 => out_q2 |= overlap,
                3 => out_q3 |= overlap,
                _ => unreachable!(),
            }
        }
    }

    Rotor {
        q1_mask: out_q1,
        q2_mask: out_q2,
        q3_mask: out_q3,
        sign,
    }
}