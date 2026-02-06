use crate::core::bits::{Blade, Rotor, Sector};
use crate::core::compute::blade::gp_blade;
use crate::core::compute::local::{
    compose_rotor_reference,
    push_blade_through_left_sector,
    push_blade_through_right_sector,
    push_rotor_through_left_sector,
    push_rotor_through_right_sector,
};
use crate::core::ir::Term;

#[inline(always)]
fn infer_term_width(a: &Term, b: &Term) -> u8 {
    let mut max_bit: i32 = -1;

    let masks = [
        a.blade.x,
        a.blade.z,
        b.blade.x,
        b.blade.z,
        a.left.map(|s| s.bits).unwrap_or(0),
        a.right.map(|s| s.bits).unwrap_or(0),
        b.left.map(|s| s.bits).unwrap_or(0),
        b.right.map(|s| s.bits).unwrap_or(0),
        a.rotor
            .map(|r| r.q1_mask | r.q2_mask | r.q3_mask)
            .unwrap_or(0),
        b.rotor
            .map(|r| r.q1_mask | r.q2_mask | r.q3_mask)
            .unwrap_or(0),
    ];

    for m in masks {
        if m != 0 {
            let bit = 63 - m.leading_zeros() as i32;
            max_bit = max_bit.max(bit);
        }
    }

    let max_n = [
        a.left.map(|s| s.n).unwrap_or(0),
        a.right.map(|s| s.n).unwrap_or(0),
        b.left.map(|s| s.n).unwrap_or(0),
        b.right.map(|s| s.n).unwrap_or(0),
    ]
    .into_iter()
    .max()
    .unwrap_or(0) as i32;

    let inferred = if max_bit < 0 { 0 } else { max_bit + 1 };
    max_n.max(inferred) as u8
}

#[inline(always)]
fn id_rotor() -> Rotor {
    Rotor {
        q1_mask: 0,
        q2_mask: 0,
        q3_mask: 0,
        sign: false,
    }
}

/// Slow, explicit reference multiplication for two terms.
///
/// Semantics:
/// - exact inner-sector match required
/// - blade lane multiplies with `gp_blade`
/// - rotor lane composes with `compose_rotor_reference`
/// - then we do the same local reductions as the fast path,
///   but through the reference local rewrite functions
///
/// Coefficients are NOT multiplied here. This is a structural kernel,
/// just like `gp_term`.
pub fn gp_term_reference(a: &Term, b: &Term) -> Option<Term> {
    let n = infer_term_width(a, b).max(1);

    // Track which sectors were originally None (unconstrained) before conversion to Sector::new(0, n)
    let right_a_was_none = a.right.is_none();
    let left_b_was_none = b.left.is_none();

    let left_a = a.left.unwrap_or_else(|| Sector::new(0, n));
    let right_a = a.right.unwrap_or_else(|| Sector::new(0, n));
    let left_b = b.left.unwrap_or_else(|| Sector::new(0, n));
    let right_b = b.right.unwrap_or_else(|| Sector::new(0, n));

    if right_a != left_b {
        return None;
    }

    let blade_prod = gp_blade(a.blade, b.blade);

    let rotor_a = a.rotor.unwrap_or_else(id_rotor);
    let rotor_b = b.rotor.unwrap_or_else(id_rotor);
    let rotor_prod = compose_rotor_reference(rotor_a, rotor_b, n);

    // Start with outer sectors and raw payloads.
    let mut left = left_a;
    let mut right = right_b;

    let mut blade = Blade::new(blade_prod.x, blade_prod.z, false);
    let mut sign = blade_prod.sign ^ rotor_prod.sign;

    let mut rotor = Rotor {
        q1_mask: rotor_prod.q1_mask,
        q2_mask: rotor_prod.q2_mask,
        q3_mask: rotor_prod.q3_mask,
        sign: false,
    };

    // Equivalence class reduction: only apply when BOTH INTERMEDIATE sectors were explicitly constrained.
    // If either intermediate sector is unconstrained (came from None), it represents  
    // a "global to global" or "constrained to global" composition that should preserve blade structure.
    // The sector-based rewriting assumes all sectors are explicitly specified Peirce projectors.
    if right_a_was_none || left_b_was_none {
        // At least one intermediate sector was unconstrained: skip reductions
        // Keep blade_prod as the final blade without sector-based equivalence reduction
    } else {
        // Both intermediate sectors were explicitly constrained: apply reductions
        let (new_right, new_blade, s1) = push_blade_through_right_sector(blade, right);
        right = new_right;
        blade = new_blade;
        sign ^= s1;

        let (new_blade, new_left, s2) = push_blade_through_left_sector(left, blade);
        blade = new_blade;
        left = new_left;
        sign ^= s2;

        let (new_right, new_rotor, s3) = push_rotor_through_right_sector(rotor, right);
        right = new_right;
        rotor = new_rotor;
        sign ^= s3;

        let (new_rotor, new_left, s4) = push_rotor_through_left_sector(left, rotor);
        rotor = new_rotor;
        left = new_left;
        sign ^= s4;
    }

    let out_left = if left.bits == 0 { None } else { Some(left) };
    let out_right = if right.bits == 0 { None } else { Some(right) };

    let out_rotor = if (rotor.q1_mask | rotor.q2_mask | rotor.q3_mask) == 0 {
        None
    } else {
        Some(rotor)
    };

    Some(Term {
        left: out_left,
        blade: Blade::new(blade.x, blade.z, sign),
        right: out_right,
        rotor: out_rotor,
        coeff: 1.0,
    })
}