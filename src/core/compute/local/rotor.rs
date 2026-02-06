use crate::core::bits::Rotor;

use super::rewrite::{local_rotor_at, set_local_rotor};
use super::tables::mul_local_rotors;

/// Reference implementation using the local enum table.
/// Slower than `crate::core::compute::rotor::compose_rotor`,
/// but useful for tests and local reasoning.
pub fn compose_rotor_reference(a: Rotor, b: Rotor, n: u8) -> Rotor {
    let mut out = Rotor {
        q1_mask: 0,
        q2_mask: 0,
        q3_mask: 0,
        sign: a.sign ^ b.sign,
    };

    for i in 0..(n as usize) {
        let la = local_rotor_at(a, i);
        let lb = local_rotor_at(b, i);

        let (lc, sgn) = mul_local_rotors(la, lb);
        out = set_local_rotor(out, i, lc);
        out.sign ^= sgn;
    }

    out
}