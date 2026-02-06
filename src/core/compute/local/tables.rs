use super::types::{LocalBlade, LocalRotor};

//////////////////////////////////////////////////////////////
// LOCAL BLADE MULTIPLICATION
//////////////////////////////////////////////////////////////

/// Multiply two one-qubit blades.
///
/// Returns:
/// - resulting blade
/// - sign flip (true = multiply by -1)
///
/// Table derived from:
///
/// e1^2 = 1
/// e2^2 = 1
/// J = e1 e2
/// e2 e1 = -J
#[inline(always)]
pub fn mul_local_blades(a: LocalBlade, b: LocalBlade) -> (LocalBlade, bool) {
    use LocalBlade::*;

    match (a, b) {
        (I, x) | (x, I) => (x, false),

        (E1, E1) => (I, false),
        (E2, E2) => (I, false),
        (J, J) => (I, true),

        (E1, E2) => (J, false),
        (E2, E1) => (J, true),

        (E1, J) => (E2, false),
        (J, E1) => (E2, true),

        (E2, J) => (E1, true),
        (J, E2) => (E1, false),
    }
}




//////////////////////////////////////////////////////////////
// LOCAL CLIFFORD ACTIONS
//////////////////////////////////////////////////////////////

/// Local action of Hadamard.
///
/// H:
///   e1 <-> e2
///   J -> -J
#[inline(always)]
pub fn local_h_action(blade: LocalBlade) -> (LocalBlade, bool) {
    use LocalBlade::*;

    match blade {
        I => (I, false),
        E1 => (E2, false),
        E2 => (E1, false),
        J => (J, true),
    }
}

/// Local action of S gate.
///
/// S:
///   e2 -> J
///   e1 -> e1
///   J -> -e2
#[inline(always)]
pub fn local_s_action(blade: LocalBlade) -> (LocalBlade, bool) {
    use LocalBlade::*;

    match blade {
        I => (I, false),
        E1 => (E1, false),
        E2 => (J, false),
        J => (E2, true),
    }
}



#[inline(always)]
pub fn mul_local_rotors(a: LocalRotor, b: LocalRotor) -> (LocalRotor, bool) {
    use LocalRotor::*;

    let ka = match a { I => 0, Q1 => 1, Q2 => 2, Q3 => 3 };
    let kb = match b { I => 0, Q1 => 1, Q2 => 2, Q3 => 3 };

    let sum = ka + kb;
    let sign = sum >= 4;
    let out = match sum % 4 {
        0 => I,
        1 => Q1,
        2 => Q2,
        3 => Q3,
        _ => unreachable!(),
    };
    (out, sign)
}