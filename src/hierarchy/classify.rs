//! Classification of operators into the Clifford hierarchy C_k.
//!
//! Design:
//!
//! 1. Try exact named recognition first:
//!    - C1: Pauli gates (X/Y/Z)
//!    - C2: Clifford gates (H/S/CNOT)
//!    - C3: third-level named gates (T, Controlled-S, Toffoli)
//!
//! 2. Expose a recursive hierarchy-depth interface:
//!
//!        level(U) = 1 + max_P level(U P U^\dagger)
//!
//!    over Pauli generators P = X_i, Z_i.
//!
//! For now, exact recursive descent is only implemented for named operators,
//! because generic multivector conjugation is not yet available.
//!
//! This file is therefore both:
//! - a usable classifier today
//! - the natural place to extend once the full conjugation engine lands.

use crate::core::bits::Blade;
use crate::core::ir::Multivector;
use crate::hierarchy::levels::c1::{C1, C1Gate};
use crate::hierarchy::levels::c2::{C2, C2Gate};
use crate::hierarchy::levels::c3::{C3, C3Gate};

/// Result of classifying an operator in the Clifford hierarchy.
#[derive(Clone, Debug, PartialEq)]
pub enum ClassifiedOp {
    /// Exact Pauli-layer gate.
    C1(C1Gate),

    /// Exact Clifford-layer gate.
    C2(C2Gate),

    /// Exact third-level gate.
    C3(C3Gate),

    /// Operator appears to lie in a higher level, but no exact named
    /// decomposition is currently available.
    Higher {
        min_level: usize,
        op: Multivector,
    },

    /// Could not currently determine hierarchy membership.
    Unknown(Multivector),
}

impl ClassifiedOp {
    /// Best-known hierarchy level attached to this classification result.
    pub fn level(&self) -> Option<usize> {
        match self {
            ClassifiedOp::C1(_) => Some(1),
            ClassifiedOp::C2(_) => Some(2),
            ClassifiedOp::C3(_) => Some(3),
            ClassifiedOp::Higher { min_level, .. } => Some(*min_level),
            ClassifiedOp::Unknown(_) => None,
        }
    }
}

/// Classify an operator using exact recognizers first, then structural fallbacks.
///
/// This version does not require the caller to specify `n_qubits`.
pub fn classify(op: &Multivector) -> ClassifiedOp {
    let inferred_n = infer_num_qubits(op).max(1);
    classify_with_qubits(op, inferred_n)
}

/// Classify an operator with an explicit qubit count.
///
/// This is the main entry point if you already know the ambient width.
pub fn classify_with_qubits(op: &Multivector, n_qubits: usize) -> ClassifiedOp {
    if let Some(g) = try_classify_c1(op) {
        return ClassifiedOp::C1(g);
    }

    if let Some(g) = try_classify_c2(op) {
        return ClassifiedOp::C2(g);
    }

    if let Some(g) = try_classify_c3(op) {
        return ClassifiedOp::C3(g);
    }

    // Structural fallback:
    //
    // If it is clearly C3-shaped, mark it as "at least level 3".
    if C3::is_c3_shaped(op) {
        return ClassifiedOp::Higher {
            min_level: 3,
            op: op.clone(),
        };
    }

    // If it is Clifford-shaped but not an exact named C2 gate, mark it as at least C2.
    if C2::is_clifford_shaped(op) {
        return ClassifiedOp::Higher {
            min_level: 2,
            op: op.clone(),
        };
    }

    // Try recursive named-depth classification.
    if let Some(depth) = hierarchy_depth_named(op, n_qubits, 6) {
        return match depth {
            1 => {
                // In practice this path should have already been caught above.
                ClassifiedOp::Higher {
                    min_level: 1,
                    op: op.clone(),
                }
            }
            2 => ClassifiedOp::Higher {
                min_level: 2,
                op: op.clone(),
            },
            3 => ClassifiedOp::Higher {
                min_level: 3,
                op: op.clone(),
            },
            k => ClassifiedOp::Higher {
                min_level: k,
                op: op.clone(),
            },
        };
    }

    ClassifiedOp::Unknown(op.clone())
}

/// Return the best-known hierarchy depth of `op`.
///
/// Current behavior:
/// - exact named C1/C2/C3 are recognized immediately
/// - otherwise tries named recursive descent if possible
/// - otherwise returns `None`
pub fn hierarchy_depth(op: &Multivector, n_qubits: usize, max_depth: usize) -> Option<usize> {
    if max_depth == 0 {
        return None;
    }

    if try_classify_c1(op).is_some() {
        return Some(1);
    }
    if try_classify_c2(op).is_some() {
        return Some(2);
    }
    if try_classify_c3(op).is_some() {
        return Some(3);
    }

    hierarchy_depth_named(op, n_qubits, max_depth)
}

/// Recursive hierarchy-depth computation for currently recognized named operators.
///
/// This uses the defining recursion:
///
///     level(U) = 1 + max_P level(U P U^\dagger)
///
/// but only when `U` is a currently recognized named operator for which we have
/// explicit symbolic action on Pauli generators.
pub fn hierarchy_depth_named(
    op: &Multivector,
    n_qubits: usize,
    max_depth: usize,
) -> Option<usize> {
    if max_depth == 0 {
        return None;
    }

    if try_classify_c1(op).is_some() {
        return Some(1);
    }
    if try_classify_c2(op).is_some() {
        return Some(2);
    }
    if try_classify_c3(op).is_some() {
        return Some(3);
    }

    let named = NamedOp::from_multivector(op)?;
    recursive_named_depth(named, n_qubits, max_depth)
}

/// Try to classify as an exact C1 gate.
fn try_classify_c1(op: &Multivector) -> Option<C1Gate> {
    if op.terms.len() != 1 {
        return None;
    }

    let term = &op.terms[0];

    if term.left.is_some() || term.right.is_some() {
        return None;
    }

    let is_bare = term
        .rotor
        .map(|r| r.is_identity())
        .unwrap_or(true);

    if is_bare {
        return C1::get_gate(term.blade);
    }

    if let Some(rotor) = term.rotor {
        if let Some(C1Gate::Y { qubit }) = C1::get_gate(term.blade) {
            if rotor == crate::core::bits::Rotor::j_at(qubit)
                && (term.coeff - (-1.0)).abs() < 1e-9
            {
                return Some(C1Gate::Y { qubit });
            }
        }
    }

    None
}

/// Try to classify as an exact C2 gate.
fn try_classify_c2(op: &Multivector) -> Option<C2Gate> {
    C2::get_gate(op)
}

/// Try to classify as an exact C3 gate.
fn try_classify_c3(op: &Multivector) -> Option<C3Gate> {
    C3::get_gate(op)
}

/// Internal tagged operator vocabulary used by recursive descent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NamedOp {
    C1(C1Gate),
    C2(C2Gate),
    C3(C3Gate),
}

impl NamedOp {
    fn from_multivector(op: &Multivector) -> Option<Self> {
        if let Some(g) = try_classify_c1(op) {
            return Some(Self::C1(g));
        }
        if let Some(g) = try_classify_c2(op) {
            return Some(Self::C2(g));
        }
        if let Some(g) = try_classify_c3(op) {
            return Some(Self::C3(g));
        }
        None
    }

    #[allow(dead_code)]
    fn level(self) -> usize {
        match self {
            NamedOp::C1(_) => 1,
            NamedOp::C2(_) => 2,
            NamedOp::C3(_) => 3,
        }
    }
}

/// Recursive depth for named gates only.
///
/// For now:
/// - C1/C2/C3 exact named operators are returned directly
/// - this function also contains the recursion skeleton for future extension
fn recursive_named_depth(op: NamedOp, n_qubits: usize, max_depth: usize) -> Option<usize> {
    if max_depth == 0 {
        return None;
    }

    match op {
        NamedOp::C1(_) => Some(1),
        NamedOp::C2(_) => Some(2),
        NamedOp::C3(_) => Some(3),
    }
    .or_else(|| {
        // Future generic recursion goes here:
        //
        // let mut max_image_level = 1usize;
        // for p in pauli_generators(n_qubits) {
        //     let image = conjugate_pauli_by_named(op, p)?;
        //     let image_depth = recursive_named_depth(image, n_qubits, max_depth - 1)?;
        //     max_image_level = max_image_level.max(image_depth);
        // }
        // Some(max_image_level + 1)
        //
        // At the moment all currently supported named gates have direct exact levels,
        // so we do not need this branch yet.
        let _ = n_qubits;
        None
    })
}

/// Generate the Pauli generators X_i and Z_i.
pub fn pauli_generators(n_qubits: usize) -> Vec<Blade> {
    let mut out = Vec::with_capacity(2 * n_qubits);
    for i in 0..n_qubits {
        out.push(Blade::x(i));
        out.push(Blade::z(i));
    }
    out
}

/// Infer the ambient qubit count from the support appearing in an operator.
///
/// This is conservative:
/// - if the operator is empty, returns 0
/// - otherwise returns 1 + highest referenced qubit index
pub fn infer_num_qubits(op: &Multivector) -> usize {
    let mut max_bit: i32 = -1;

    for t in &op.terms {
        let support =
            t.blade.x
            | t.blade.z
            | t.left.map(|s| s.bits).unwrap_or(0)
            | t.right.map(|s| s.bits).unwrap_or(0)
            | t.rotor.map(|r| r.q1_mask | r.q2_mask | r.q3_mask).unwrap_or(0);

        if support != 0 {
            let bit = 63 - support.leading_zeros() as i32;
            max_bit = max_bit.max(bit);
        }

        if let Some(left) = t.left {
            if left.n > 0 {
                max_bit = max_bit.max(left.n as i32 - 1);
            }
        }

        if let Some(right) = t.right {
            if right.n > 0 {
                max_bit = max_bit.max(right.n as i32 - 1);
            }
        }
    }

    if max_bit < 0 {
        0
    } else {
        (max_bit as usize) + 1
    }
}

/// Check whether the operator is at most level `k`, as far as current recognition can tell.
pub fn is_at_most_level(op: &Multivector, n_qubits: usize, k: usize) -> bool {
    match hierarchy_depth(op, n_qubits, k) {
        Some(level) => level <= k,
        None => false,
    }
}

/// Check whether the operator is exactly level `k`, as far as current recognition can tell.
pub fn is_exact_level(op: &Multivector, n_qubits: usize, k: usize) -> bool {
    matches!(hierarchy_depth(op, n_qubits, k), Some(level) if level == k)
}

/*
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_pauli_x() {
        let op = C1::x_gate(0);
        assert_eq!(classify(&op), ClassifiedOp::C1(C1Gate::X { qubit: 0 }));
    }

    #[test]
    fn classify_pauli_y() {
        let op = C1::y_gate(1);
        assert_eq!(classify(&op), ClassifiedOp::C1(C1Gate::Y { qubit: 1 }));
    }

    #[test]
    fn classify_pauli_z() {
        let op = C1::z_gate(2);
        assert_eq!(classify(&op), ClassifiedOp::C1(C1Gate::Z { qubit: 2 }));
    }

    #[test]
    fn classify_h() {
        let op = C2::h_gate(0);
        assert_eq!(classify(&op), ClassifiedOp::C2(C2Gate::H { qubit: 0 }));
    }

    #[test]
    fn classify_s() {
        let op = C2::s_gate(1);
        assert_eq!(classify(&op), ClassifiedOp::C2(C2Gate::S { qubit: 1 }));
    }

    #[test]
    fn classify_cnot() {
        let op = C2::cnot_gate(0, 3);
        assert_eq!(
            classify(&op),
            ClassifiedOp::C2(C2Gate::CNOT {
                control: 0,
                target: 3
            })
        );
    }

    #[test]
    fn classify_t() {
        let op = C3::t_gate(2);
        assert_eq!(classify(&op), ClassifiedOp::C3(C3Gate::T { qubit: 2 }));
    }

    #[test]
    fn classify_controlled_s() {
        let op = C3::controlled_s_gate(1, 4);
        assert_eq!(
            classify(&op),
            ClassifiedOp::C3(C3Gate::ControlledS {
                control: 1,
                target: 4
            })
        );
    }

    #[test]
    fn classify_toffoli() {
        let op = C3::toffoli_gate(0, 2, 5);
        assert_eq!(
            classify(&op),
            ClassifiedOp::C3(C3Gate::Toffoli {
                control1: 0,
                control2: 2,
                target: 5
            })
        );
    }

    #[test]
    fn hierarchy_depth_named_levels_are_correct() {
        assert_eq!(hierarchy_depth(&C1::x_gate(0), 1, 6), Some(1));
        assert_eq!(hierarchy_depth(&C2::s_gate(0), 1, 6), Some(2));
        assert_eq!(hierarchy_depth(&C3::t_gate(0), 1, 6), Some(3));
    }

    #[test]
    fn pauli_generators_count_is_2n() {
        let gens = pauli_generators(4);
        assert_eq!(gens.len(), 8);
        assert!(gens.contains(&Blade::x(0)));
        assert!(gens.contains(&Blade::z(0)));
        assert!(gens.contains(&Blade::x(3)));
        assert!(gens.contains(&Blade::z(3)));
    }

    #[test]
    fn infer_num_qubits_works_for_cnot() {
        let op = C2::cnot_gate(2, 5);
        assert_eq!(infer_num_qubits(&op), 6);
    }

    #[test]
    fn infer_num_qubits_works_for_empty() {
        let op = Multivector::from_terms(vec![]);
        assert_eq!(infer_num_qubits(&op), 0);
    }

    #[test]
    fn unknown_fallback_for_unrecognized_operator() {
        let weird = Multivector::from_terms(vec![
            crate::core::ir::Term {
                left: None,
                blade: Blade::x(0),
                right: None,
                rotor: None,
                coeff: 0.123,
            },
            crate::core::ir::Term {
                left: None,
                blade: Blade::z(1),
                right: None,
                rotor: None,
                coeff: 0.456,
            },
        ]);

        match classify(&weird) {
            ClassifiedOp::Higher { .. } | ClassifiedOp::Unknown(_) => {}
            other => panic!("unexpected classification: {:?}", other),
        }
    }
}
*/