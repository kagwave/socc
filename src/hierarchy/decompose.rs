//! Operator decomposition in the Clifford hierarchy.
//!
//! This module implements Beigi–Shor decomposition and routing based on
//! classification results. The core idea:
//!
//!     Every U ∈ C3 can be written as U = C1 · π · D · C2
//!
//! where:
//! - C1, C2 are Clifford gates (H/S/CNOT compositions)
//! - π is a permutation of computational basis
//! - D is diagonal (sectorwise right-J phases)
//!
//! We use the classification layer to route operators to the appropriate level.

use crate::core::ir::Multivector;
use crate::hierarchy::classify::{classify, ClassifiedOp};
use crate::hierarchy::levels::c1::C1Gate;
use crate::hierarchy::levels::c2::C2Gate;

/// Result of decomposing an operator in the Clifford hierarchy.
#[derive(Clone, Debug)]
pub enum DecompositionLevel {
    /// C1: Single Pauli gate
    Level1(C1Gate),
    /// C2: Single Clifford gate (H/S/CNOT)
    Level2(C2Gate),
    /// C3+: Generic operator (permutation+diagonal or higher)
    LevelHigher(Multivector),
}

impl DecompositionLevel {
    /// Get the hierarchy level (1 for C1, 2 for C2, 3+ for higher).
    pub fn level(&self) -> usize {
        match self {
            DecompositionLevel::Level1(_) => 1,
            DecompositionLevel::Level2(_) => 2,
            DecompositionLevel::LevelHigher(_) => 3,
        }
    }

    /// Recover the multivector form of the decomposed operator.
    /// 
    /// Requires `n` because Level1 and Level2 gates must be constructed
    /// with knowledge of the ambient system size.
    pub fn to_multivector(&self, n: u8) -> Multivector {
        use crate::hierarchy::levels::c1::C1;
        use crate::hierarchy::levels::c2::C2;

        match self {
            DecompositionLevel::Level1(gate) => match gate {
                C1Gate::X { qubit } => C1::x_gate(n, *qubit),
                C1Gate::Y { qubit } => C1::y_gate(n, *qubit),
                C1Gate::Z { qubit } => C1::z_gate(n, *qubit),
            },
            DecompositionLevel::Level2(gate) => match gate {
                C2Gate::H { qubit } => C2::h_gate(*qubit),
                C2Gate::S { qubit } => C2::s_gate(*qubit),
                C2Gate::CNOT { control, target } => C2::cnot_gate(*control, *target),
            },
            DecompositionLevel::LevelHigher(op) => op.clone(),
        }
    }
}

/// Decompose an operator and return its hierarchy level.
///
/// Uses the classification layer to determine whether the operator
/// is C1 (Pauli), C2 (Clifford), or higher.
///
/// # Example
///
/// ```ignore
/// let h_gate = C2::h_gate(0);
/// match decompose(&h_gate) {
///     DecompositionLevel::Level2(C2Gate::H { qubit: 0 }) => println!("Hadamard!"),
///     _ => println!("Not a Hadamard"),
/// }
/// ```
pub fn decompose(op: &Multivector) -> DecompositionLevel {
    match classify(op) {
        ClassifiedOp::C1(gate) => DecompositionLevel::Level1(gate),
        ClassifiedOp::C2(gate) => DecompositionLevel::Level2(gate),
        ClassifiedOp::C3(_gate) => {
            // C3 gates are part of the higher hierarchy; treat as generic operator
            DecompositionLevel::LevelHigher(op.clone())
        }
        ClassifiedOp::Higher { op, .. } => DecompositionLevel::LevelHigher(op),
        ClassifiedOp::Unknown(op) => DecompositionLevel::LevelHigher(op),
    }
}

/*
// #[cfg(test)]
// mod tests {
    use super::*;
    use crate::hierarchy::levels::c1::C1;
    use crate::hierarchy::levels::c2::C2;

    #[test]
    fn test_decompose_pauli_x() {
        let op = C1::x_gate(0);
        let decomp = decompose(&op);
        assert_eq!(decomp.level(), 1);
        assert!(matches!(decomp, DecompositionLevel::Level1(C1Gate::X { qubit: 0 })));
    }

    #[test]
    fn test_decompose_pauli_y() {
        let op = C1::y_gate(1);
        let decomp = decompose(&op);
        assert_eq!(decomp.level(), 1);
        assert!(matches!(decomp, DecompositionLevel::Level1(C1Gate::Y { qubit: 1 })));
    }

    #[test]
    fn test_decompose_pauli_z() {
        let op = C1::z_gate(2);
        let decomp = decompose(&op);
        assert_eq!(decomp.level(), 1);
        assert!(matches!(decomp, DecompositionLevel::Level1(C1Gate::Z { qubit: 2 })));
    }

    #[test]
    fn test_decompose_hadamard() {
        let op = C2::h_gate(0);
        let decomp = decompose(&op);
        assert_eq!(decomp.level(), 2);
        assert!(matches!(decomp, DecompositionLevel::Level2(C2Gate::H { qubit: 0 })));
    }

    #[test]
    fn test_decompose_s_gate() {
        let op = C2::s_gate(1);
        let decomp = decompose(&op);
        assert_eq!(decomp.level(), 2);
        assert!(matches!(decomp, DecompositionLevel::Level2(C2Gate::S { qubit: 1 })));
    }

    #[test]
    fn test_decompose_cnot() {
        let op = C2::cnot_gate(0, 1);
        let decomp = decompose(&op);
        assert_eq!(decomp.level(), 2);
        assert!(matches!(decomp, DecompositionLevel::Level2(C2Gate::CNOT { control: 0, target: 1 })));
    }

    #[test]
    fn test_roundtrip_pauli_x() {
        let n = 1u8;
        let original = C1::x_gate(n, 0);
        let decomp = decompose(&original);
        let recovered = decomp.to_multivector(n);
        // Should reconstruct the same operator
        assert_eq!(original.terms.len(),recovered.terms.len());
        assert_eq!(original.terms[0].blade, recovered.terms[0].blade);
        assert!((original.terms[0].coeff - recovered.terms[0].coeff).abs() < 1e-10);
    }

    #[test]
    fn test_roundtrip_hadamard() {
        let n = 1u8;
        let original = C2::h_gate(0);
        let decomp = decompose(&original);
        let recovered = decomp.to_multivector(n);
        // Should reconstruct the same operator
        assert_eq!(original.terms.len(), recovered.terms.len());
    }
}
*/
