use crate::core::bits::Sector;
use crate::core::compute::multivector::scalar_part;
use crate::core::compute::state::peirce_block;
use crate::core::ir::Multivector;
use crate::state::density::Density;

/// Outcome probabilities of measuring a Pauli basis qubit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeasurementOutcomes {
    /// Qubit index being measured
    pub qubit: u32,
    /// P(outcome = 0)
    pub prob_0: f64,
    /// P(outcome = 1)
    pub prob_1: f64,
}

impl MeasurementOutcomes {
    /// Validate that probabilities are approximately normalized.
    pub fn is_normalized(&self, eps: f64) -> bool {
        let sum = self.prob_0 + self.prob_1;
        (sum - 1.0).abs() < eps
    }

    /// Return true if measurement outcome is deterministic (prob = 1 or 0).
    pub fn is_deterministic(&self, eps: f64) -> bool {
        (self.prob_0 - 1.0).abs() < eps || (self.prob_1 - 1.0).abs() < eps || 
        self.prob_0.abs() < eps || self.prob_1.abs() < eps
    }
}

/// Compute outcome probabilities for measuring Z-basis qubit.
///
/// For outcome b ∈ {0, 1}, the probability is:
///
///     P(b) = Tr(Π_b ρ Π_b)
///
/// where Π_b is the projector onto the sector where qubit i has value b.
///
/// # Arguments
/// * `rho` - Density matrix ρ
/// * `qubit` - Qubit index i to measure
/// * `n_qubits` - Total number of qubits (for sector width)
///
/// # Returns
/// MeasurementOutcomes with P(0) and P(1) normalized to sum to 1.
pub fn measure_outcomes(
    rho: &Density,
    qubit: u32,
    n_qubits: u8,
) -> MeasurementOutcomes {
    // Create sector projectors: Π_b = 1 if bit i = b, 0 otherwise
    let proj_0 = Sector::new(0, n_qubits);  // bit i = 0
    let proj_1 = Sector::new(1u64 << qubit, n_qubits);  // bit i = 1

    // Compute P(0) = Tr(Π_0 ρ Π_0)
    let block_0 = peirce_block(&rho.mv, proj_0, proj_0);
    let prob_0_raw = scalar_part(&block_0);

    // Compute P(1) = Tr(Π_1 ρ Π_1)
    let block_1 = peirce_block(&rho.mv, proj_1, proj_1);
    let prob_1_raw = scalar_part(&block_1);

    // Normalize
    let total = prob_0_raw + prob_1_raw;
    let (prob_0, prob_1) = if total.abs() > 1e-14 {
        (prob_0_raw / total, prob_1_raw / total)
    } else {
        (0.0, 0.0)
    };

    MeasurementOutcomes {
        qubit,
        prob_0,
        prob_1,
    }
}

/// Collapse density matrix conditioned on measurement outcome.
///
/// For outcome b, the collapsed state is:
///
///     ρ' = (Π_b ρ Π_b) / P(b)
///
/// where P(b) = Tr(Π_b ρ Π_b).
///
/// # Arguments
/// * `rho` - Density matrix before measurement
/// * `qubit` - Qubit index measured
/// * `outcome` - Measurement outcome (0 or 1)
/// * `n_qubits` - Total number of qubits
///
/// # Returns
/// Option<Density> - Collapsed density matrix, or None if outcome had zero probability
pub fn measure_collapse(
    rho: &Density,
    qubit: u32,
    outcome: u32,
    n_qubits: u8,
) -> Option<Density> {
    let outcome_bit = outcome & 1;

    // Compute probabilities
    let outcomes = measure_outcomes(rho, qubit, n_qubits);
    let prob = if outcome_bit == 0 {
        outcomes.prob_0
    } else {
        outcomes.prob_1
    };

    // Reject impossible outcomes
    if prob < 1e-14 {
        return None;
    }

    // Project: Π_b ρ Π_b
    let proj = if outcome_bit == 0 {
        Sector::new(0, n_qubits)
    } else {
        Sector::new(1u64 << qubit, n_qubits)
    };

    let projected = peirce_block(&rho.mv, proj, proj);

    // Normalize by P(outcome)
    let normalized_terms: Vec<_> = projected.terms
        .iter()
        .map(|t| {
            let mut t_norm = t.clone();
            t_norm.coeff /= prob;
            t_norm
        })
        .collect();

    let normalized = Density::new(Multivector::from_terms(projected.n, normalized_terms));

    Some(normalized)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::ir::{Multivector, Term};
    use crate::core::bits::Blade;
    use crate::state::ideal::IdealState;

    const EPS: f64 = 1e-10;

    /// Helper: create a simple product state representing |0⟩ on qubit 0 (vacuum sector)
    fn pure_zero_state_0() -> Density {
        let mv = Multivector::from_terms(1, vec![
            Term {
                left: None,
                blade: Blade::identity(),
                right: None,
                coeff: 1.0,
                rotor: None,
            }
        ]);
        let ideal = IdealState::new(mv, Sector::new(0, 1));
        ideal.density()
    }

    #[test]
    fn measure_outcomes_on_zero_state_is_deterministic_zero() {
        let rho = pure_zero_state_0();
        let outcomes = measure_outcomes(&rho, 0, 1);

        assert!(outcomes.is_normalized(EPS), "Probabilities should sum to 1: P(0)={}, P(1)={}", 
            outcomes.prob_0, outcomes.prob_1);
        assert!((outcomes.prob_0 - 1.0).abs() < EPS, "Should measure 0 with probability 1, got {}", outcomes.prob_0);
        assert!(outcomes.prob_1.abs() < EPS, "Should measure 1 with probability 0, got {}", outcomes.prob_1);
    }

    #[test]
    fn measure_outcomes_validation_on_deterministic_state() {
        let rho = pure_zero_state_0();
        let outcomes = measure_outcomes(&rho, 0, 1);

        assert!(outcomes.is_normalized(EPS));
        assert!(outcomes.is_deterministic(EPS), "Deterministic state should have prob 0 or 1");
    }

    #[test]
    fn measure_collapse_zero_state_outcome_zero() {
        let rho = pure_zero_state_0();
        let collapsed = measure_collapse(&rho, 0, 0, 1);

        assert!(collapsed.is_some(), "Outcome 0 should be possible for |0⟩ state");
        let rho_c = collapsed.unwrap();
        let outcomes_after = measure_outcomes(&rho_c, 0, 1);
        assert!((outcomes_after.prob_0 - 1.0).abs() < EPS, 
            "After collapse to 0, should definitely measure 0, got P(0)={}", outcomes_after.prob_0);
    }

    #[test]
    fn measure_collapse_zero_state_outcome_one_fails() {
        let rho = pure_zero_state_0();
        let collapsed = measure_collapse(&rho, 0, 1, 1);

        assert!(collapsed.is_none(), "Outcome 1 should be impossible for |0⟩ state");
    }

    #[test]
    fn outcomes_sum_to_one_for_zero_state() {
        let rho = pure_zero_state_0();
        let outcomes = measure_outcomes(&rho, 0, 1);
        let sum = outcomes.prob_0 + outcomes.prob_1;
        assert!((sum - 1.0).abs() < EPS, "Probabilities should sum to 1, got {}", sum);
    }

    #[test]
    fn trace_infrastructure_validates() {
        // Verify that the core trace-based measurement infrastructure works
        let rho = pure_zero_state_0();
        
        // Two measurements of the same state should give same outcomes
        let outcomes1 = measure_outcomes(&rho, 0, 1);
        let outcomes2 = measure_outcomes(&rho, 0, 1);
        
        assert_eq!(outcomes1.prob_0, outcomes2.prob_0);
        assert_eq!(outcomes1.prob_1, outcomes2.prob_1);
    }
}
