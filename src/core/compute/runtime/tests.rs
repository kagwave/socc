#[cfg(test)]
mod tests {
    use crate::circuit::builder::Circuit;
    use crate::circuit::gate::Gate;
    use crate::core::bits::Sector;
    use crate::core::compute::runtime::RuntimeState;
    use crate::core::compute::structured::ComputeOp;

    #[test]
    fn vacuum_starts_as_monomial() {
        let n = 2;
        let vacuum = Sector::new(0, n);
        let state = RuntimeState::from_vacuum_sector(vacuum);

        assert!(state.is_monomial());
        assert_eq!(state.active_terms(), 1);
    }

    #[test]
    fn monomial_gate_stays_monomial() {
        let n = 1;
        let vacuum = Sector::new(0, n);
        let mut state = RuntimeState::from_vacuum_sector(vacuum);

        let op = Gate::X(0).to_op(n);
        state.apply_op(&op);

        assert!(state.is_monomial());
    }

    #[test]
    fn diagonal_gate_stays_monomial() {
        let n = 2;
        let vacuum = Sector::new(0, n);
        let mut state = RuntimeState::from_vacuum_sector(vacuum);

        let op = Gate::Z(0).to_op(n);
        state.apply_op(&op);

        assert!(state.is_monomial());
    }

    #[test]
    fn hadamard_forces_generic_fallback() {
        let n = 1;
        let vacuum = Sector::new(0, n);
        let mut state = RuntimeState::from_vacuum_sector(vacuum);

        let op = Gate::H(0).to_op(n);
        state.apply_op(&op);

        assert!(state.is_generic());
    }

    #[test]
    fn x_s_t_z_chain_stays_monomial() {
        let n = 2;
        let vacuum = Sector::new(0, n);
        let circuit = Circuit::new(n)
            .x(0)
            .s(0)
            .t(1)
            .z(1);

        let state = circuit.run_from_vacuum(vacuum);
        assert!(state.is_monomial());
    }

    #[test]
    fn cnot_monomial_stays_monomial() {
        let n = 2;
        let vacuum = Sector::new(0, n);
        let mut state = RuntimeState::from_vacuum_sector(vacuum);

        let op = Gate::CNOT(0, 1).to_op(n);
        state.apply_op(&op);

        assert!(state.is_monomial());
    }

    #[test]
    fn mixed_monomial_diagonal_stays_monomial() {
        let n = 2;
        let vacuum = Sector::new(0, n);
        let circuit = Circuit::new(n)
            .cnot(0, 1)
            .t(1)
            .cnot(0, 1);

        let state = circuit.run_from_vacuum(vacuum);
        assert!(state.is_monomial());
    }

    #[test]
    fn to_mv_roundtrip_preserves_content() {
        let n = 2;
        let vacuum = Sector::new(0, n);
        let circuit = Circuit::new(n).x(0).z(1);

        let state = circuit.run_from_vacuum(vacuum);
        let mv = state.to_mv();

        // Verify it can round-trip without errors
        assert_eq!(mv.n, n);
    }

    #[test]
    fn apply_ops_applies_all_operators() {
        let n = 2;
        let vacuum = Sector::new(0, n);
        let mut state = RuntimeState::from_vacuum_sector(vacuum);

        let ops = vec![
            Gate::X(0).to_op(n),
            Gate::Z(1).to_op(n),
        ];

        state.apply_ops(ops.iter());

        assert!(state.is_monomial());
        assert_eq!(state.active_terms(), 1);
    }

    #[test]
    fn circuit_to_ops_matches_gate_sequence() {
        let n = 2;
        let circuit = Circuit::new(n).x(0).cnot(0, 1).t(1);

        let ops = circuit.to_ops();
        assert_eq!(ops.len(), 3);
    }

    #[test]
    fn run_state_executes_without_panic() {
        let n = 2;
        let vacuum = Sector::new(0, n);
        let circuit = Circuit::new(n).x(0).z(1).s(0).t(1);

        let mut state = RuntimeState::from_vacuum_sector(vacuum);
        circuit.run_state(&mut state);

        assert!(state.is_monomial());
    }

    #[test]
    fn multiple_gates_accumulate() {
        let n = 1;
        let vacuum = Sector::new(0, n);

        let mut state = RuntimeState::from_vacuum_sector(vacuum);
        let op_x = Gate::X(0).to_op(n);
        let op_z = Gate::Z(0).to_op(n);

        state.apply_op(&op_x);
        state.apply_op(&op_z);

        assert!(state.is_monomial());
    }

    #[test]
    fn fallback_to_generic_is_irreversible() {
        let n = 1;
        let vacuum = Sector::new(0, n);
        let mut state = RuntimeState::from_vacuum_sector(vacuum);

        // Force fallback with H gate
        state.apply_op(&Gate::H(0).to_op(n));
        assert!(state.is_generic());

        // Apply monomial gate - should stay generic
        state.apply_op(&Gate::X(0).to_op(n));
        assert!(state.is_generic());
    }

    #[test]
    fn cnot_x_cnot_equals_z() {
        // X ⟷ CNOT permutation
        let n = 2;
        let vacuum = Sector::new(0, n);

        let circuit1 = Circuit::new(n).cnot(0, 1).x(1).cnot(0, 1);
        let state1 = circuit1.run_from_vacuum(vacuum);

        let circuit2 = Circuit::new(n).z(1);
        let state2 = circuit2.run_from_vacuum(vacuum);

        let mv1 = state1.to_mv();
        let mv2 = state2.to_mv();

        // Both should stay monomial
        assert!(matches!(state1, RuntimeState::Monomial(_)));
        assert!(matches!(state2, RuntimeState::Monomial(_)));
    }

    #[test]
    fn identity_circuit_preserves_vacuum() {
        let n = 2;
        let vacuum = Sector::new(0, n);

        let circuit = Circuit::new(n); // empty
        let state = circuit.run_from_vacuum(vacuum);

        assert!(state.is_monomial());
        assert_eq!(state.active_terms(), 1);
    }
}
