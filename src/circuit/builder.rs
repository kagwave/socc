use crate::core::bits::Sector;
use crate::core::compute::multivector::gp_mv;
use crate::core::compute::runtime::RuntimeState;
use crate::core::compute::structured::ComputeOp;
use crate::core::ir::Multivector;

use super::gate::Gate;

/// Circuit = ordered sequence of gates + optional compiled operator.
#[derive(Clone, Debug)]
pub struct Circuit {
    pub n: u8,
    pub gates: Vec<Gate>,

    // Optional compiled operator (lazy cache)
    compiled: Option<Multivector>,
}

impl Circuit {
    /// Create empty circuit
    pub fn new(n: u8) -> Self {
        Self {
            n,
            gates: Vec::new(),
            compiled: None,
        }
    }

    /// Push a gate
    pub fn push(mut self, gate: Gate) -> Self {
        self.gates.push(gate);
        self.compiled = None; // invalidate cache
        self
    }

    ////////////////////////////////////////////////////////////
    // Ergonomic helpers
    ////////////////////////////////////////////////////////////

    pub fn x(self, q: usize) -> Self {
        self.push(Gate::X(q))
    }

    pub fn y(self, q: usize) -> Self {
        self.push(Gate::Y(q))
    }

    pub fn z(self, q: usize) -> Self {
        self.push(Gate::Z(q))
    }

    pub fn h(self, q: usize) -> Self {
        self.push(Gate::H(q))
    }

    pub fn cnot(self, c: usize, t: usize) -> Self {
        self.push(Gate::CNOT(c, t))
    }

    pub fn s(self, q: usize) -> Self {
        self.push(Gate::S(q))
    }

    pub fn t(self, q: usize) -> Self {
        self.push(Gate::T(q))
    }

    ////////////////////////////////////////////////////////////
    // Compilation
    ////////////////////////////////////////////////////////////

    /// Compile to Multivector (lazy)
    pub fn to_mv(mut self) -> Multivector {
        if let Some(op) = self.compiled {
            return op;
        }

        let mut op = Multivector::identity(self.n);

        for g in &self.gates {
            let u = g.to_mv(self.n);
            op = gp_mv(&u, &op);
        }

        self.compiled = Some(op.clone());
        op
    }

    /// Compile to structured operator sequence
    pub fn to_ops(&self) -> Vec<ComputeOp> {
        self.gates.iter().map(|g| g.to_op(self.n)).collect()
    }

    /// Execute circuit on a runtime state
    pub fn run_state(&self, state: &mut RuntimeState) {
        for op in self.to_ops() {
            state.apply_op(&op);
        }
    }

    /// Execute circuit from vacuum sector, returning final state
    pub fn run_from_vacuum(&self, vacuum: Sector) -> RuntimeState {
        let mut state = RuntimeState::from_vacuum_sector(vacuum);
        self.run_state(&mut state);
        state
    }
}