use crate::core::bits::Sector;
use crate::core::compute::multivector_packed::PackedMultivector;
use crate::core::compute::structured::ComputeOp;
use crate::core::ir::Multivector;
use crate::state::ideal::IdealState;

use super::monomial_state::MonomialState;

/// Runtime execution representation.
///
/// This is deliberately a compute-layer object, not a semantic state object.
#[derive(Clone, Debug, PartialEq)]
pub enum RuntimeState {
    Monomial(MonomialState),
    Generic(PackedMultivector),
}

impl RuntimeState {
    #[inline(always)]
    pub fn from_vacuum(n: u8, vacuum_bits: u64) -> Self {
        Self::Monomial(MonomialState::from_vacuum(n, vacuum_bits))
    }

    #[inline(always)]
    pub fn from_vacuum_sector(vacuum: Sector) -> Self {
        Self::from_vacuum(vacuum.n, vacuum.bits)
    }

    /// Materialized-state boundary:
    /// take a Multivector that already represents the materialized state.
    #[inline(always)]
    pub fn from_materialized_mv(mv: &Multivector) -> Self {
        Self::Generic(PackedMultivector::from_mv(mv))
    }

    /// Safe IdealState -> RuntimeState boundary for phase 1.
    ///
    /// We materialize the ideal and then enter the runtime layer.
    #[inline(always)]
    pub fn from_ideal_materialized(state: &IdealState) -> Self {
        Self::from_materialized_mv(&state.materialize())
    }

    #[inline(always)]
    pub fn is_monomial(&self) -> bool {
        matches!(self, Self::Monomial(_))
    }

    #[inline(always)]
    pub fn is_generic(&self) -> bool {
        matches!(self, Self::Generic(_))
    }

    #[inline(always)]
    pub fn active_terms(&self) -> usize {
        match self {
            Self::Monomial(state) => state.len(),
            Self::Generic(pm) => pm.terms.len(),
        }
    }

    #[inline(always)]
    pub fn to_packed(&self) -> PackedMultivector {
        match self {
            Self::Monomial(state) => state.to_packed(),
            Self::Generic(pm) => pm.clone(),
        }
    }

    #[inline(always)]
    pub fn to_mv(&self) -> Multivector {
        self.to_packed().to_mv()
    }

    /// Central runtime dispatcher.
    ///
    /// Fast path:
    /// - MonomialState × Monomial
    /// - MonomialState × Diagonal
    ///
    /// Fallback:
    /// - everything else becomes Generic(PackedMultivector)
    pub fn apply_op(&mut self, op: &ComputeOp) {
        match self {
            RuntimeState::Monomial(state) => {
                match op {
                    ComputeOp::Monomial(m) => {
                        state.apply_monomial(m);
                    }
                    ComputeOp::Diagonal(d) => {
                        state.apply_diagonal(d);
                    }
                    _ => {
                        let current = state.to_packed();
                        let out = PackedMultivector::gp(&op.to_packed(), &current);
                        *self = RuntimeState::Generic(out);
                    }
                }
            }
            RuntimeState::Generic(state) => {
                let lhs = op.to_packed();
                let out = PackedMultivector::gp(&lhs, state);
                *state = out;
            }
        }
    }

    #[inline]
    pub fn apply_ops<'a, I>(&mut self, ops: I)
    where
        I: IntoIterator<Item = &'a ComputeOp>,
    {
        for op in ops {
            self.apply_op(op);
        }
    }
}

impl From<MonomialState> for RuntimeState {
    #[inline(always)]
    fn from(value: MonomialState) -> Self {
        Self::Monomial(value)
    }
}

impl From<PackedMultivector> for RuntimeState {
    #[inline(always)]
    fn from(value: PackedMultivector) -> Self {
        Self::Generic(value)
    }
}

impl From<Multivector> for RuntimeState {
    #[inline(always)]
    fn from(value: Multivector) -> Self {
        Self::Generic(PackedMultivector::from_mv(&value))
    }
}
