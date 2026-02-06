use crate::core::bits::Sector;
use crate::core::compute::multivector::{gp_mv, simplify_mv};
use crate::core::compute::state::apply_to_vacuum;
use crate::core::ir::Multivector;

#[derive(Clone, Debug, PartialEq)]
pub struct IdealState {
    /// Representative operator A in the quotient class [A].
    pub rep: Multivector,

    /// Primitive idempotent / vacuum sector P_n.
    pub vacuum: Sector,

    /// Optional: active sectors reachable from P_n under rep.
    /// None = not tracking (default).
    pub active_sectors: Option<Vec<u64>>,
}

impl IdealState {
    #[inline(always)]
    pub fn new(rep: Multivector, vacuum: Sector) -> Self {
        Self {
            rep,
            vacuum,
            active_sectors: None,
        }
    }

    /// Enable sector tracking (starts at vacuum sector).
    pub fn with_tracking(mut self) -> Self {
        self.active_sectors = Some(vec![self.vacuum.bits]);
        self
    }

    /// Materialize the actual ideal element A P_n.
    #[inline(always)]
    pub fn materialize(&self) -> Multivector {
        apply_to_vacuum(&self.rep, self.vacuum)
    }

    /// Current canonical representative of the state class.
    #[inline(always)]
    pub fn lift(&self) -> Multivector {
        simplify_mv(self.rep.clone())
    }

    /// Plain algebraic evolution (always correct).
    #[inline(always)]
    pub fn left_apply(&self, u: &Multivector) -> Self {
        Self {
            rep: gp_mv(u, &self.rep),
            vacuum: self.vacuum,
            active_sectors: None,
        }
    }

    /// Structured evolution (fast path when possible).
    pub fn apply(&self, u: &Multivector) -> Self {
        crate::state::evolve::evolve(self, u)
    }

    /// Physical equality in the ideal.
    #[inline(always)]
    pub fn physically_eq(&self, other: &Self) -> bool {
        self.vacuum == other.vacuum && self.materialize() == other.materialize()
    }
}