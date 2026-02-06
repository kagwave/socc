use crate::core::compute::involutions::reverse_mv;
use crate::core::compute::multivector::gp_mv;
use crate::core::ir::Multivector;
use crate::state::ideal::IdealState;

#[derive(Clone, Debug, PartialEq)]
pub struct Density {
    pub mv: Multivector,
}

impl Density {
    #[inline(always)]
    pub fn new(mv: Multivector) -> Self {
        Self { mv }
    }
}

impl IdealState {
    /// Construct the density operator
    ///
    ///     ρ = ψ ψ~
    ///
    /// where ψ = A P_n.
    pub fn density(&self) -> Density {
        let psi = self.materialize();
        let psi_rev = reverse_mv(&psi);
        let rho = gp_mv(&psi, &psi_rev);

        Density::new(rho)
    }
}