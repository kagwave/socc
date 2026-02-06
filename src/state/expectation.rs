use crate::core::bits::Sector;
use crate::core::compute::multivector::gp_mv;
use crate::core::compute::multivector::scalar_part;
use crate::core::ir::Multivector;
use crate::state::density::Density;

/// Expectation value
///
///     <A> = Tr(ρ A)
pub fn expectation(rho: &Density, op: &Multivector, vacuum: Sector) -> f64 {
    let prod = gp_mv(&rho.mv, op);
    // With the current representation, the trace over the vacuum sector reduces
    // to the scalar part of the product. The vacuum sector is kept in the
    // signature for future refinements but is unused for now.
    let _ = vacuum;
    scalar_part(&prod)
}