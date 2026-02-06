use crate::core::bits::Sector;
use crate::core::compute::multivector::scalar_part;
use crate::core::compute::state::{apply_mv_to_left_sector, apply_mv_to_right_sector};
use crate::core::ir::Multivector;

/// Spinor trace relative to the vacuum sector.
///
///     Tr(A) = scalar_part(P_N A P_N)
pub fn trace_spinor(mv: &Multivector, vacuum: Sector) -> f64 {
    let right = apply_mv_to_right_sector(mv, vacuum);
    let block = apply_mv_to_left_sector(vacuum, &right);

    scalar_part(&block)
}