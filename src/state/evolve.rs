use crate::core::forms::{
    diagonal::DiagonalPacked,
    monomial::MonomialPacked,
};
use crate::core::compute::multivector::gp_mv;
use crate::core::compute::structured::ComputeOp;
use crate::core::ir::Multivector;
use crate::state::ideal::IdealState;

#[inline]
pub fn evolve(state: &IdealState, u: &Multivector) -> IdealState {
    // Try structured fast paths
    match ComputeOp::lower_mv(u) {
        ComputeOp::Monomial(m) => evolve_monomial(state, &m),
        ComputeOp::Diagonal(d) => evolve_diagonal(state, &d),
        // Could add Controlled here similarly
        _ => {
            // Fallback: exact algebra
            IdealState {
                rep: gp_mv(u, &state.rep),
                vacuum: state.vacuum,
                active_sectors: None,
            }
        }
    }
}

/// Monomial fast path:
/// - permutes sectors
/// - applies per-sector payload
fn evolve_monomial(state: &IdealState, m: &MonomialPacked) -> IdealState {
    let n = m.n;

    // If we have active sectors, update them sparsely
    let active = state.active_sectors.clone().unwrap_or_else(|| {
        vec![state.vacuum.bits] // start from vacuum sector
    });

    let mut new_active = Vec::with_capacity(active.len());

    for &x in &active {
        let y = m.perm[x as usize];
        new_active.push(y);
    }

    // Apply operator to representative (still exact)
    let new_rep = gp_mv(&m.to_mv(), &state.rep);

    IdealState {
        rep: new_rep,
        vacuum: state.vacuum,
        active_sectors: Some(dedup(new_active)),
    }
}

/// Diagonal fast path:
/// - no sector movement
/// - optionally scale / prune sectors
fn evolve_diagonal(state: &IdealState, d: &DiagonalPacked) -> IdealState {
    let active = state.active_sectors.clone().unwrap_or_else(|| {
        vec![state.vacuum.bits]
    });

    // Optionally prune zero-weight sectors
    let mut new_active = Vec::with_capacity(active.len());
    for &x in &active {
        if d.coeff_of(x) != 0.0 {
            new_active.push(x);
        }
    }

    let new_rep = gp_mv(&d.to_mv(), &state.rep);

    IdealState {
        rep: new_rep,
        vacuum: state.vacuum,
        active_sectors: Some(dedup(new_active)),
    }
}

#[inline]
fn dedup(mut v: Vec<u64>) -> Vec<u64> {
    v.sort_unstable();
    v.dedup();
    v
}