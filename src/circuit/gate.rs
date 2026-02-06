use crate::core::ir::Multivector;
use crate::core::compute::multivector_packed::PackedMultivector;

// Compute layer
use crate::core::compute::structured::ComputeOp;
use crate::core::forms::{
    monomial::MonomialPacked,
    diagonal::DiagonalPacked,
};

/// Unified gate language for circuits.
///
/// Dimension-free description.
/// Lowering happens via `to_mv(n)` or `to_op(n)`.
#[derive(Clone, Debug, PartialEq)]
pub enum Gate {
    // C1
    X(usize),
    Y(usize),
    Z(usize),

    // C2
    H(usize),
    CNOT(usize, usize),
    S(usize),

    // C3
    T(usize),

    // Escape hatch
    Custom(Multivector),
}

impl Gate {

    // ============================================================
    // Gate → Multivector (reference / fallback path)
    // ============================================================

    pub fn to_mv(&self, n: u8) -> Multivector {
        match self {

            // C1
            Gate::X(q) => crate::hierarchy::levels::c1::C1::x_gate(n, *q),
            Gate::Y(q) => crate::hierarchy::levels::c1::C1::y_gate(n, *q),
            Gate::Z(q) => crate::hierarchy::levels::c1::C1::z_gate(n, *q),

            // C2
            Gate::H(q) => crate::hierarchy::levels::c2::C2::h_gate_n(n, *q),
            Gate::CNOT(c, t) => crate::hierarchy::levels::c2::C2::cnot_gate(*c, *t),
            Gate::S(q) => crate::hierarchy::levels::c2::C2::s_gate_n(n, *q),

            // C3
            Gate::T(q) => crate::hierarchy::levels::c3::C3::t_gate(n, *q),

            // Custom
            Gate::Custom(mv) => mv.clone(),
        }
    }

    // ============================================================
    // Gate → ComputeOp (fast path)
    // ============================================================

    pub fn to_op(&self, n: u8) -> ComputeOp {
        match self {

            // ----------------------------------------------------
            // C1
            // ----------------------------------------------------

            Gate::X(q) => {
                ComputeOp::Monomial(MonomialPacked::x(n, *q))
            }

            Gate::Z(q) => {
                ComputeOp::Diagonal(DiagonalPacked::z(n, *q))
            }

            Gate::Y(_q) => {
                // Not yet implemented as structured form
                ComputeOp::Generic(PackedMultivector::from_mv(&self.to_mv(n)))
            }

            // ----------------------------------------------------
            // C2
            // ----------------------------------------------------

            Gate::CNOT(c, t) => {
                ComputeOp::Monomial(MonomialPacked::cnot(n, *c, *t))
            }

            Gate::H(_q) => {
                // No stable monomial form yet → fallback
                ComputeOp::Generic(PackedMultivector::from_mv(&self.to_mv(n)))
            }

            Gate::S(q) => {
                ComputeOp::Diagonal(DiagonalPacked::s(n, *q))
            }

            // ----------------------------------------------------
            // C3
            // ----------------------------------------------------

            Gate::T(q) => {
                ComputeOp::Diagonal(DiagonalPacked::t(n, *q))
            }

            // ----------------------------------------------------
            // fallback
            // ----------------------------------------------------

            Gate::Custom(mv) => {
                ComputeOp::Generic(PackedMultivector::from_mv(mv))
            }
        }
    }

    // ============================================================
    // Metadata helpers
    // ============================================================

    pub fn arity(&self) -> usize {
        match self {
            Gate::CNOT(_, _) => 2,
            _ => 1,
        }
    }

    pub fn qubits(&self) -> Vec<usize> {
        match self {
            Gate::X(q)
            | Gate::Y(q)
            | Gate::Z(q)
            | Gate::H(q)
            | Gate::S(q)
            | Gate::T(q) => vec![*q],

            Gate::CNOT(c, t) => vec![*c, *t],

            Gate::Custom(_) => vec![],
        }
    }
}