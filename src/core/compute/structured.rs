use crate::core::forms::{
    controlled::ControlledPacked,
    diagonal::DiagonalPacked,
    monomial::MonomialPacked,
};
use crate::core::compute::multivector_packed::PackedMultivector;
use crate::core::ir::Multivector;

#[derive(Clone, Debug, PartialEq)]
pub enum ComputeOp {
    Controlled(ControlledPacked),
    Diagonal(DiagonalPacked),
    Monomial(MonomialPacked),
    Generic(PackedMultivector),
}

impl ComputeOp {
    #[inline(always)]
    pub fn n(&self) -> u8 {
        match self {
            Self::Controlled(op) => op.n,
            Self::Diagonal(op) => op.n,
            Self::Monomial(op) => op.n,
            Self::Generic(op) => op.n,
        }
    }

    #[inline(always)]
    pub fn identity(n: u8) -> Self {
        Self::Monomial(MonomialPacked::identity(n))
    }

    #[inline(always)]
    pub fn is_generic(&self) -> bool {
        matches!(self, Self::Generic(_))
    }

    /// Lower IR → structured compute form
    #[inline]
    pub fn lower_mv(mv: &Multivector) -> Self {
        if let Some(ctrl) = ControlledPacked::try_from_mv(mv) {
            Self::Controlled(ctrl)
        } else if let Some(diag) = DiagonalPacked::try_from_mv(mv) {
            Self::Diagonal(diag)
        } else if let Some(mono) = MonomialPacked::try_from_mv(mv) {
            Self::Monomial(mono)
        } else {
            Self::Generic(PackedMultivector::from_mv(mv))
        }
    }

    /// Lift compute → IR
    #[inline]
    pub fn to_mv(&self) -> Multivector {
        match self {
            Self::Controlled(op) => op.to_mv(),
            Self::Diagonal(op) => op.to_mv(),
            Self::Monomial(op) => op.to_mv(),
            Self::Generic(op) => op.to_mv(),
        }
    }

    /// Convert any form → generic packed
    #[inline]
    pub fn to_packed(&self) -> PackedMultivector {
        match self {
            Self::Controlled(op) => PackedMultivector::from_mv(&op.to_mv()),
            Self::Diagonal(op) => PackedMultivector::from_mv(&op.to_mv()),
            Self::Monomial(op) => PackedMultivector::from_mv(&op.to_mv()),
            Self::Generic(op) => op.clone(),
        }
    }

    /// Scale
    #[inline]
    pub fn scale(self, scalar: f64) -> Self {
        match self {
            Self::Controlled(mut op) => {
                op.off_coeff *= scalar;
                op.on_coeff *= scalar;
                Self::Controlled(op)
            }
            Self::Diagonal(op) => Self::Diagonal(op.scale(scalar)),
            Self::Monomial(mut op) => {
                for c in &mut op.coeffs {
                    *c *= scalar;
                }
                Self::Monomial(op)
            }
            Self::Generic(op) => {
                Self::Generic(PackedMultivector::scale(&op, scalar))
            }
        }
    }
}

//////////////////////////////////////////////////////////////
// Core compute dispatcher
//////////////////////////////////////////////////////////////

#[inline]
pub fn gp_compute(a: &ComputeOp, b: &ComputeOp) -> ComputeOp {
    match (a, b) {
        ////////////////////////////////////////////////////////
        // Controlled × Controlled
        ////////////////////////////////////////////////////////
        (ComputeOp::Controlled(ac), ComputeOp::Controlled(bc)) => {
            if let Some(out) = ac.gp(*bc) {
                ComputeOp::Controlled(out)
            } else {
                fallback(a, b)
            }
        }

        ////////////////////////////////////////////////////////
        // Diagonal × Diagonal
        ////////////////////////////////////////////////////////
        (ComputeOp::Diagonal(ad), ComputeOp::Diagonal(bd)) => {
            if let Some(out) = ad.gp(bd) {
                ComputeOp::Diagonal(out)
            } else {
                fallback(a, b)
            }
        }

        ////////////////////////////////////////////////////////
        // Monomial × Monomial
        ////////////////////////////////////////////////////////
        (ComputeOp::Monomial(am), ComputeOp::Monomial(bm)) => {
            if let Some(out) = am.gp(bm) {
                ComputeOp::Monomial(out)
            } else {
                fallback(a, b)
            }
        }

        ////////////////////////////////////////////////////////
        // Diagonal × Controlled
        ////////////////////////////////////////////////////////
        (ComputeOp::Diagonal(d), ComputeOp::Controlled(c)) => {
            ComputeOp::Controlled(diagonal_left_controlled(d, c))
        }

        ////////////////////////////////////////////////////////
        // Controlled × Diagonal
        ////////////////////////////////////////////////////////
        (ComputeOp::Controlled(c), ComputeOp::Diagonal(d)) => {
            ComputeOp::Controlled(diagonal_right_controlled(c, d))
        }

        ////////////////////////////////////////////////////////
        // Diagonal × Monomial
        ////////////////////////////////////////////////////////

        (ComputeOp::Diagonal(d), ComputeOp::Monomial(m)) => {
            ComputeOp::Monomial(diagonal_left_monomial(d, m))
        }

        ////////////////////////////////////////////////////////
        // Monomial × Diagonal
        ////////////////////////////////////////////////////////

        (ComputeOp::Monomial(m), ComputeOp::Diagonal(d)) => {
            ComputeOp::Monomial(diagonal_right_monomial(m, d))
        }

        ////////////////////////////////////////////////////////
        // Everything else → fallback
        ////////////////////////////////////////////////////////
        _ => fallback(a, b),
    }
}

//////////////////////////////////////////////////////////////
// Addition
//////////////////////////////////////////////////////////////

#[inline]
pub fn add_compute(a: &ComputeOp, b: &ComputeOp) -> ComputeOp {
    match (a, b) {
        ////////////////////////////////////////////////////////
        // Controlled + Controlled
        ////////////////////////////////////////////////////////
        (ComputeOp::Controlled(ac), ComputeOp::Controlled(bc))
            if ac.same_split(bc) && ac.k == bc.k =>
        {
            ComputeOp::Controlled(ControlledPacked {
                off_bits: ac.off_bits,
                on_bits: ac.on_bits,
                off_coeff: ac.off_coeff + bc.off_coeff,
                on_coeff: ac.on_coeff + bc.on_coeff,
                k: ac.k,
                n: ac.n,
            })
        }

        ////////////////////////////////////////////////////////
        // Diagonal + Diagonal
        ////////////////////////////////////////////////////////
        (ComputeOp::Diagonal(ad), ComputeOp::Diagonal(bd)) => {
            if let Some(out) = ad.add(bd) {
                ComputeOp::Diagonal(out)
            } else {
                fallback_add(a, b)
            }
        }

        ////////////////////////////////////////////////////////
        // Monomial + Monomial (rare merge case)
        ////////////////////////////////////////////////////////
        (ComputeOp::Monomial(am), ComputeOp::Monomial(bm))
            if am.perm == bm.perm && am.payload == bm.payload =>
        {
            let mut out = am.clone();
            for i in 0..out.coeffs.len() {
                out.coeffs[i] += bm.coeffs[i];
            }
            ComputeOp::Monomial(out)
        }

        ////////////////////////////////////////////////////////
        // fallback
        ////////////////////////////////////////////////////////
        _ => fallback_add(a, b),
    }
}

#[inline]
pub fn sub_compute(a: &ComputeOp, b: &ComputeOp) -> ComputeOp {
    add_compute(a, &scale_compute(b, -1.0))
}

#[inline]
pub fn scale_compute(op: &ComputeOp, scalar: f64) -> ComputeOp {
    op.clone().scale(scalar)
}

//////////////////////////////////////////////////////////////
// Fast mixed rules
//////////////////////////////////////////////////////////////

#[inline]
fn diagonal_left_controlled(
    d: &DiagonalPacked,
    c: &ControlledPacked,
) -> ControlledPacked {
    ControlledPacked {
        off_bits: c.off_bits,
        on_bits: c.on_bits,
        off_coeff: c.off_coeff * d.coeff_of(c.off_bits),
        on_coeff: c.on_coeff * d.coeff_of(c.on_bits),
        k: c.k,
        n: c.n,
    }
}

#[inline]
fn diagonal_right_controlled(
    c: &ControlledPacked,
    d: &DiagonalPacked,
) -> ControlledPacked {
    ControlledPacked {
        off_bits: c.off_bits,
        on_bits: c.on_bits,
        off_coeff: c.off_coeff * d.coeff_of(c.off_bits),
        on_coeff: c.on_coeff * d.coeff_of(c.on_bits),
        k: c.k,
        n: c.n,
    }
}

//////////////////////////////////////////////////////////////
// Diagonal × Monomial
//////////////////////////////////////////////////////////////

#[inline]
fn diagonal_left_monomial(
    d: &DiagonalPacked,
    m: &MonomialPacked,
) -> MonomialPacked {
    let mut out = m.clone();

    for x in 0..out.coeffs.len() {
        let y = out.perm[x];
        out.coeffs[x] *= d.coeff_of(y);
    }

    out
}

//////////////////////////////////////////////////////////////
// Monomial × Diagonal
//////////////////////////////////////////////////////////////

#[inline]
fn diagonal_right_monomial(
    m: &MonomialPacked,
    d: &DiagonalPacked,
) -> MonomialPacked {
    let mut out = m.clone();

    for x in 0..out.coeffs.len() {
        out.coeffs[x] *= d.coeff_of(x as u64);
    }

    out
}

//////////////////////////////////////////////////////////////
// Fallback helpers
//////////////////////////////////////////////////////////////

#[inline]
fn fallback(a: &ComputeOp, b: &ComputeOp) -> ComputeOp {
    ComputeOp::Generic(PackedMultivector::gp(
        &a.to_packed(),
        &b.to_packed(),
    ))
}

#[inline]
fn fallback_add(a: &ComputeOp, b: &ComputeOp) -> ComputeOp {
    ComputeOp::Generic(PackedMultivector::add(
        &a.to_packed(),
        &b.to_packed(),
    ))
}

//////////////////////////////////////////////////////////////
// Public API
//////////////////////////////////////////////////////////////

#[inline]
pub fn gp_mv_structured(a: &Multivector, b: &Multivector) -> Multivector {
    let la = ComputeOp::lower_mv(a);
    let lb = ComputeOp::lower_mv(b);
    gp_compute(&la, &lb).to_mv()
}

#[inline]
pub fn add_mv_structured(a: &Multivector, b: &Multivector) -> Multivector {
    let la = ComputeOp::lower_mv(a);
    let lb = ComputeOp::lower_mv(b);
    add_compute(&la, &lb).to_mv()
}

#[inline]
pub fn sub_mv_structured(a: &Multivector, b: &Multivector) -> Multivector {
    let la = ComputeOp::lower_mv(a);
    let lb = ComputeOp::lower_mv(b);
    sub_compute(&la, &lb).to_mv()
}

#[inline]
pub fn scale_mv_structured(a: &Multivector, scalar: f64) -> Multivector {
    let la = ComputeOp::lower_mv(a);
    scale_compute(&la, scalar).to_mv()
}