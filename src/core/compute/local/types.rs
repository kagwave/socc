/// One-qubit local blade kind extracted from packed `(x_i, z_i)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocalBlade {
    I,
    E1,
    E2,
    J,
}

/// One-qubit local primitive idempotent sector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocalSector {
    P,
    Q,
}

/// One-qubit local rotor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocalRotor {
    I,   // class 0
    Q1,  // pi/4
    Q2,  // pi/2 = J
    Q3,  // 3pi/4
}

/// Useful for simulator kernels where both blade and sector
/// are tracked together.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LocalState {
    pub blade: LocalBlade,
    pub sector: LocalSector,
    pub rotor: LocalRotor,
}