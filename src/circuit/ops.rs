use super::builder::Circuit;

// Pull from your hierarchy modules
use crate::core::hierarchy::{c1, c2, c3};

impl Circuit {
    ////////////////////////////////////////////////////////////
    // C1 (if you have it)
    ////////////////////////////////////////////////////////////

    pub fn id(self) -> Self {
        self.apply(c1::id(self.n))
    }

    ////////////////////////////////////////////////////////////
    // C2 (Clifford)
    ////////////////////////////////////////////////////////////

    pub fn x(self, q: u8) -> Self {
        self.apply(c2::x(self.n, q))
    }

    pub fn z(self, q: u8) -> Self {
        self.apply(c2::z(self.n, q))
    }

    pub fn h(self, q: u8) -> Self {
        self.apply(c2::h(self.n, q))
    }

    pub fn cnot(self, c: u8, t: u8) -> Self {
        self.apply(c2::cnot(self.n, c, t))
    }

    ////////////////////////////////////////////////////////////
    // C3 (your semi-Clifford layer)
    ////////////////////////////////////////////////////////////

    pub fn s(self, q: u8) -> Self {
        self.apply(c3::s(self.n, q))
    }

    pub fn t(self, q: u8) -> Self {
        self.apply(c3::t(self.n, q))
    }

    ////////////////////////////////////////////////////////////
    // Generic hook (important)
    ////////////////////////////////////////////////////////////

    /// Apply any Multivector directly.
    pub fn op(self, mv: crate::core::ir::Multivector) -> Self {
        self.apply(mv)
    }
}