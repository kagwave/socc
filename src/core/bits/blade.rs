use core::fmt;

/// A packed basis blade in Cl_{2,0}^{⊗ n}.
///
/// Per qubit i:
/// - x = 0, z = 0 => 1
/// - x = 0, z = 1 => e1  (z)
/// - x = 1, z = 0 => e2  (x) 
/// - x = 1, z = 1 => e1 e2  (j)
///
/// sign ±1
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct Blade {
    pub z: u64,
    pub x: u64,
    pub sign: bool,
}

impl fmt::Debug for Blade {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = if self.sign { "-" } else { "+" };
        f.debug_struct("Blade")
            .field("sign", &s)
            .field("z", &format_args!("{:#066b}", self.z))
            .field("x", &format_args!("{:#066b}", self.x))
            .finish()
    }
}


impl Blade {
    /// Construct a blade with the specified z/x bits and sign.
    #[inline(always)]
    pub const fn new(x: u64, z: u64, sign: bool) -> Self {
        Self { z, x, sign }
    }

    /// Construct the identity blade (all qubits have (z,x) = (0,0)).
    #[inline(always)]
    pub const fn identity() -> Self {
        Self {
            z: 0,
            x: 0,
            sign: false,
        }
    }

    /// Construct a blade with a single Z on qubit i.
    #[inline(always)]
    pub const fn e1(i: usize) -> Self {
        Self {
            z: 1u64 << i,
            x: 0,
            sign: false,
        }
    }

    /// Construct a blade with a single X on qubit i.
    #[inline(always)]
    pub const fn e2(i: usize) -> Self {
        Self {
            z: 0,
            x: 1u64 << i,
            sign: false,
        }
    }

    /// Construct a blade with a single J = ZX on qubit i.
    #[inline(always)]
    pub const fn j(i: usize) -> Self {
        Self {
            z: 1u64 << i,
            x: 1u64 << i,
            sign: false,
        }
    }

    /// Alias for e2 (x).
    #[inline(always)]
    pub const fn x(i: usize) -> Self {
        Self::e2(i)
    }

    /// Alias for e1 (z).
    #[inline(always)]
    pub const fn z(i: usize) -> Self {
        Self::e1(i)
    }

    /// Alias for j (j).
    #[inline(always)]
    pub const fn y(i: usize) -> Self {
        Self::j(i)
    }

    /// Construct a blade with the same z/x bits but the specified sign.
    #[inline(always)]
    pub const fn with_sign(self, sign: bool) -> Self {
        Self {
            z: self.z,
            x: self.x,
            sign,
        }
    }

    /// Construct a blade with the same z/x bits but flipped sign.
    #[inline(always)]
    pub const fn negate(self) -> Self {
        Self {
            z: self.z,
            x: self.x,
            sign: !self.sign,
        }
    }

    /// Construct a blade with the same z/x bits but positive sign.
    #[inline(always)]
    pub const fn unsigned(self) -> Self {
        Self {
            z: self.z,
            x: self.x,
            sign: false,
        }
    }

    /// Return true if this blade is the identity (up to sign).
    #[inline(always)]
    pub const fn is_identity(self) -> bool {
        self.z == 0 && self.x == 0
    }
}