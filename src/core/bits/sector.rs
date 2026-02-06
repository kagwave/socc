use core::fmt;

/// Fully-specified primitive idempotent sector over n qubits.
///
/// # Peirce Decomposition Semantics
///
/// SOCC uses Peirce decomposition to index quantum state spaces by sector.
/// Each sector corresponds to a choice of Clifford projectors {P, Q} on each qubit:
///
/// - Bit i = 0 means projector P_i (maps to eigenspace λ=0)
/// - Bit i = 1 means projector Q_i (maps to eigenspace λ=1)
///
/// Over n qubits, there are 2^n possible sectors. This struct compactly
/// encodes one choice via a bitmask (bits ∈ [0, 2^n)).
///
/// # Design
///
/// - `bits`: The projector choice mask (bit i ∈ {0, 1})
/// - `n`: Number of qubits (width of the sector space)
///
/// The design is optimized for:
/// 1. **Fast bitwise operations**: XOR, AND, OR on sector boundaries
/// 2. **Cache locality**: Fixed 16-byte footprint (u64 + u8 + padding)
/// 3. **GPU readiness**: C layout with power-of-2 alignment
/// 4. **Algebraic compatibility**: Direct use in blade-sector rewrites (Π_L · B · Π_R)
///
/// # Example
///
/// ```
/// let n = 4;  // 4-qubit state
/// let s = Sector::new(0b1010, n);  // P₀, Q₁, P₂, Q₃
/// assert!(s.bit(1));  // Q₁ chosen
/// assert!(!s.bit(2)); // P₂ chosen
/// ```
///
/// # Memory Layout
///
/// ```text
/// ┌──────────────────┬──────┬──────┐
/// │      bits        │  n   │ pad  │  (16 bytes total)
/// │    (8 bytes)     │(1b)  │ (7b) │
/// └──────────────────┴──────┴──────┘
/// ```
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct Sector {
    pub bits: u64,
    pub n: u8,
}

impl fmt::Debug for Sector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Sector")
            .field("bits", &format_args!("{:#066b}", self.bits))
            .field("n", &self.n)
            .finish()
    }
}

// ============================================================================
// CONSTRUCTORS & CONVERSIONS
// ============================================================================

impl Sector {
    /// Create a sector from a bitmask and qubit count.
    ///
    /// The bitmask is automatically masked to the first n bits to prevent
    /// accidental out-of-range values. Safe even if bits >> 2^n.
    ///
    /// # Example
    /// ```
    /// let s = Sector::new(0xFFFF, 4);  // Bits masked to 0b1111
    /// assert_eq!(s.bits, 0b1111);
    /// ```
    #[inline(always)]
    pub const fn new(bits: u64, n: u8) -> Self {
        let masked = if n == 64 {
            bits
        } else {
            bits & ((1u64 << n) - 1)
        };
        Self { bits: masked, n }
    }

    /// The all-P sector (zero bitmask) on n qubits.
    ///
    /// # Example
    /// ```
    /// let s = Sector::zero(8);
    /// assert_eq!(s.bits, 0);
    /// ```
    #[inline(always)]
    pub const fn zero(n: u8) -> Self {
        Self { bits: 0, n }
    }

    /// Alias for `new()` for clarity when constructing from raw bits.
    #[inline(always)]
    pub const fn from_bits(bits: u64, n: u8) -> Self {
        Self::new(bits, n)
    }
}

// ============================================================================
// BIT-LEVEL ACCESS
// ============================================================================

impl Sector {
    /// Read the projector choice at qubit position i.
    ///
    /// Returns true for Q_i, false for P_i.
    ///
    /// # Panics
    /// Debug builds assert i < n. Release builds silently wrap.
    #[inline(always)]
    pub fn bit(self, i: u32) -> bool {
        debug_assert!(i < self.n as u32);
        ((self.bits >> i) & 1) != 0
    }

    /// Set the projector at qubit i and return a new sector.
    ///
    /// # Panics
    /// Debug builds assert i < n.
    #[inline(always)]
    pub fn set_bit(self, i: u32, value: bool) -> Self {
        debug_assert!(i < self.n as u32);
        let mask = 1u64 << i;
        let bits = if value {
            self.bits | mask
        } else {
            self.bits & !mask
        };
        Self { bits, n: self.n }
    }

    /// Flip the projector at qubit i (P ↔ Q) and return a new sector.
    ///
    /// # Panics
    /// Debug builds assert i < n.
    #[inline(always)]
    pub fn flip_bit(self, i: u32) -> Self {
        debug_assert!(i < self.n as u32);
        Self {
            bits: self.bits ^ (1u64 << i),
            n: self.n,
        }
    }
}

// ============================================================================
// BULK OPERATIONS
// ============================================================================

impl Sector {
    /// XOR a bitmask into this sector (componentwise flip).
    ///
    /// Automatically masks the result to the first n bits.
    ///
    /// # Use Case
    /// Applying a diagonal clifford operation that flips certain projectors.
    ///
    /// # Example
    /// ```
    /// let s = Sector::new(0b0010, 4);
    /// let flipped = s.xor_mask(0b1010);  // Flip qubits 1 and 3
    /// assert_eq!(flipped.bits, 0b1000);
    /// ```
    #[inline(always)]
    pub fn xor_mask(self, mask: u64) -> Self {
        let keep = if self.n == 64 {
            u64::MAX
        } else {
            (1u64 << self.n) - 1
        };
        Self {
            bits: (self.bits ^ mask) & keep,
            n: self.n,
        }
    }

    /// Bitwise equality check (both bits and n must match).
    ///
    /// Equivalent to `self == rhs` but more explicit for clarity.
    #[inline(always)]
    pub fn equals(self, rhs: Self) -> bool {
        self.n == rhs.n && self.bits == rhs.bits
    }
}

// ============================================================================
// CONTROL MASKS (PARTIAL SECTOR PREDICATES)
// ============================================================================

/// Partial sector predicate: matches sectors meeting a specific projector pattern.
///
/// Rather than specifying all n projectors, a control mask specifies:
/// - `mask`: Which qubit positions to check
/// - `bits`: What projector values are required at those positions
///
/// A sector matches if `(sector.bits & mask) == (bits & mask)`.
///
/// # Use Case
///
/// In quantum circuit optimization, we often want to prune terms by:
/// - "All terms where qubit 0 is in P" (matches ControlMask::on_zero(0))
/// - "All terms where qubits 1,3 both have Q" (matches ControlMask::on_bits(0b1010, 0b1010))
///
/// This avoids redundant Peirce block computations when sector structure guarantees
/// zero contributions.
///
/// # Example
///
/// ```
/// let s = Sector::from_bits(0b1101, 4);  // P₀, Q₁, P₂, Q₃
/// assert!(ControlMask::on_one(1).matches(s));   // Qubit 1 is Q ✓
/// assert!(!ControlMask::on_zero(1).matches(s)); // Qubit 1 is not P ✗
/// ```
///
/// # Memory Layout
///
/// ```text
/// ┌──────────────┬──────────────┐
/// │     mask     │     bits     │  (16 bytes total)
/// │  (8 bytes)   │  (8 bytes)   │
/// └──────────────┴──────────────┘
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct ControlMask {
    pub mask: u64,
    pub bits: u64,
}

impl ControlMask {
    /// Create a control mask from explicit mask and bits.
    #[inline(always)]
    pub const fn new(mask: u64, bits: u64) -> Self {
        Self { mask, bits }
    }

    /// Check if a sector matches this control mask.
    ///
    /// Returns true iff `(sector.bits & mask) == (bits & mask)`.
    #[inline(always)]
    pub fn matches(self, sector: Sector) -> bool {
        (sector.bits & self.mask) == (self.bits & self.mask)
    }

    // ========== Common Patterns ==========

    /// Predicate: qubit i must be P (bit = 0).
    #[inline(always)]
    pub const fn on_zero(i: u32) -> Self {
        Self {
            mask: 1u64 << i,
            bits: 0,
        }
    }

    /// Predicate: qubit i must be Q (bit = 1).
    #[inline(always)]
    pub const fn on_one(i: u32) -> Self {
        Self {
            mask: 1u64 << i,
            bits: 1u64 << i,
        }
    }

    /// Predicate: qubits in mask must have specific bits.
    ///
    /// Alias for `new()` for semantic clarity.
    #[inline(always)]
    pub const fn on_bits(mask: u64, bits: u64) -> Self {
        Self { mask, bits }
    }
}

#[cfg(test)]
mod tests {
    use super::{ControlMask, Sector};

    // ========== Sector Tests ==========

    #[test]
    fn sector_bit_access_and_modification() {
        let s = Sector::zero(4)
            .set_bit(1, true)
            .set_bit(3, true);

        assert!(!s.bit(0), "Bit 0 should be unset (P)");
        assert!(s.bit(1), "Bit 1 should be set (Q)");
        assert!(!s.bit(2), "Bit 2 should be unset (P)");
        assert!(s.bit(3), "Bit 3 should be set (Q)");
    }

    #[test]
    fn sector_flip_bit_toggles() {
        let s = Sector::zero(3).flip_bit(2);
        assert_eq!(s.bits, 0b100, "Flipping bit 2 should set bit 2");

        let s2 = s.flip_bit(2);
        assert_eq!(s2.bits, 0b000, "Flipping again should unset");
    }

    #[test]
    fn sector_xor_mask_applies_bulk_flip() {
        let s = Sector::from_bits(0b010, 3);
        let flipped = s.xor_mask(0b101);  // Flip qubits 0 and 2
        assert_eq!(flipped.bits, 0b111, "XOR should flip selected qubits");
    }

    #[test]
    fn sector_new_masks_out_of_range_bits() {
        let s = Sector::new(0xFFFF, 4);
        assert_eq!(s.bits, 0b1111, "Bits beyond n should be masked");
    }

    #[test]
    fn sector_equals_checks_both_bits_and_n() {
        let s1 = Sector::new(0b101, 3);
        let s2 = Sector::new(0b101, 3);
        let s3 = Sector::new(0b101, 4);  // Different n

        assert!(s1.equals(s2), "Identical sectors should be equal");
        assert!(!s1.equals(s3), "Different n should not be equal");
    }

    // ========== ControlMask Tests ==========

    #[test]
    fn control_mask_single_qubit_predicates() {
        let s = Sector::from_bits(0b101, 3);

        assert!(ControlMask::on_one(0).matches(s), "Qubit 0 is Q");
        assert!(!ControlMask::on_zero(0).matches(s), "Qubit 0 is not P");
        assert!(!ControlMask::on_one(1).matches(s), "Qubit 1 is not Q");
        assert!(ControlMask::on_zero(1).matches(s), "Qubit 1 is P");
        assert!(ControlMask::on_one(2).matches(s), "Qubit 2 is Q");
    }

    #[test]
    fn control_mask_multi_qubit_pattern() {
        let s = Sector::from_bits(0b101101, 6);
        let pattern = ControlMask::on_bits(0b001101, 0b001101);
        assert!(pattern.matches(s), "Sector should match multi-qubit pattern");
    }

    #[test]
    fn control_mask_all_zeros_pattern() {
        let s = Sector::from_bits(0b0, 4);
        let all_p = ControlMask::on_bits(0xF, 0x0);
        assert!(all_p.matches(s), "All-P sector should match");
    }

    #[test]
    fn control_mask_all_ones_pattern() {
        let s = Sector::from_bits(0xF, 4);
        let all_q = ControlMask::on_bits(0xF, 0xF);
        assert!(all_q.matches(s), "All-Q sector should match");
    }
}