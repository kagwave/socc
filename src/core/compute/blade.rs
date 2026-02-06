use crate::core::bits::Blade;
use core::ops::Mul;

/// # Blade Algebra Operations
///
/// This module implements core clifford algebra operations on packed blades.
/// A blade represents a clifford basis element as (x, z, sign) where:
/// - x ∧ z = bivector parts (local squares e₁∧e₂)
/// - x ⊕ z = vector parts (e₁, e₂, or ε mixed)
/// - sign: overall ±1 coefficient

// ============================================================================
// MULTIPLICATION
// ============================================================================

/// Geometric product of two blades using grassmann algebra rules.
/// The x-support and z-support of each blade anticommute according to:
/// ```
/// e₁² = 1    e₂² = 1    e₁e₂ = -e₂e₁ (sign flip on crossing)
/// ```
///
/// Algorithm:
/// 1. XOR x and z bits separately (grassmann ideal)
/// 2. Count crossings where left's x-bits meet right's z-bits (sign contribution)
/// 3. Combine signs from both operands and crossing parity
///
/// # Example
/// ```
/// let e1 = Blade::z(0);  // e₁ at qubit 0
/// let e2 = Blade::x(0);  // e₂ at qubit 0
/// let j = gp_blade(e1, e2);  // e₁∧e₂ bivector, sign=-1
/// assert!(j.sign);  // negative
/// ```
#[inline(always)]
pub fn gp_blade(a: Blade, b: Blade) -> Blade {
    // Count sign flips from x-z crossings: left x meeting right z
    let crossing_parity = ((a.x & b.z).count_ones() & 1) != 0;

    Blade {
        x: a.x ^ b.x,
        z: a.z ^ b.z,
        sign: a.sign ^ b.sign ^ crossing_parity,
    }
}

/// Bitwise test: do two blades anticommute?
///
/// Two blades anticommute if their basis elements have an odd number of
/// "bad crossings" (x-support of one meeting z-support of other and vice versa).
/// This is equivalent to [a, b]₊ ≠ 0 (anticommutator is nonzero).
///
/// Returns true iff odd(edges between support sets).
///
/// # Example
/// ```
/// assert!(anticommutes(Blade::x(0), Blade::z(0)));  // {e₂, e₁} = -e₁e₂ - e₂e₁
/// assert!(!anticommutes(Blade::x(0), Blade::x(1)));  // same qubit separation
/// ```
#[inline(always)]
pub fn anticommutes(a: Blade, b: Blade) -> bool {
    // Edge count: (a.x ∩ b.z) + (a.z ∩ b.x)
    let p = ((a.x & b.z).count_ones() + (a.z & b.x).count_ones()) & 1;
    p != 0
}

/// Bitwise test: do two blades commute?
///
/// Equivalent to [a, b] = 0 (commutator vanishes).
/// This is the logical negation of anticommutes.
///
/// # Example
/// ```
/// assert!(commutes(Blade::x(0), Blade::x(1)));  // disjoint qubits
/// assert!(!commutes(Blade::x(0), Blade::z(0)));  // same qubit, different type
/// ```
#[inline(always)]
pub fn commutes(a: Blade, b: Blade) -> bool {
    !anticommutes(a, b)
}

// ============================================================================
// ALGEBRAIC PROPERTIES
// ============================================================================

/// Grassmann-Clifford grade of a blade.
///
/// Counts the number of linearly-independent basis elements in the wedge product:
/// - Grade 0: Scalar (empty product)
/// - Grade 1: Vector (e₁ or e₂)
/// - Grade 2: Bivector (e₁∧e₂ or cross-qubit vectors)
/// - Grade 3+: Higher wedge products
///
/// Formula: grade(b) = |b.x ⊕ b.z| + 2·|b.x ∧ b.z|
/// where |S| = number of 1-bits in S (popcount)
///
/// The bivector term (x∧z) counts twice because e₁∧e₂ is a single basis element
/// but occupies one position with two component qubits.
///
/// # Example
/// ```
/// assert_eq!(grade(Blade::x(0)), 1);      // e₂ is grade 1
/// assert_eq!(grade(Blade::j(0)), 2);      // e₁∧e₂ is grade 2
/// assert_eq!(grade(Blade::x(0) * Blade::z(1)), 2);  // e₂∧e₁ is grade 2
/// ```
#[inline(always)]
pub fn grade(b: Blade) -> u32 {
    (b.x ^ b.z).count_ones() + 2 * (b.x & b.z).count_ones()
}

/// Number of qubits participating in this blade.
///
/// Counts distinct qubit positions where either x or z bit is set.
/// Used to determine if a blade is "single-qubit" vs "multi-qubit".
///
/// Formula: support_size(b) = |b.x ∪ b.z| (union of x and z support sets)
///
/// # Example
/// ```
/// assert_eq!(support_size(Blade::x(0)), 1);           // single qubit 0
/// assert_eq!(support_size(Blade::x(0) * Blade::z(1)), 2);  // qubits 0,1
/// ```
#[inline(always)]
pub fn support_size(b: Blade) -> u32 {
    (b.x | b.z).count_ones()
}

// ============================================================================
// RUST TRAIT IMPL
// ============================================================================

/// Implement Rust's multiplication operator for blades.
///
/// Allows natural algebraic syntax: `let c = a * b;` instead of `gp_blade(a, b)`.
impl Mul for Blade {
    type Output = Blade;

    #[inline(always)]
    fn mul(self, rhs: Blade) -> Blade {
        gp_blade(self, rhs)
    }
}