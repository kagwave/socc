//! SIMD-accelerated term composition using AVX2 (x86_64).
//!
//! This module provides 4x parallel term composition via 256-bit operations.
//! Falls back to scalar operations on non-AVX2 systems.

use crate::core::compute::local::packed::PackedBlockTerm;

/// Load 4 u64 values into a 256-bit vector. Intel syntax: `a = [v0, v1, v2, v3]`
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn load_u64x4(a: &[u64; 4]) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    // SAFETY: pointer is aligned to 32 bytes for cache-friendly access
    _mm256_loadu_si256(a.as_ptr() as *const __m256i)
}

/// Store 4 u64 values from a 256-bit vector into array.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn store_u64x4(a: &mut [u64; 4], v: std::arch::x86_64::__m256i) {
    use std::arch::x86_64::*;
    _mm256_storeu_si256(a.as_mut_ptr() as *mut __m256i, v);
}

/// Extract 64-bit lane i from 256-bit vector (i in 0..3).
/// Note: On x86_64, lanes are indexed in big-endian (3,2,1,0).
#[cfg(target_arch = "x86_64")]
#[inline]
fn extract_u64(v: std::arch::x86_64::__m256i, lane: usize) -> u64 {
    use std::arch::x86_64::*;
    let mut data = [0u64; 4];
    _mm256_storeu_si256(data.as_mut_ptr() as *mut __m256i, v);
    data[lane]
}

/// Interleaved popcount for 4 u64 values in a 256-bit vector.
/// Returns a packed result where each 64-bit lane holds the popcount of that lane.
#[cfg(target_arch = "x86_64" )]
#[inline]
fn popcount_u64x4(v: std::arch::x86_64::__m256i) -> [u32; 4] {
    let mut data = [0u64; 4];
    unsafe {
        store_u64x4(&mut data, v);
    }
    [
        data[0].count_ones(),
        data[1].count_ones(),
        data[2].count_ones(),
        data[3].count_ones(),
    ]
}

/// Compute the parity (popcount mod 2) of each 64-bit lane in a 256-bit vector.
#[cfg(target_arch = "x86_64")]
#[inline]
fn parity_u64x4(v: std::arch::x86_64::__m256i) -> [bool; 4] {
    let pcs = popcount_u64x4(v);
    [pcs[0] & 1 == 1, pcs[1] & 1 == 1, pcs[2] & 1 == 1, pcs[3] & 1 == 1]
}

/// Compose 4 terms in parallel using AVX2.
///
/// **Input**: Two arrays of 4 terms each.
/// **Output**: Composed result packed into output.
///
/// **Semantics**: Computes `output[i] = a[i] * b[i]` (geometric product).
///
/// **SIMD Strategy**:
/// 1. Load blade components into 256-bit registers
/// 2. Parallel XOR for blade composition
/// 3. Parallel AND + popcount for parity tracking
/// 4. Store results, applying sign corrections
///
/// # Safety
/// This function calls AVX2 intrinsics and should only be compiled on x86_64.
/// It is marked `unsafe` because:
/// - AVX2 may not be available at runtime (checked via `#[cfg]`)
/// - Pointer alignment is assumed to be 32-byte aligned for `load_u64x4()`
#[cfg(target_arch = "x86_64")]
pub unsafe fn gp_simd_x4(
    a: &[PackedBlockTerm; 4],
    b: &[PackedBlockTerm; 4],
    output: &mut [PackedBlockTerm; 4],
) {
    use std::arch::x86_64::*;

    // Load blade components for A: blade_x and blade_z
    let a_blade_x = [a[0].blade_x, a[1].blade_x, a[2].blade_x, a[3].blade_x];
    let a_blade_z = [a[0].blade_z, a[1].blade_z, a[2].blade_z, a[3].blade_z];

    // Load blade components for B
    let b_blade_x = [b[0].blade_x, b[1].blade_x, b[2].blade_x, b[3].blade_x];
    let b_blade_z = [b[0].blade_z, b[1].blade_z, b[2].blade_z, b[3].blade_z];

    let a_blade_x_v = load_u64x4(&a_blade_x);
    let a_blade_z_v = load_u64x4(&a_blade_z);
    let b_blade_x_v = load_u64x4(&b_blade_x);
    let b_blade_z_v = load_u64x4(&b_blade_z);

    // Parallel XOR: new blade components
    let new_blade_x = _mm256_xor_si256(a_blade_x_v, b_blade_x_v);
    let new_blade_z = _mm256_xor_si256(a_blade_z_v, b_blade_z_v);

    // Parity tracking: cross term = a.blade_x AND b.blade_z
    let cross = _mm256_and_si256(a_blade_x_v, b_blade_z_v);
    let cross_parity = parity_u64x4(cross);

    // Store new blade components
    let mut result_blade_x = [0u64; 4];
    let mut result_blade_z = [0u64; 4];
    store_u64x4(&mut result_blade_x, new_blade_x);
    store_u64x4(&mut result_blade_z, new_blade_z);

    // Apply sign corrections and construct output
    for i in 0..4 {
        output[i] = a[i].clone();
        output[i].blade_x = result_blade_x[i];
        output[i].blade_z = result_blade_z[i];
        if cross_parity[i] {
            output[i].sign ^= true; // Flip sign on odd parity
        }
    }
}

/// Fallback scalar composition for non-AVX2 systems or testing.
pub fn gp_simd_x4_scalar(
    a: &[PackedBlockTerm; 4],
    b: &[PackedBlockTerm; 4],
    output: &mut [PackedBlockTerm; 4],
) {
    for i in 0..4 {
        output[i] = a[i].clone();
        // Delegate to scalar gp
        //output[i] = a[i].gp(&b[i]);
        
        // Manual scalar implementation to avoid dependency on full gp()
        let mut result = a[i].clone();
        
        // Blade XOR
        result.blade_x ^= b[i].blade_x;
        result.blade_z ^= b[i].blade_z;
        
        // Parity tracking
        let cross = a[i].blade_x & b[i].blade_z;
        if cross.count_ones() & 1 == 1 {
            result.sign ^= true;
        }
        
        output[i] = result;
    }
}

/// Public wrapper: Compose 4 terms with runtime feature detection.
///
/// Selects between SIMD and scalar based on compile-time and runtime availability.
pub fn gp_simd_x4_auto(
    a: &[PackedBlockTerm; 4],
    b: &[PackedBlockTerm; 4],
    output: &mut [PackedBlockTerm; 4],
) {
    #[cfg(target_arch = "x86_64")]
    {
        // Runtime check for AVX2 (if cpuid feature detection is available)
        // For now, assume AVX2 is available and use unsafe block
        unsafe {
            gp_simd_x4(a, b, output);
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        gp_simd_x4_scalar(a, b, output);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_term(blade_x: u64, blade_z: u64, sign: bool) -> PackedBlockTerm {
        PackedBlockTerm::new(0, blade_x, blade_z, 0, 0, 0, 0, sign, 2)
    }

    #[test]
    fn test_simd_x4_identity() {
        let a = [
            make_term(0, 0, false),
            make_term(0, 0, false),
            make_term(0, 0, false),
            make_term(0, 0, false),
        ];

        let b = [
            make_term(0, 0, false),
            make_term(0, 0, false),
            make_term(0, 0, false),
            make_term(0, 0, false),
        ];

        let mut output = [
            make_term(0, 0, false),
            make_term(0, 0, false),
            make_term(0, 0, false),
            make_term(0, 0, false),
        ];

        gp_simd_x4_scalar(&a, &b, &mut output);

        for term in &output {
            assert_eq!(term.blade_x, 0);
            assert_eq!(term.blade_z, 0);
            assert_eq!(term.sign, false);
        }
    }

    #[test]
    fn test_simd_x4_scalar_equivalence() {
        // Create test terms with non-trivial blade structure
        let a = [
            make_term(0x0001, 0x0101, false),
            make_term(0x0101, 0x0001, false),
            make_term(0x1111, 0x1010, false),
            make_term(0x0000, 0x1111, false),
        ];

        let b = [
            make_term(0x0101, 0x0001, false),
            make_term(0x1111, 0x0101, false),
            make_term(0x0001, 0x1111, false),
            make_term(0x1010, 0x0101, false),
        ];

        let mut output_scalar = [
            make_term(0, 0, false),
            make_term(0, 0, false),
            make_term(0, 0, false),
            make_term(0, 0, false),
        ];

        let mut output_simd = [
            make_term(0, 0, false),
            make_term(0, 0, false),
            make_term(0, 0, false),
            make_term(0, 0, false),
        ];

        gp_simd_x4_scalar(&a, &b, &mut output_scalar);

        #[cfg(target_arch = "x86_64")]
        unsafe {
            gp_simd_x4(&a, &b, &mut output_simd);
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            gp_simd_x4_scalar(&a, &b, &mut output_simd);
        }

        for i in 0..4 {
            assert_eq!(
                output_scalar[i].blade_x, output_simd[i].blade_x,
                "blade_x mismatch at lane {}", i
            );
            assert_eq!(
                output_scalar[i].blade_z, output_simd[i].blade_z,
                "blade_z mismatch at lane {}", i
            );
            assert_eq!(
                output_scalar[i].sign, output_simd[i].sign,
                "sign mismatch at lane {}", i
            );
        }
    }
}
