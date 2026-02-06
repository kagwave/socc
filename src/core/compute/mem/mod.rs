//! Memory-layout optimized geometric product kernels.
//!
//! This module provides vectorized and memory-optimized implementations
//! of core GP operations, including portable SIMD-style loop unrolling.

pub mod simd_packed;
pub mod simd_term;

pub use simd_term::{gp_simd_x4_scalar, gp_simd_x4_auto, gp_bulk_unrolled};
