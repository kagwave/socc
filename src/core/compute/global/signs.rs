use crate::core::config::AlgebraConfig;

/// Koszul sign for swapping grade-k and grade-l elements.
#[inline(always)]
pub fn koszul_sign(k: u32, l: u32) -> bool {
    if AlgebraConfig::USE_KOSZUL {
        ((k * l) & 1) != 0
    } else {
        false
    }
}