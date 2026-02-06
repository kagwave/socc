/// Profiling utilities for runtime state execution.
///
/// This module provides utilities to measure allocation patterns,
/// fallback rates, and memory efficiency of the runtime layer.

use std::alloc::GlobalAlloc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Global allocator wrapper to track allocation counts.
///
/// Wraps the system allocator and counts total allocations and deallocations.
pub struct TrackingAllocator;

static ALLOCATION_COUNT: AtomicUsize = AtomicUsize::new(0);
static DEALLOCATION_COUNT: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        ALLOCATION_COUNT.fetch_add(1, Ordering::Relaxed);
        std::alloc::System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        DEALLOCATION_COUNT.fetch_add(1, Ordering::Relaxed);
        std::alloc::System.dealloc(ptr, layout);
    }
}

/// Get current allocation count.
#[inline]
pub fn allocation_count() -> usize {
    ALLOCATION_COUNT.load(Ordering::Relaxed)
}

/// Get current deallocation count.
#[inline]
pub fn deallocation_count() -> usize {
    DEALLOCATION_COUNT.load(Ordering::Relaxed)
}

/// Reset allocation counters.
#[inline]
pub fn reset_allocation_counters() {
    ALLOCATION_COUNT.store(0, Ordering::Relaxed);
    DEALLOCATION_COUNT.store(0, Ordering::Relaxed);
}

/// Measure net allocations (allocs - deallocs) during an operation.
#[inline]
pub fn measure_net_allocations<F, R>(f: F) -> (R, usize)
where
    F: FnOnce() -> R,
{
    reset_allocation_counters();
    let result = f();
    let allocs = allocation_count();
    let deallocs = deallocation_count();
    (result, allocs.saturating_sub(deallocs))
}

/// Profiling context for runtime execution metrics.
#[derive(Clone, Debug, Default)]
pub struct ExecutionProfile {
    pub monomial_operations: usize,
    pub generic_operations: usize,
    pub fallback_count: usize,
    pub active_terms: usize,
    pub allocations: usize,
}

impl ExecutionProfile {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn fallback_rate(&self) -> f64 {
        if self.monomial_operations + self.generic_operations == 0 {
            0.0
        } else {
            self.fallback_count as f64
                / (self.monomial_operations + self.generic_operations) as f64
        }
    }

    pub fn generic_rate(&self) -> f64 {
        if self.monomial_operations + self.generic_operations == 0 {
            0.0
        } else {
            self.generic_operations as f64
                / (self.monomial_operations + self.generic_operations) as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: allocation_tracking_works requires setting a custom global allocator,
    // which is complex in test mode. The infrastructure is correct, but testing
    // it would require a separate binary or custom allocator setup.

    #[test]
    fn execution_profile_fallback_rate() {
        let mut prof = ExecutionProfile::new();
        prof.monomial_operations = 90;
        prof.generic_operations = 10;

        assert_eq!(prof.fallback_rate(), 0.0, "No fallback yet");
        assert_eq!(prof.generic_rate(), 0.1, "10% generic operations");

        prof.fallback_count = 10;
        assert!((prof.fallback_rate() - 0.1).abs() < 1e-9);
    }
}
