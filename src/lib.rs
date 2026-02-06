// Minimal crate root exposing only the stable modules used by tests.

pub mod core;
pub mod circuit;
pub mod state;
pub mod operator;
pub mod hierarchy;
pub mod stabilizer;

// Add other top-level modules (circuit, gates, hierarchy, py, socc, action)
// back here once their definitions are stable and present on disk.

// No curated re-exports for now; keep the root simple while the core
// compute and state layers are under active development.
