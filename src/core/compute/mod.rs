pub mod blade;
pub mod involutions;
pub mod local;
pub mod lower;
pub mod mem;
pub mod multivector;
pub mod rotor;
pub mod sector;
pub mod state;
pub mod term;
pub mod structured;
pub mod reference;
pub mod multivector_packed;
pub mod runtime;
pub mod profiling;
// pub mod pqk;  // TODO: Implement or remove

#[cfg(test)]
mod rotor_tests;

#[cfg(test)]
mod sector_tests;

#[cfg(test)]
mod structured_tests;

// #[cfg(test)]
// mod term_tests;  // TODO: fix to use new gp_term(a, b, n) signature

// #[cfg(test)]
// mod tests;  // TODO: fix Term::sector_map and gp_term calls