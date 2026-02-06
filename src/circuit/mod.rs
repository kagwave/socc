pub mod builder;
pub mod gate;
// TODO: Fix circuit ops implementation - missing apply method
// pub mod ops;

pub use builder::Circuit;

#[cfg(test)]
mod tests;

// #[cfg(test)]
// mod tests;  // TODO: Fix circuit implementation and ops