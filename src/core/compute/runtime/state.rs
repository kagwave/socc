use crate::core::compute::runtime::monomial_state::MonomialState;
use crate::core::ir::Multivector;

#[derive(Clone, Debug, PartialEq)]
pub enum RuntimeState {
    Monomial(MonomialState),
    Generic(Multivector),
}