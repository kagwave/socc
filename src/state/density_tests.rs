#[cfg(test)]
mod tests {
    use crate::core::bits::{Blade, Sector};
    use crate::core::ir::Multivector;
    use crate::state::ideal::IdealState;

    #[test]
    fn vacuum_density_is_nonempty() {
        // ψ = 1 * P
        let p = Sector::from_bits(0, 1);

        let st = IdealState::new(
            Multivector::from_blade(1, Blade::identity(), 1.0),
            p,
        );

        let rho = st.density();
        assert!(!rho.mv.terms.is_empty());
    }

    #[test]
    fn x_state_density_is_nonempty() {
        // ψ = X * P
        let p = Sector::from_bits(0, 1);

        let st = IdealState::new(
            Multivector::from_blade(1, Blade::x(0), 1.0),
            p,
        );

        let rho = st.density();
        assert!(!rho.mv.terms.is_empty());
    }

    #[test]
    fn density_of_same_state_is_deterministic() {
        let p = Sector::from_bits(0, 1);

        let st = IdealState::new(
            Multivector::from_blade(1, Blade::x(0), 1.0),
            p,
        );

        let rho1 = st.density();
        let rho2 = st.density();

        assert_eq!(rho1, rho2);
    }
}