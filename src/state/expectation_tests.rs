#[cfg(test)]
mod tests {
    use crate::core::bits::{Blade, Sector};
    use crate::core::compute::multivector::scalar_part;
    use crate::core::ir::Multivector;
    use crate::state::expectation::expectation;
    use crate::state::ideal::IdealState;

    const EPS: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    #[test]
    fn expectation_runs_on_vacuum_state() {
        let p = Sector::from_bits(0, 1);

        let st = IdealState::new(
            Multivector::from_blade(1, Blade::x(0), 1.0),
            p,
        );

    let rho = st.density();
    let z = Multivector::from_blade(1, Blade::z(0), 1.0);

    let _val = expectation(&rho, &z, p);
    }

    #[test]
    fn expectation_of_identity_on_vacuum_is_nonzero_scalar() {
        // This is a weak but useful sanity test:
        // the identity should produce some scalar contribution on a nonzero state.
        let p = Sector::from_bits(0, 1);

        let st = IdealState::new(
            Multivector::from_blade(1, Blade::identity(), 1.0),
            p,
        );

    let rho = st.density();
    let id = Multivector::from_blade(1, Blade::identity(), 1.0);

    let val = expectation(&rho, &id, p);

        // We don't overconstrain normalization yet; just require it be finite and nonzero.
        assert!(val.is_finite());
        assert!(!approx_eq(val, 0.0));
    }

    #[test]
    fn scalar_part_of_identity_is_extracted() {
        let mv = Multivector::from_blade(1, Blade::identity(), 3.5);
        let s = scalar_part(&mv);

        assert!(approx_eq(s, 3.5));
    }
}