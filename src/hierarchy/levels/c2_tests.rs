#[cfg(test)]
mod tests {
    use crate::core::bits::{Blade, Rotor};
    use crate::core::ir::{Multivector, Term};
    use crate::hierarchy::levels::c2::{C2, C2Gate};

    // ===== Conjugation Tests (Blade-level) =====

    #[test]
    fn h_swaps_x_and_z() {
        let x = Blade::x(0);
        let z = Blade::z(0);

        assert_eq!(C2::conjugate_blade_by_h(x, 0), z);
        assert_eq!(C2::conjugate_blade_by_h(z, 0), x);
    }

    #[test]
    fn h_flips_sign_on_j() {
        let j = Blade::j(0);
        let result = C2::conjugate_blade_by_h(j, 0);
        assert_eq!(result.x, j.x);
        assert_eq!(result.z, j.z);
        assert_eq!(result.sign, !j.sign);
    }

    #[test]
    fn s_maps_x_to_xz() {
        let x = Blade::x(0);
        let result = C2::conjugate_blade_by_s(x, 0);
        assert_eq!(result, Blade::new((1u64 << 0), (1u64 << 0), false));
    }

    #[test]
    fn s_leaves_z_unchanged() {
        let z = Blade::z(0);
        let result = C2::conjugate_blade_by_s(z, 0);
        assert_eq!(result, z);
    }

    #[test]
    fn s_maps_j_to_minus_x() {
        let j = Blade::j(0);
        let result = C2::conjugate_blade_by_s(j, 0);
        let expected = Blade::x(0);
        assert_eq!(result.x, expected.x);
        assert_eq!(result.z, expected.z);
        assert_eq!(result.sign, !expected.sign);
    }

    #[test]
    fn cnot_maps_xc_to_xc_xt() {
        let x_c = Blade::x(0);
        let result = C2::conjugate_blade_by_cnot(x_c, 0, 1);
        let expected = Blade::new((1u64 << 0) | (1u64 << 1), 0, false);
        assert_eq!(result, expected);
    }

    #[test]
    fn cnot_maps_zt_to_zc_zt() {
        let z_t = Blade::z(1);
        let result = C2::conjugate_blade_by_cnot(z_t, 0, 1);
        let expected = Blade::new(0, (1u64 << 0) | (1u64 << 1), false);
        assert_eq!(result, expected);
    }

    #[test]
    fn cnot_leaves_zc_unchanged() {
        let z_c = Blade::z(0);
        let result = C2::conjugate_blade_by_cnot(z_c, 0, 1);
        assert_eq!(result, z_c);
    }

    #[test]
    fn generic_gate_dispatch_works() {
        let x = Blade::x(0);
        let out = C2::conjugate_blade(x, C2Gate::H { qubit: 0 });
        assert_eq!(out, Blade::z(0));
    }

    // ===== Gate Operator Constructor Tests =====

    #[test]
    fn h_gate_is_bare_superposition() {
        let h = C2::h_gate(0);
        assert_eq!(h.terms.len(), 2);
        let s = 1.0 / 2.0_f64.sqrt();

        for t in &h.terms {
            assert!(t.left.is_none());
            assert!(t.right.is_none());
            assert!(t.rotor.is_none());
            assert!((t.coeff.abs() - s).abs() < 1e-9);
        }

        let blades = vec![h.terms[0].blade, h.terms[1].blade];
        assert!(blades.contains(&Blade::x(0)));
        assert!(blades.contains(&Blade::z(0)));
    }

    #[test]
    fn s_gate_is_sectored_with_rotor() {
        let s = C2::s_gate(0);
        assert_eq!(s.terms.len(), 2);

        assert!(s.terms[0].left.is_some());
        assert!(s.terms[1].left.is_some());

        let sector0 = s.terms[0].left.unwrap();
        let sector1 = s.terms[1].left.unwrap();

        assert_eq!(sector0.n, 1);
        assert_eq!(sector1.n, 1);
        assert_eq!(sector0.bits ^ sector1.bits, 1);

        for t in &s.terms {
            assert!(t.blade.is_identity());
            assert!(t.right.is_none());
            assert!((t.coeff - 1.0).abs() < 1e-9);
        }

        let rotors = vec![s.terms[0].rotor, s.terms[1].rotor];
        assert!(rotors.contains(&None));
        assert!(rotors.contains(&Some(Rotor::j_at(0))));
    }

    #[test]
    fn cnot_gate_is_sectored_with_xtarget() {
        let cnot = C2::cnot_gate(0, 1);
        assert_eq!(cnot.terms.len(), 2);

        assert!(cnot.terms[0].left.is_some());
        assert!(cnot.terms[1].left.is_some());

        let sector0 = cnot.terms[0].left.unwrap();
        let sector1 = cnot.terms[1].left.unwrap();

        assert_eq!(sector0.n, 2);
        assert_eq!(sector1.n, 2);
        assert!((sector0.bits == 0 && sector1.bits == 1) || (sector0.bits == 1 && sector1.bits == 0));

        let blades = vec![cnot.terms[0].blade, cnot.terms[1].blade];
        assert!(blades.contains(&Blade::identity()));
        assert!(blades.contains(&Blade::x(1)));

        for t in &cnot.terms {
            assert!(t.right.is_none());
            assert!(t.rotor.is_none());
            assert!((t.coeff - 1.0).abs() < 1e-9);
        }
    }

    // ===== Gate Recognition Tests =====

    #[test]
    fn get_gate_recognizes_h() {
        let h = C2::h_gate(0);
        let recognized = C2::get_gate(&h);
        assert_eq!(recognized, Some(C2Gate::H { qubit: 0 }));
    }

    #[test]
    fn get_gate_recognizes_h_different_qubits() {
        let h2 = C2::h_gate(3);
        let recognized = C2::get_gate(&h2);
        assert_eq!(recognized, Some(C2Gate::H { qubit: 3 }));
    }

    #[test]
    fn get_gate_recognizes_s() {
        let s = C2::s_gate(0);
        let recognized = C2::get_gate(&s);
        assert_eq!(recognized, Some(C2Gate::S { qubit: 0 }));
    }

    #[test]
    fn get_gate_recognizes_s_different_qubits() {
        let s2 = C2::s_gate(2);
        let recognized = C2::get_gate(&s2);
        assert_eq!(recognized, Some(C2Gate::S { qubit: 2 }));
    }

    #[test]
    fn get_gate_recognizes_cnot() {
        let cnot = C2::cnot_gate(0, 1);
        let recognized = C2::get_gate(&cnot);
        assert_eq!(recognized, Some(C2Gate::CNOT { control: 0, target: 1 }));
    }

    #[test]
    fn get_gate_recognizes_cnot_different_qubits() {
        let cnot = C2::cnot_gate(2, 5);
        let recognized = C2::get_gate(&cnot);
        assert_eq!(recognized, Some(C2Gate::CNOT { control: 2, target: 5 }));
    }

    #[test]
    fn get_gate_rejects_wrong_structure() {
        let single = Multivector::from_blade(1, Blade::x(0), 1.0);
        assert_eq!(C2::get_gate(&single), None);

        let three = Multivector::from_terms(1, vec![
            Term {
                left: None,
                blade: Blade::x(0),
                right: None,
                rotor: None,
                coeff: 0.5,
            },
            Term {
                left: None,
                blade: Blade::z(0),
                right: None,
                rotor: None,
                coeff: 0.5,
            },
            Term {
                left: None,
                blade: Blade::j(0),
                right: None,
                rotor: None,
                coeff: 0.0,
            },
        ]);
        assert_eq!(C2::get_gate(&three), None);
    }

    #[test]
    fn get_gate_rejects_wrong_coefficients() {
        let bad = Multivector::from_terms(1, vec![
            Term {
                left: None,
                blade: Blade::x(0),
                right: None,
                rotor: None,
                coeff: 0.5,
            },
            Term {
                left: None,
                blade: Blade::z(0),
                right: None,
                rotor: None,
                coeff: 0.5,
            },
        ]);
        assert_eq!(C2::get_gate(&bad), None);
    }

    #[test]
    fn get_gate_rejects_old_s_form_with_left_j_blade() {
        let bad = Multivector::from_terms(1, vec![
            Term {
                left: Some(crate::core::bits::Sector::new(0, 1)),
                blade: Blade::identity(),
                right: None,
                rotor: None,
                coeff: 1.0,
            },
            Term {
                left: Some(crate::core::bits::Sector::new(1, 1)),
                blade: Blade::j(0),
                right: None,
                rotor: None,
                coeff: 1.0,
            },
        ]);
        assert_eq!(C2::get_gate(&bad), None);
    }

    #[test]
    fn is_clifford_shaped_for_gates() {
        assert!(C2::is_clifford_shaped(&C2::h_gate(0)));
        assert!(C2::is_clifford_shaped(&C2::s_gate(0)));
        assert!(C2::is_clifford_shaped(&C2::cnot_gate(0, 1)));
    }

    #[test]
    fn is_clifford_shaped_for_paulis() {
        let pauli = Multivector::from_blade(1, Blade::x(0), 1.0);
        assert!(C2::is_clifford_shaped(&pauli));

        let pauli2 = Multivector::from_blade(2, Blade::z(1), 1.0);
        assert!(C2::is_clifford_shaped(&pauli2));
    }

    #[test]
    fn is_clifford_shaped_rejects_empty() {
        let empty = Multivector::from_terms(1, vec![]);
        assert!(!C2::is_clifford_shaped(&empty));
    }
}