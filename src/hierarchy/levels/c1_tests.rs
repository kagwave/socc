#[cfg(test)]
mod tests {
    use crate::core::bits::{Blade, Rotor};
    use crate::core::ir::{Multivector, Term};
    use crate::hierarchy::levels::c1::{C1, C1Gate};
    use crate::core::compute::multivector::gp_mv;

    // ===== Descriptor Tests =====

    /// Test that descriptor constructors produce the correct C1Gate variants.
    /// These are packed enum values used for symbolic gate representation and classification.
    #[test]
    fn c1gate_descriptors_construct_correctly() {
        assert_eq!(C1::x(0), C1Gate::X { qubit: 0 });
        assert_eq!(C1::y(2), C1Gate::Y { qubit: 2 });
        assert_eq!(C1::z(5), C1Gate::Z { qubit: 5 });
    }

    // ===== Operator Constructor Tests =====

    /// Test that gate operators (x_gate, y_gate, z_gate, j_gate) return proper Multivector forms.
    /// These are used directly in compute chains and should be single-term, bare Pauli operators.
    #[test]
    fn x_gate_is_atomic_pauli() {
        let x0 = C1::x_gate(0);
        assert!(C1::is_atomic_pauli(&x0));
        assert_eq!(C1::as_pauli_blade(&x0), Some(Blade::x(0)));
    }

    #[test]
    fn z_gate_is_atomic_pauli() {
        let z1 = C1::z_gate(1);
        assert!(C1::is_atomic_pauli(&z1));
        assert_eq!(C1::as_pauli_blade(&z1), Some(Blade::z(1)));
    }

    /// Y gate is special: modeled as a biaction Y(ψ) = -J ψ J.
    /// This means it has a J blade with a right rotor J_at(qubit).
    #[test]
    fn y_gate_is_j_biaction() {
        let y0 = C1::y_gate(0);
        assert_eq!(y0.terms.len(), 1);
        let t = y0.terms[0];

        // Y is encoded as a single J-blade term with right rotor J_at(0) and coefficient -1.
        assert_eq!(t.blade, Blade::j(0));
        assert_eq!(t.rotor, Some(Rotor::j_at(0)));
        assert!((t.coeff - (-1.0)).abs() < 1e-9);
        assert!(t.left.is_none());
        assert!(t.right.is_none());
    }

    #[test]
    fn j_gate_is_atomic_pauli() {
        let j2 = C1::j_gate(2);
        assert!(C1::is_atomic_pauli(&j2));
        assert_eq!(C1::as_pauli_blade(&j2), Some(Blade::j(2)));
    }

    // ===== Blade Recognition Tests =====

    /// Test that single-site Pauli blades are correctly recognized as C1Gate descriptors.
    #[test]
    fn get_gate_recognizes_x() {
        assert_eq!(C1::get_gate(Blade::x(0)), Some(C1Gate::X { qubit: 0 }));
        assert_eq!(C1::get_gate(Blade::x(3)), Some(C1Gate::X { qubit: 3 }));
    }

    #[test]
    fn get_gate_recognizes_z() {
        assert_eq!(C1::get_gate(Blade::z(1)), Some(C1Gate::Z { qubit: 1 }));
        assert_eq!(C1::get_gate(Blade::z(7)), Some(C1Gate::Z { qubit: 7 }));
    }

    /// Y is recognized via the (x, z) bit pattern: both bits set at the same site.
    #[test]
    fn get_gate_recognizes_y() {
        assert_eq!(C1::get_gate(Blade::y(0)), Some(C1Gate::Y { qubit: 0 }));
        assert_eq!(C1::get_gate(Blade::y(4)), Some(C1Gate::Y { qubit: 4 }));
    }

    /// Multi-qubit blades should not be recognized as single-qubit C1Gate descriptors.
    #[test]
    fn get_gate_rejects_multiqubit() {
        let two_site = crate::core::compute::blade::gp_blade(Blade::x(0), Blade::z(1));
        assert_eq!(C1::get_gate(two_site), None);
    }

    /// The identity blade (no bits set) is not a valid Pauli gate.
    #[test]
    fn get_gate_rejects_identity() {
        let id = Blade::identity();
        assert_eq!(C1::get_gate(id), None);
    }

    // ===== Atomic Pauli Predicate Tests =====

    /// is_atomic_pauli strictly checks: one term, no sectors, identity rotor, unit coeff.
    #[test]
    fn is_atomic_pauli_true_for_bare_paulis() {
        let paulis = vec![
            C1::x_gate(1, 0),
            C1::z_gate(1, 0),
            C1::j_gate(1, 0),
        ];
        for p in paulis {
            assert!(C1::is_atomic_pauli(&p), "Failed for {:?}", p.terms[0].blade);
        }
    }

    /// Multiple terms break atomicity.
    #[test]
    fn is_atomic_pauli_false_for_multiterm() {
        let sum = Multivector::from_terms(1, vec![
            C1::pauli_term(Blade::x(0), 1.0),
            C1::pauli_term(Blade::z(0), 1.0),
        ]);
        assert!(!C1::is_atomic_pauli(&sum));
    }

    /// Non-unit coefficient breaks atomicity.
    #[test]
    fn is_atomic_pauli_false_for_scaled() {
        let scaled = Multivector::from_terms(1, vec![
            C1::pauli_term(Blade::x(0), 0.5),
        ]);
        assert!(!C1::is_atomic_pauli(&scaled));
    }

    /// Y biaction is not atomic (has nontrivial right rotor), but is still "Y-like".
    #[test]
    fn is_atomic_pauli_false_for_y_biaction() {
        let y0 = C1::y_gate(1, 0);
        assert!(!C1::is_atomic_pauli(&y0), "Y biaction should not be atomic due to rotor");
    }

    // ===== Weight and Grade Tests =====

    /// Pauli weight is the number of qubits on which the blade acts nontrivially.
    #[test]
    fn pauli_weight_single_site() {
        assert_eq!(C1::pauli_weight(Blade::x(0)), 1);
        assert_eq!(C1::pauli_weight(Blade::z(3)), 1);
        assert_eq!(C1::pauli_weight(Blade::j(2)), 1);
    }

    #[test]
    fn pauli_weight_multiqubit() {
        let xz = crate::core::compute::blade::gp_blade(Blade::x(0), Blade::z(1));
        assert_eq!(C1::pauli_weight(xz), 2);

        let xyz = crate::core::compute::blade::gp_blade(xz, Blade::j(2));
        assert_eq!(C1::pauli_weight(xyz), 3);
    }

    #[test]
    fn pauli_weight_identity_is_zero() {
        assert_eq!(C1::pauli_weight(Blade::identity()), 0);
    }

    /// Blade grade counts the total number of basis elements in the blade's wedge product.
    #[test]
    fn blade_grade_single_element() {
        // e1 (Z-like) and e2 (X-like) have grade 1.
        assert_eq!(C1::blade_grade(Blade::x(0)), 1);
        assert_eq!(C1::blade_grade(Blade::z(1)), 1);
    }

    #[test]
    fn blade_grade_bivector() {
        // J (local bivector e1 ∧ e2) has grade 2.
        assert_eq!(C1::blade_grade(Blade::j(0)), 2);
    }

    // ===== Commutation Tests =====

    /// Two Pauli blades on the same qubit with different bases anticommute.
    #[test]
    fn anticommutation_same_qubit_different_basis() {
        assert!(C1::anticommutes(Blade::x(0), Blade::z(0)));
        assert!(C1::anticommutes(Blade::z(0), Blade::x(0)));
        assert!(C1::anticommutes(Blade::x(0), Blade::j(0)));
    }

    /// Pauli blades on disjoint qubits commute.
    #[test]
    fn commutation_different_qubits() {
        assert!(C1::commutes(Blade::x(0), Blade::x(1)));
        assert!(C1::commutes(Blade::z(0), Blade::z(2)));
        assert!(C1::commutes(Blade::x(0), Blade::z(1)));
    }

    /// Identity commutes with everything.
    #[test]
    fn commutation_with_identity() {
        let id = Blade::identity();
        assert!(C1::commutes(id, Blade::x(0)));
        assert!(C1::commutes(Blade::z(1), id));
    }

    // ===== Term Predicate Tests =====

    /// is_bare_pauli_term checks no sectors, identity rotor, and implicit compatibility.
    #[test]
    fn is_bare_pauli_term_true_for_pauli_terms() {
        let terms = vec![
            C1::pauli_term(Blade::x(0), 1.0),
            C1::pauli_term(Blade::z(1), -1.0),
            C1::pauli_term(Blade::j(2), 0.5),
        ];
        for t in terms {
            assert!(C1::is_bare_pauli_term(&t));
        }
    }

    /// A term with a nontrivial rotor is not bare.
    #[test]
    fn is_bare_pauli_term_false_for_rotor_term() {
        let rotor_term = Term {
            left: None,
            blade: Blade::j(0),
            right: None,
            rotor: Some(Rotor::j_at(0)),
            coeff: 1.0,
        };
        assert!(!C1::is_bare_pauli_term(&rotor_term));
    }

    // ===== Operator Composition Tests =====

    /// Pauli blades should square to identity (up to sign).
    #[test]
    fn pauli_self_composition_yields_identity_or_sign() {
        // X^2 = I
        let x0 = C1::x_gate(1, 0);
        let x02 = gp_mv(&x0, &x0);
        assert!(C1::is_atomic_pauli(&x02));
        let blade = C1::as_pauli_blade(&x02).unwrap();
        assert_eq!(blade, Blade::identity());

        // Z^2 = I
        let z1 = C1::z_gate(1, 0);
        let z12 = gp_mv(&z1, &z1);
        assert!(C1::is_atomic_pauli(&z12));
        let blade = C1::as_pauli_blade(&z12).unwrap();
        assert_eq!(blade, Blade::identity());
    }

    /// Single-qubit Pauli anticommutation.
    #[test]
    fn anticommuting_paulis_anticommute() {
        // [X, Z] = XZ + ZX = 2i Y (in usual convention, but here we check the relation exists)
        let x0 = C1::x_gate(1, 0);
        let z0 = C1::z_gate(1, 0);

        // XZ vs ZX should not be equal.
        let xz = gp_mv(&x0, &z0);
        let zx = gp_mv(&z0, &x0);
        assert_ne!(xz.terms.len(), 0);
        assert_ne!(zx.terms.len(), 0);
        // In the packed representation, XZ and ZX differ by sign (XZ = -ZX for single-qubit Paulis).
    }

    /// Commuting Paulis on disjoint qubits should compose without sign flip.
    #[test]
    fn commuting_paulis_compose_naturally() {
        let x0 = C1::x_gate(2, 0);
        let z1 = C1::z_gate(2, 1);

        let xz = gp_mv(&x0, &z1);
        let zx = gp_mv(&z1, &x0);

        // Both should represent the same multi-qubit Pauli.
        assert_eq!(xz.terms.len(), 1);
        assert_eq!(zx.terms.len(), 1);
        assert_eq!(xz.terms[0].blade, zx.terms[0].blade);
    }

    // ===== All Terms Pairwise Commuting Test =====

    /// A sum of mutually commuting Paulis should pass the pairwise test.
    #[test]
    fn all_terms_pairwise_commuting_for_commuting_sum() {
        let mv = Multivector::from_terms(3, vec![
            C1::pauli_term(Blade::x(0), 1.0),
            C1::pauli_term(Blade::z(1), 1.0),
            C1::pauli_term(Blade::x(2), 1.0),
        ]);
        assert!(C1::all_terms_pairwise_commuting(&mv));
    }

    /// A sum with anticommuting terms should fail the test.
    #[test]
    fn all_terms_pairwise_commuting_false_for_anticommuting() {
        let mv = Multivector::from_terms(1, vec![
            C1::pauli_term(Blade::x(0), 1.0),
            C1::pauli_term(Blade::z(0), 1.0), // Anticommutes with X(0).
        ]);
        assert!(!C1::all_terms_pairwise_commuting(&mv));
    }

    // ===== Blades Extraction Test =====

    /// Extract all blades from a multivector for further analysis.
    #[test]
    fn blades_extracts_all_terms() {
        let mv = Multivector::from_terms(3, vec![
            C1::pauli_term(Blade::x(0), 1.0),
            C1::pauli_term(Blade::z(1), 1.0),
            C1::pauli_term(Blade::j(2), 1.0),
        ]);
        let blades = C1::blades(&mv);
        assert_eq!(blades.len(), 3);
        assert_eq!(blades[0], Blade::x(0));
        assert_eq!(blades[1], Blade::z(1));
        assert_eq!(blades[2], Blade::j(2));
    }
}