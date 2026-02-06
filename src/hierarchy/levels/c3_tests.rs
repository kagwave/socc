
#[cfg(test)]
mod tests {
    use crate::core::bits::{Blade, Rotor};
    use crate::core::ir::{Multivector, Term};
    use crate::hierarchy::levels::c3::{C3, C3Gate};

    #[test]
    fn c3gate_descriptors_construct() {
        assert_eq!(C3::t(0), C3Gate::T { qubit: 0 });
        assert_eq!(
            C3::controlled_s(1, 3),
            C3Gate::ControlledS {
                control: 1,
                target: 3
            }
        );
        assert_eq!(
            C3::toffoli(0, 2, 5),
            C3Gate::Toffoli {
                control1: 0,
                control2: 2,
                target: 5
            }
        );
    }

    #[test]
    fn t_gate_is_sector_form_with_quarter_turn_rotor() {
        let t = C3::t_gate(2);
        assert_eq!(t.terms.len(), 2);

        let a = t.terms[0];
        let b = t.terms[1];

        assert!(a.left.is_some());
        assert!(b.left.is_some());

        let s0 = a.left.unwrap();
        let s1 = b.left.unwrap();

        assert_eq!(s0.n, 3);
        assert_eq!(s1.n, 3);
        assert_eq!(s0.bits ^ s1.bits, 1u64 << 2);

        for term in &t.terms {
            assert!(term.blade.is_identity());
            assert!(term.right.is_none());
            assert!((term.coeff - 1.0).abs() < 1e-9);
        }

        let rotors = vec![a.rotor, b.rotor];
        assert!(rotors.contains(&None));
        assert!(rotors.contains(&Some(Rotor::quarter_turn_at(2))));
    }

    #[test]
    fn controlled_s_gate_is_sector_form_with_j_rotor() {
        let cs = C3::controlled_s_gate(0, 2);
        assert_eq!(cs.terms.len(), 2);

        let a = cs.terms[0];
        let b = cs.terms[1];

        assert!(a.left.is_some());
        assert!(b.left.is_some());

        let rotors = vec![a.rotor, b.rotor];
        assert!(rotors.contains(&None));
        assert!(rotors.contains(&Some(Rotor::j_at(2))));

        for term in &cs.terms {
            assert!(term.blade.is_identity());
            assert!(term.right.is_none());
            assert!((term.coeff - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn toffoli_gate_has_four_sector_branches() {
        let toff = C3::toffoli_gate(0, 1, 3);
        assert_eq!(toff.terms.len(), 4);

        let mut sectors: Vec<u64> = toff
            .terms
            .iter()
            .map(|t| t.left.unwrap().bits)
            .collect();
        sectors.sort_unstable();

        assert_eq!(sectors, vec![0, 1, 2, 3]);

        let x_count = toff.terms.iter().filter(|t| t.blade == Blade::x(3)).count();
        let id_count = toff
            .terms
            .iter()
            .filter(|t| t.blade.is_identity())
            .count();

        assert_eq!(x_count, 1);
        assert_eq!(id_count, 3);
    }

    #[test]
    fn get_gate_recognizes_t() {
        let t = C3::t_gate(1);
        assert_eq!(C3::get_gate(&t), Some(C3Gate::T { qubit: 1 }));
    }

    #[test]
    fn get_gate_recognizes_controlled_s() {
        let cs = C3::controlled_s_gate(1, 4);
        assert_eq!(
            C3::get_gate(&cs),
            Some(C3Gate::ControlledS {
                control: 1,
                target: 4
            })
        );
    }

    #[test]
    fn get_gate_recognizes_toffoli() {
        let toff = C3::toffoli_gate(0, 2, 5);
        assert_eq!(
            C3::get_gate(&toff),
            Some(C3Gate::Toffoli {
                control1: 0,
                control2: 2,
                target: 5
            })
        );
    }

    #[test]
    fn get_gate_rejects_old_bare_t_form() {
        let bad = Multivector::from_terms(1, vec![
            Term {
                left: None,
                blade: Blade::identity(),
                right: None,
                rotor: None,
                coeff: 1.0 / 2.0_f64.sqrt(),
            },
            Term {
                left: None,
                blade: Blade::j(0),
                right: None,
                rotor: None,
                coeff: 1.0 / 2.0_f64.sqrt(),
            },
        ]);

        assert_eq!(C3::get_gate(&bad), None);
    }

    #[test]
    fn is_c3_shaped_accepts_constructed_gates() {
        assert!(C3::is_c3_shaped(&C3::t_gate(1, 0)));
        assert!(C3::is_c3_shaped(&C3::controlled_s_gate(2, 0, 1)));
        assert!(C3::is_c3_shaped(&C3::toffoli_gate(3, 0, 1, 2)));
    }

    #[test]
    fn is_c3_shaped_rejects_empty() {
        assert!(!C3::is_c3_shaped(&Multivector::from_terms(1, vec![])));
    }
}