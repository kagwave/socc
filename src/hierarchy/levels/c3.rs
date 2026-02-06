use crate::core::bits::{Blade, Rotor, Sector};
use crate::core::ir::{Multivector, Term};

/// Packed C3 gate descriptors.
///
/// These are level-3 hierarchy gates:
/// - T: sends Paulis to Cliffords under conjugation
/// - Controlled-S: controlled phase in C3
/// - Toffoli: doubly-controlled X in C3
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum C3Gate {
    T { qubit: usize },
    ControlledS { control: usize, target: usize },
    Toffoli { control1: usize, control2: usize, target: usize },
}

/// Level-C3 utilities: sector-form third-level gates.
pub struct C3;

impl C3 {
    // ============================================================
    // Gate descriptors
    // ============================================================

    #[inline(always)]
    pub fn t(qubit: usize) -> C3Gate {
        C3Gate::T { qubit }
    }

    #[inline(always)]
    pub fn controlled_s(control: usize, target: usize) -> C3Gate {
        C3Gate::ControlledS { control, target }
    }

    #[inline(always)]
    pub fn toffoli(control1: usize, control2: usize, target: usize) -> C3Gate {
        C3Gate::Toffoli {
            control1,
            control2,
            target,
        }
    }

    // ============================================================
    // Operator constructors
    // ============================================================

    /// T gate:
    ///
    ///     T_i = P_i + Q_i R_i^(π/4)
    ///
    /// The nontrivial branch is represented by a local right rotor
    /// `Rotor::quarter_turn_at(i)`.
    pub fn t_gate(n: u8, qubit: usize) -> Multivector {
        debug_assert!(qubit < n as usize);

        Multivector::from_terms(
            n,
            vec![
                Term {
                    left: Some(Sector::new(0, n)),
                    blade: Blade::identity(),
                    right: None,
                    rotor: None,
                    coeff: 1.0,
                },
                Term {
                    left: Some(Sector::new(1u64 << qubit, n)),
                    blade: Blade::identity(),
                    right: None,
                    rotor: Some(Rotor::quarter_turn_at(qubit)),
                    coeff: 1.0,
                },
            ],
        )
    }

    /// Controlled-S:
    ///
    ///     CS(c,t) = P_c + Q_c R_t
    ///
    /// where `R_t = J_t` is represented as a right rotor on target.
    pub fn controlled_s_gate(n: u8, control: usize, target: usize) -> Multivector {
        debug_assert!(control < n as usize);
        debug_assert!(target < n as usize);
        debug_assert_ne!(control, target);

        Multivector::from_terms(
            n,
            vec![
                Term {
                    left: Some(Sector::new(0, n)),
                    blade: Blade::identity(),
                    right: None,
                    rotor: None,
                    coeff: 1.0,
                },
                Term {
                    left: Some(Sector::new(1u64 << control, n)),
                    blade: Blade::identity(),
                    right: None,
                    rotor: Some(Rotor::j_at(target)),
                    coeff: 1.0,
                },
            ],
        )
    }

    /// Toffoli:
    ///
    /// - P_c1 P_c2 -> I
    /// - Q_c1 P_c2 -> I
    /// - P_c1 Q_c2 -> I
    /// - Q_c1 Q_c2 -> X_t
    ///
    /// This is the natural sector-controlled generalization of CNOT.
    pub fn toffoli_gate(n: u8, control1: usize, control2: usize, target: usize) -> Multivector {
        debug_assert!(control1 < n as usize);
        debug_assert!(control2 < n as usize);
        debug_assert!(target < n as usize);
        debug_assert_ne!(control1, control2);
        debug_assert_ne!(target, control1);
        debug_assert_ne!(target, control2);

        let b1 = 1u64 << control1;
        let b2 = 1u64 << control2;

        Multivector::from_terms(
            n,
            vec![
                Term {
                    left: Some(Sector::new(0, n)),
                    blade: Blade::identity(),
                    right: None,
                    rotor: None,
                    coeff: 1.0,
                },
                Term {
                    left: Some(Sector::new(b1, n)),
                    blade: Blade::identity(),
                    right: None,
                    rotor: None,
                    coeff: 1.0,
                },
                Term {
                    left: Some(Sector::new(b2, n)),
                    blade: Blade::identity(),
                    right: None,
                    rotor: None,
                    coeff: 1.0,
                },
                Term {
                    left: Some(Sector::new(b1 | b2, n)),
                    blade: Blade::x(target),
                    right: None,
                    rotor: None,
                    coeff: 1.0,
                },
            ],
        )
    }

    // ============================================================
    // Recognition
    // ============================================================

    /// Try to recognize an operator as a single canonical C3 gate.
    ///
    /// Recognizes:
    /// - T: sectored identity + quarter-turn right rotor
    /// - Controlled-S: sectored identity + right-rotor J on target
    /// - Toffoli: four control sectors with X(target) only on QQ branch
    pub fn get_gate(mv: &Multivector) -> Option<C3Gate> {
        match mv.terms.len() {
            2 => {
                let t0 = &mv.terms[0];
                let t1 = &mv.terms[1];

                if let Some(qubit) = Self::find_t_qubit(t0, t1) {
                    return Some(C3Gate::T { qubit });
                }

                if let Some((control, target)) = Self::find_controlled_s(t0, t1) {
                    return Some(C3Gate::ControlledS { control, target });
                }

                None
            }
            4 => Self::find_toffoli(mv),
            _ => None,
        }
    }

    /// Conservative predicate for C3-shaped operators:
    /// either recognized explicitly, or at least has sector/rotor structure.
    pub fn is_c3_shaped(mv: &Multivector) -> bool {
        if Self::get_gate(mv).is_some() {
            return true;
        }

        !mv.terms.is_empty()
            && mv
                .terms
                .iter()
                .any(|t| t.left.is_some() || t.right.is_some() || t.rotor.is_some())
    }

    fn find_t_qubit(t0: &Term, t1: &Term) -> Option<usize> {
        Self::match_t_branches(t0, t1).or_else(|| Self::match_t_branches(t1, t0))
    }

    fn match_t_branches(p_branch: &Term, q_branch: &Term) -> Option<usize> {
        let p_sector = p_branch.left?;
        let q_sector = q_branch.left?;

        if p_sector.n != q_sector.n || p_sector.bits != 0 {
            return None;
        }

        if q_sector.bits.count_ones() != 1 {
            return None;
        }

        if !p_branch.blade.is_identity() || !q_branch.blade.is_identity() {
            return None;
        }

        if p_branch.rotor.is_some() {
            return None;
        }

        let qubit = q_sector.bits.trailing_zeros() as usize;

        if q_branch.rotor != Some(Rotor::quarter_turn_at(qubit)) {
            return None;
        }

        if (p_branch.coeff - 1.0).abs() >= 1e-9
            || (q_branch.coeff - 1.0).abs() >= 1e-9
        {
            return None;
        }

        Some(qubit)
    }

    fn find_controlled_s(t0: &Term, t1: &Term) -> Option<(usize, usize)> {
        Self::match_controlled_s_branches(t0, t1)
            .or_else(|| Self::match_controlled_s_branches(t1, t0))
    }

    fn match_controlled_s_branches(
        p_branch: &Term,
        q_branch: &Term,
    ) -> Option<(usize, usize)> {
        let p_sector = p_branch.left?;
        let q_sector = q_branch.left?;

        if p_sector.n != q_sector.n || p_sector.bits != 0 {
            return None;
        }

        if q_sector.bits.count_ones() != 1 {
            return None;
        }

        if !p_branch.blade.is_identity() || !q_branch.blade.is_identity() {
            return None;
        }

        if p_branch.rotor.is_some() {
            return None;
        }

        let control = q_sector.bits.trailing_zeros() as usize;
        let target = Self::extract_local_j_target(q_branch.rotor?)?;

        if control == target {
            return None;
        }

        if (p_branch.coeff - 1.0).abs() >= 1e-9
            || (q_branch.coeff - 1.0).abs() >= 1e-9
        {
            return None;
        }

        Some((control, target))
    }

    fn find_toffoli(mv: &Multivector) -> Option<C3Gate> {
        if mv.terms.len() != 4 {
            return None;
        }

        let first_sector = mv.terms[0].left?;
        let n = first_sector.n;

        if n != mv.n {
            return None;
        }

        for t in &mv.terms {
            let s = t.left?;

            if s.n != n || t.right.is_some() || t.rotor.is_some() {
                return None;
            }

            if (t.coeff - 1.0).abs() >= 1e-9 {
                return None;
            }
        }

        let mut zero_sector: Option<&Term> = None;
        let mut one_bit_terms: Vec<&Term> = Vec::new();
        let mut two_bit_term: Option<&Term> = None;

        for t in &mv.terms {
            let bits = t.left?.bits;

            match bits.count_ones() {
                0 => zero_sector = Some(t),
                1 => one_bit_terms.push(t),
                2 => two_bit_term = Some(t),
                _ => return None,
            }
        }

        if one_bit_terms.len() != 2 {
            return None;
        }

        let p00 = zero_sector?;
        let p1 = one_bit_terms[0];
        let p2 = one_bit_terms[1];
        let p11 = two_bit_term?;

        if !p00.blade.is_identity() || !p1.blade.is_identity() || !p2.blade.is_identity() {
            return None;
        }

        let target = Self::extract_single_x_target(p11.blade)?;

        let s1 = p1.left?;
        let s2 = p2.left?;
        let s11 = p11.left?;

        if s1.bits.count_ones() != 1 || s2.bits.count_ones() != 1 || s11.bits.count_ones() != 2 {
            return None;
        }

        let c1 = s1.bits.trailing_zeros() as usize;
        let c2 = s2.bits.trailing_zeros() as usize;

        if c1 == c2 || target == c1 || target == c2 {
            return None;
        }

        let expected_qq = (1u64 << c1) | (1u64 << c2);

        if s11.bits != expected_qq {
            return None;
        }

        let (control1, control2) = if c1 < c2 { (c1, c2) } else { (c2, c1) };

        Some(C3Gate::Toffoli {
            control1,
            control2,
            target,
        })
    }

    fn extract_local_j_target(rotor: Rotor) -> Option<usize> {
        if rotor.sign
            || rotor.q1_mask != 0
            || rotor.q3_mask != 0
            || rotor.q2_mask.count_ones() != 1
        {
            return None;
        }

        Some(rotor.q2_mask.trailing_zeros() as usize)
    }

    fn extract_single_x_target(blade: Blade) -> Option<usize> {
        if blade.sign || blade.z != 0 || blade.x.count_ones() != 1 {
            return None;
        }

        Some(blade.x.trailing_zeros() as usize)
    }
}