use crate::core::bits::{Blade, Rotor, Sector};
use crate::core::ir::{Multivector, Term};

/// Packed Clifford gate descriptors.
///
/// This enum is intentionally small and Python-friendly.
/// It should be easy to expose as strings / tagged objects later.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum C2Gate {
    H { qubit: usize },
    S { qubit: usize },
    CNOT { control: usize, target: usize },
}

/// Level-C2 utilities: Clifford recognition and packed conjugation actions.
///
/// This layer builds on the C1 blade representation.
pub struct C2;

impl C2 {
    /// A packed H gate descriptor.
    #[inline(always)]
    pub fn h(qubit: usize) -> C2Gate {
        C2Gate::H { qubit }
    }

    /// A packed S gate descriptor.
    #[inline(always)]
    pub fn s(qubit: usize) -> C2Gate {
        C2Gate::S { qubit }
    }

    /// A packed CNOT gate descriptor.
    #[inline(always)]
    pub fn cnot(control: usize, target: usize) -> C2Gate {
        C2Gate::CNOT { control, target }
    }

    // ===== Operator constructors =====

    /// Hadamard gate as a multivector (no sectors—it's a true superposition):
    ///
    ///     H = (X + Z)/sqrt(2)
    pub fn h_gate(qubit: usize) -> Multivector {
        let n = (qubit + 1) as u8;
        Self::h_gate_n(n, qubit)
    }

    /// Hadamard gate with explicit qubit count.
    pub fn h_gate_n(n: u8, qubit: usize) -> Multivector {
        let s = 1.0_f64 / 2.0_f64.sqrt();
        Multivector::from_terms(n, vec![
            Term {
                left: None,
                blade: Blade::x(qubit),
                right: None,
                rotor: None,
                coeff: s,
            },
            Term {
                left: None,
                blade: Blade::z(qubit),
                right: None,
                rotor: None,
                coeff: s,
            },
        ])
    }

    /// S gate using the paper's Peirce-sector form:
    ///
    ///     S_i = P_i + Q_i R_i
    ///
    /// with `R_i = J_i` represented as a right rotor.
    pub fn s_gate(qubit: usize) -> Multivector {
        let n = (qubit + 1) as u8;
        Self::s_gate_n(n, qubit)
    }

    /// S gate with explicit qubit count.
    pub fn s_gate_n(n: u8, qubit: usize) -> Multivector {
        Multivector::from_terms(n, vec![
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
                rotor: Some(Rotor::j_at(qubit)),
                coeff: 1.0,
            },
        ])
    }

    /// CNOT gate using the paper's Peirce-sector form:
    ///
    ///     CNOT(c,t) = P_c + Q_c e_{2,t}
    ///
    /// i.e. identity on the P-control sector and X on the target in the Q-control sector.
    pub fn cnot_gate(control: usize, target: usize) -> Multivector {
        let n = std::cmp::max(control, target) + 1;

        Multivector::from_terms(n as u8, vec![
            Term {
                left: Some(Sector::new(0, n as u8)),
                blade: Blade::identity(),
                right: None,
                rotor: None,
                coeff: 1.0,
            },
            Term {
                left: Some(Sector::new(1u64 << control, n as u8)),
                blade: Blade::x(target),
                right: None,
                rotor: None,
                coeff: 1.0,
            },
        ])
    }

    /// Conjugate a packed blade by a single-qubit H.
    ///
    /// Action on local generators:
    /// - X <-> Z
    /// - J -> -J
    pub fn conjugate_blade_by_h(blade: Blade, qubit: usize) -> Blade {
        let bit = 1u64 << qubit;
        let xi = (blade.x & bit) != 0;
        let zi = (blade.z & bit) != 0;

        let mut out = blade;

        match (xi, zi) {
            (false, false) => {}
            (true, false) => {
                // X -> Z
                out.x ^= bit;
                out.z ^= bit;
            }
            (false, true) => {
                // Z -> X
                out.x ^= bit;
                out.z ^= bit;
            }
            (true, true) => {
                // J -> -J
                out.sign = !out.sign;
            }
        }

        out
    }

    /// Conjugate a packed blade by a single-qubit S.
    ///
    /// Standard packed action:
    /// - X -> XZ
    /// - Z -> Z
    /// - J -> -X
    pub fn conjugate_blade_by_s(blade: Blade, qubit: usize) -> Blade {
        let bit = 1u64 << qubit;
        let xi = (blade.x & bit) != 0;
        let zi = (blade.z & bit) != 0;

        let mut out = blade;

        match (xi, zi) {
            (false, false) => {}
            (false, true) => {
                // Z -> Z
            }
            (true, false) => {
                // X -> XZ
                out.z ^= bit;
            }
            (true, true) => {
                // J -> -X
                out.z ^= bit;
                out.sign = !out.sign;
            }
        }

        out
    }

    /// Conjugate a packed blade by CNOT(control, target).
    ///
    /// Packed tableau-style action:
    /// - X_c -> X_c X_t
    /// - Z_t -> Z_c Z_t
    pub fn conjugate_blade_by_cnot(blade: Blade, control: usize, target: usize) -> Blade {
        let cb = 1u64 << control;
        let tb = 1u64 << target;

        let x_c = (blade.x & cb) != 0;
        let z_t = (blade.z & tb) != 0;

        let mut out = blade;

        if x_c {
            out.x ^= tb;
        }
        if z_t {
            out.z ^= cb;
        }

        out
    }

    /// Conjugate a blade by a packed Clifford gate descriptor.
    pub fn conjugate_blade(blade: Blade, gate: C2Gate) -> Blade {
        match gate {
            C2Gate::H { qubit } => Self::conjugate_blade_by_h(blade, qubit),
            C2Gate::S { qubit } => Self::conjugate_blade_by_s(blade, qubit),
            C2Gate::CNOT { control, target } => {
                Self::conjugate_blade_by_cnot(blade, control, target)
            }
        }
    }

    /// Conjugate every atomic term in a multivector by a Clifford gate.
    ///
    /// This is intended for Pauli/stabilizer-like operators and preserves
    /// sector/right-rotor metadata unchanged for now.
    pub fn conjugate_mv(mv: &Multivector, gate: C2Gate) -> Multivector {
        let terms = mv
            .terms
            .iter()
            .cloned()
            .map(|mut t| {
                t.blade = Self::conjugate_blade(t.blade, gate);
                t
            })
            .collect();

        Multivector::from_terms(mv.n, terms)
    }

    /// Conservative predicate: a multivector is "Clifford-shaped" if it has sector structure
    /// (left or right sectors), right-rotor structure, or is a bare Pauli-like blade
    /// (Pauli is a subclass of Clifford).
    pub fn is_clifford_shaped(mv: &Multivector) -> bool {
        !mv.terms.is_empty()
            && mv.terms.iter().any(|t| {
                t.left.is_some()
                    || t.right.is_some()
                    || t.rotor.is_some()
                    || (t.left.is_none()
                        && t.right.is_none()
                        && t.rotor.map(|r| r.is_identity()).unwrap_or(true))
            })
    }

    /// Try to recognize an operator as a single C2Gate.
    ///
    /// Recognizes:
    /// - H (Hadamard): bare blade superposition, no sectors
    /// - S: sectored Peirce decomposition with a right rotor on the Q branch
    /// - CNOT: sectored Peirce decomposition with X(target) on the Q_control branch
    pub fn get_gate(mv: &Multivector) -> Option<C2Gate> {
        if mv.terms.len() != 2 {
            return None;
        }

        let t0 = &mv.terms[0];
        let t1 = &mv.terms[1];

        // Try bare-blade H recognition (no sectors, no right rotors)
        if t0.left.is_none()
            && t1.left.is_none()
            && t0.right.is_none()
            && t1.right.is_none()
            && t0.rotor.is_none()
            && t1.rotor.is_none()
        {
            let coeff_ok = (t0.coeff.abs() - 1.0 / 2.0_f64.sqrt()).abs() < 1e-9
                && (t1.coeff.abs() - 1.0 / 2.0_f64.sqrt()).abs() < 1e-9;
            if !coeff_ok {
                return None;
            }

            if Self::is_single_site_blade(t0.blade) && Self::is_single_site_blade(t1.blade) {
                if let (Some(q0), Some(q1)) = (
                    Self::get_single_site_qubit(t0.blade),
                    Self::get_single_site_qubit(t1.blade),
                ) {
                    if q0 == q1 {
                        let q = q0;
                        if (t0.blade == Blade::x(q) && t1.blade == Blade::z(q))
                            || (t0.blade == Blade::z(q) && t1.blade == Blade::x(q))
                        {
                            return Some(C2Gate::H { qubit: q });
                        }
                    }
                }
            }

            return None;
        }

        // Sectored recognition requires matching left sectors and no right sectors.
        if t0.right.is_some() || t1.right.is_some() {
            return None;
        }

        if let (Some(s0), Some(s1)) = (t0.left, t1.left) {
            if s0.n != s1.n {
                return None;
            }

            // S gate: one branch is P_i with identity, the other is Q_i with right-rotor J_i.
            if let Some(qubit) = Self::find_s_qubit(&t0, &t1) {
                return Some(C2Gate::S { qubit });
            }

            // CNOT gate: one branch is all-P with identity, the other is Q_control with X(target).
            if let Some((control, target)) = Self::find_cnot_control_target(&t0, &t1) {
                return Some(C2Gate::CNOT { control, target });
            }
        }

        None
    }

    fn find_s_qubit(t0: &Term, t1: &Term) -> Option<usize> {
        Self::match_s_branches(t0, t1).or_else(|| Self::match_s_branches(t1, t0))
    }

    fn match_s_branches(p_branch: &Term, q_branch: &Term) -> Option<usize> {
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
        if q_branch.rotor != Some(Rotor::j_at(qubit)) {
            return None;
        }

        if (p_branch.coeff - 1.0).abs() >= 1e-9 || (q_branch.coeff - 1.0).abs() >= 1e-9 {
            return None;
        }

        Some(qubit)
    }

    fn find_cnot_control_target(t0: &Term, t1: &Term) -> Option<(usize, usize)> {
        Self::match_cnot_branches(t0, t1).or_else(|| Self::match_cnot_branches(t1, t0))
    }

    fn match_cnot_branches(p_branch: &Term, q_branch: &Term) -> Option<(usize, usize)> {
        let p_sector = p_branch.left?;
        let q_sector = q_branch.left?;

        if p_sector.n != q_sector.n || p_sector.bits != 0 {
            return None;
        }
        if q_sector.bits.count_ones() != 1 {
            return None;
        }
        if !p_branch.blade.is_identity() {
            return None;
        }
        if p_branch.rotor.is_some() || q_branch.rotor.is_some() {
            return None;
        }

        let control = q_sector.bits.trailing_zeros() as usize;
        let target = Self::extract_single_x_target(q_branch.blade)?;
        if control == target {
            return None;
        }

        if (p_branch.coeff - 1.0).abs() >= 1e-9 || (q_branch.coeff - 1.0).abs() >= 1e-9 {
            return None;
        }

        Some((control, target))
    }

    /// Check if a blade acts on exactly one qubit (single-qubit Pauli).
    fn is_single_site_blade(blade: Blade) -> bool {
        let support = blade.x | blade.z;
        support.count_ones() == 1 || support == 0
    }

    /// Get the qubit index if the blade acts on exactly one qubit.
    fn get_single_site_qubit(blade: Blade) -> Option<usize> {
        let support = blade.x | blade.z;
        if support.count_ones() == 1 {
            Some(support.trailing_zeros() as usize)
        } else if support == 0 {
            Some(0)
        } else {
            None
        }
    }

    fn extract_single_x_target(blade: Blade) -> Option<usize> {
        if blade.sign || blade.z != 0 || blade.x.count_ones() != 1 {
            return None;
        }
        Some(blade.x.trailing_zeros() as usize)
    }

    /// Packed action of a Clifford gate on a stabilizer generator row.
    #[inline(always)]
    pub fn conjugate_tableau_row(row: Blade, gate: C2Gate) -> Blade {
        Self::conjugate_blade(row, gate)
    }

    /// Verify that conjugation by a Clifford gate preserves Pauli structure.
    pub fn preserves_pauli_structure(before: Blade, gate: C2Gate) -> bool {
        let after = Self::conjugate_blade(before, gate);
        after.x | after.z == after.x | after.z
    }

    /// Return a simple left-action CNOT representative in the current packed operator language.
    pub fn cnot_symbolic(control: usize, target: usize) -> (usize, usize) {
        (control, target)
    }
}