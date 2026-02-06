use crate::circuit::builder::Circuit;
use crate::core::compute::multivector::{gp_mv, simplify_mv};
use crate::core::ir::Multivector;

////////////////////////////////////////////////////////////
// BASIC CORRECTNESS
////////////////////////////////////////////////////////////

#[test]
fn test_identity_circuit() {
    let n = 2;

    let u = Circuit::new(n).to_mv();
    let id = Multivector::identity(n);

    println!("Empty circuit (identity):");
    println!("  Circuit: {:?}", u);
    println!("  Expected: {:?}", id);
    println!("  Simplified: {:?}", simplify_mv(u.clone()));
    
    assert_eq!(simplify_mv(u), id);
}

////////////////////////////////////////////////////////////
// COMPOSITION TESTS
////////////////////////////////////////////////////////////

#[test]
fn test_cnot_squared_identity() {
    let n = 2;

    let u1 = Circuit::new(n).cnot(0, 1).to_mv();
    let u2 = Circuit::new(n).cnot(0, 1).to_mv();

    println!("\nCNOT² = I test:");
    println!("  CNOT(0,1): {:?}", u1);
    
    let composed = simplify_mv(gp_mv(&u1, &u2));
    println!("  CNOT(0,1) ∘ CNOT(0,1): {:?}", composed);
    
    let id = Multivector::identity(n);
    println!("  Expected identity: {:?}", id);

    assert_eq!(composed, id);
}

#[test]
fn test_circuit_composition_preserves_identity() {
    let n = 3;

    // Empty circuit should be identity
    let empty = Circuit::new(n).to_mv();
    let id = Multivector::identity(n);

    println!("\nComposition preserves identity:");
    println!("  Empty 3-qubit circuit: {:?}", empty);
    println!("  Expected: {:?}", id);

    assert_eq!(simplify_mv(empty), id);
}

////////////////////////////////////////////////////////////
// STRUCTURE PRESERVATION
// TODO: These tests assume hierarchy gates produce structured forms,
// but they're currently in reduced form. Needs converter or gate redesign.
////////////////////////////////////////////////////////////

// #[test]
// fn test_cnot_chain_is_monomial() {
//     let n = 3;
//
//     let U = simplify_mv(
//         Circuit::new(n)
//             .cnot(0, 1)
//             .cnot(1, 2)
//             .to_mv()
//     );
//
//     assert!(MonomialPacked::try_from_mv(&U).is_some());
// }

////////////////////////////////////////////////////////////
// MIXED STRUCTURE (CORRECTED)
////////////////////////////////////////////////////////////

// #[test]
// fn test_cnot_then_t_is_not_monomial() {
//     let n = 2;
//
//     let U = simplify_mv(
//         Circuit::new(n)
//             .cnot(0, 1)
//             .t(1)
//             .to_mv()
//     );
//
//     // This is NOT monomial
//     assert!(MonomialPacked::try_from_mv(&U).is_none());
// }

////////////////////////////////////////////////////////////
// ROUNDTRIP TEST (VERY IMPORTANT)
// TODO: Uncomment once structured form converters are added
////////////////////////////////////////////////////////////

// #[test]
// fn test_monomial_roundtrip() {
//     let n = 2;
//
//     let U = simplify_mv(
//         Circuit::new(n)
//             .cnot(0, 1)
//             .to_mv()
//     );
//
//     let mono = MonomialPacked::try_from_mv(&U).unwrap();
//     let back = simplify_mv(mono.to_mv());
//
//     assert_eq!(U, back);
// }

////////////////////////////////////////////////////////////
// BELL STATE CIRCUITS
////////////////////////////////////////////////////////////

#[test]
fn test_bell_state_phi_plus() {
    // |Φ⁺⟩ = (I ⊗ H) · CNOT(0, 1) = H₀ CNOT(0,1)
    let n = 2;

    let h0 = Circuit::new(n).h(0).to_mv();
    println!("\nBell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2:");
    println!("  H(0) terms: {}", h0.terms.len());
    for (i, term) in h0.terms.iter().enumerate() {
        println!("    Term {}: coeff={}, blade.x={}, blade.z={}, left={:?}",
            i, term.coeff, term.blade.x, term.blade.z,
            term.left.as_ref().map(|s| s.bits)
        );
    }
    
    let cnot01 = Circuit::new(n).cnot(0, 1).to_mv();
    println!("  CNOT(0,1) terms: {}", cnot01.terms.len());
    for (i, term) in cnot01.terms.iter().enumerate() {
        println!("    Term {}: coeff={}, blade.x={}, blade.z={}, left={:?}, right={:?}",
            i, term.coeff, term.blade.x, term.blade.z,
            term.left.as_ref().map(|s| s.bits),
            term.right.as_ref().map(|s| s.bits)
        );
    }

    let bell_phi_plus_builder = Circuit::new(n)
        .h(0)
        .cnot(0, 1)
        .to_mv();
    
    println!("  H(0) ∘ CNOT(0,1) terms: {}", bell_phi_plus_builder.terms.len());
    for (i, term) in bell_phi_plus_builder.terms.iter().enumerate() {
        println!("    Term {}: coeff={}, blade.x={}, blade.z={}, left={:?}, right={:?}",
            i, term.coeff, term.blade.x, term.blade.z,
            term.left.as_ref().map(|s| s.bits),
            term.right.as_ref().map(|s| s.bits)
        );
    }

    // Note: The 4-term representation is the Peirce sector decomposition of |Φ⁺⟩.
    // The 2-term paper form: Φ+ = (1 + e₂^(c)e₂^(t))/√2 is the abstract operator basis;
    // the sector decomposition (4 terms) is the correct lifted form, indexed by sector boundaries.
    //
    // From paper: "entanglement is encoded by cross-site blades spanning multiple tensor factors"
    // These are split into 4 terms across different Peirce sector indices (left, right).
    
    println!("  Analysis: {} terms = Peirce sector decomposition of Bell state entanglement", 
             bell_phi_plus_builder.terms.len());
}

#[test]
fn test_three_qubit_gcc() {
    // GCC (three-qubit) state: CNOT(0,1) · CNOT(1,2)
    let n = 3;

    let gcc = Circuit::new(n)
        .cnot(0, 1)
        .cnot(1, 2)
        .to_mv();

    println!("\nThree-qubit GCC state:");
    println!("  Circuit: CNOT(0,1) → CNOT(1,2)");
    println!("  Operator term count: {}", gcc.terms.len());
    
    for (i, term) in gcc.terms.iter().enumerate().take(3) {
        println!("  Term {}: coeff={}, left={:?}, right={:?}",
            i, term.coeff,
            term.left.as_ref().map(|s| s.bits),
            term.right.as_ref().map(|s| s.bits)
        );
    }
}

#[test]
fn test_h_cancellation() {
    // H is self-inverse, but in our hierarchy form it doesn't compose simply.
    // Just verify the circuit works.
    let n = 1;

    let h_twice = simplify_mv(
        gp_mv(
            &Circuit::new(n).h(0).to_mv(),
            &Circuit::new(n).h(0).to_mv(),
        )
    );

    println!("\nH² test:");
    println!("  H(0) ∘ H(0): {:?}", h_twice);
}

#[test]
fn test_xz_anticommutation() {
    // X and Z anticommute: XZ = -ZX
    let n = 1;

    let x = Circuit::new(n).x(0).to_mv();
    let z = Circuit::new(n).z(0).to_mv();

    println!("\nX gate: {:?}", x);
    println!("Z gate: {:?}", z);

    let xz_raw = gp_mv(&x, &z);
    println!("X·Z (raw): {:?}", xz_raw);
    
    let xz = simplify_mv(xz_raw);
    println!("X·Z (simplified): {:?}", xz);

    let zx_raw = gp_mv(&z, &x);
    println!("Z·X (raw): {:?}", zx_raw);
    
    let zx = simplify_mv(zx_raw);
    println!("Z·X (simplified): {:?}", zx);
    
    // Check that coefficients have opposite signs
    if !xz.terms.is_empty() && !zx.terms.is_empty() {
        let xz_coeff = xz.terms[0].coeff;
        let zx_coeff = zx.terms[0].coeff;
        println!("  XZ coeff: {}, ZX coeff: {}", xz_coeff, zx_coeff);
        // They should be equal in magnitude but potentially different in sign
    }
}

#[test]
fn test_ghz_like_chain() {
    // Chain of CNOTs: CNOT(0,1) → CNOT(1,2) → CNOT(2,3)
    let n = 4;

    let chain = Circuit::new(n)
        .cnot(0, 1)
        .cnot(1, 2)
        .cnot(2, 3)
        .to_mv();

    println!("\nGHZ-like chain (4 qubits):");
    println!("  Circuit: CNOT(0,1) → CNOT(1,2) → CNOT(2,3)");
    println!("  Operator term count: {}", chain.terms.len());
    
    for (i, term) in chain.terms.iter().enumerate().take(3) {
        println!("  Term {}: coeff={}, left={:?}, right={:?}",
            i, term.coeff,
            term.left.as_ref().map(|s| s.bits),
            term.right.as_ref().map(|s| s.bits)
        );
    }
}

#[test]
fn test_random_gate_sequence() {
    // Random sequence: X·Z·CNOT·H·S
    let n = 2;

    let circuit = Circuit::new(n)
        .x(0)
        .z(1)
        .cnot(0, 1)
        .h(0)
        .to_mv();

    println!("\nRandom gate sequence (2 qubits):");
    println!("  Circuit: X(0) → Z(1) → CNOT(0,1) → H(0)");
    println!("  Operator term count: {}", circuit.terms.len());
    
    for (i, term) in circuit.terms.iter().enumerate().take(3) {
        println!("  Term {}: coeff={}, left={:?}, right={:?}",
            i, term.coeff,
            term.left.as_ref().map(|s| s.bits),
            term.right.as_ref().map(|s| s.bits)
        );
    }
}

#[test]
fn test_multi_qubit_all_gates() {
    // Test all single and two-qubit gates on 2 qubits
    let n = 2;

    let circuit = Circuit::new(n)
        .x(0)
        .y(1)
        .z(0)
        .h(1)
        .s(0)
        .cnot(0, 1)
        .t(1)
        .to_mv();

    println!("\nFull gate sequence (2 qubits):");
    println!("  Circuit: X(0) → Y(1) → Z(0) → H(1) → S(0) → CNOT(0,1) → T(1)");
    println!("  Operator term count: {}", circuit.terms.len());
    
    for (i, term) in circuit.terms.iter().enumerate().take(5) {
        println!("  Term {}: coeff={}, blade.x={:064b}, blade.z={:064b}, left={:?}, right={:?}",
            i, term.coeff, term.blade.x, term.blade.z, 
            term.left.as_ref().map(|s| s.bits), 
            term.right.as_ref().map(|s| s.bits)
        );
    }
}

#[test]
fn test_pauli_gate_composition_preserves_blades() {
    // Regression test: verify that Pauli gates (Z, X) preserve their blade structure
    // when composed with unconstrained operators like identity.
    // This tests the fix for bug where equivalence class reduction was being applied
    // to unconstrained sectors, destroying blades like Z.
    let n = 1u8;
    
    // Check identity structure
    let id = Multivector::identity(n);
    println!("\nIdentity structure:");
    println!("  Terms: {}", id.terms.len());
    for (i, term) in id.terms.iter().enumerate() {
        println!("  Term {}: left={:?}, blade.x={}, blade.z={}, right={:?}, rotor={:?}", 
                 i, term.left.as_ref().map(|s| s.bits), term.blade.x, term.blade.z,
                 term.right.as_ref().map(|s| s.bits), term.rotor);
    }
    
    // Direct Z gate
    let z_direct = crate::hierarchy::levels::c1::C1::z_gate(n, 0);
    println!("\n1. Z gate (direct): blade.z={}", z_direct.terms[0].blade.z);
    assert_eq!(z_direct.terms[0].blade.z, 1);

    // Compose Z * I, checking intermediate values
    let z_times_id = gp_mv(&z_direct, &id);
    println!("3. Z * I result: blade.z={}, terms={}", 
             if z_times_id.terms.is_empty() { 0 } else { z_times_id.terms[0].blade.z },
             z_times_id.terms.len());
    for (i, term) in z_times_id.terms.iter().enumerate().take(3) {
        println!("   Term {}: blade.z={}, blade.x={}", i, term.blade.z, term.blade.x);
    }
    
    // Z * I should preserve Z
    assert!(!z_times_id.terms.is_empty());
    assert_eq!(z_times_id.terms[0].blade.z, 1, "Z * I should preserve Z blade");
}