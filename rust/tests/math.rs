//! Analytic sanity-check tests for `math.rs`.
//!
//! These pin the closed-form identities that the Python jax_core port
//! must satisfy. Bit-identity vs the Python reference is a follow-on
//! session: it generates fixture grids by running jax_core and checks
//! the Rust output matches within libm tolerance.

use mpa_scale_solver::math::{
    Activation, MlpLayer, banach_state, caputo_flow, inv_2x2,
    laplace_covariance_from_jacobian, learned_field_substrate, lookup_squared_distance,
    tangent_flow_canonical_inverse, tangent_flow_inversion_residual, tangent_flow_substrate,
};

const TOL: f64 = 1e-12;

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs()))
}

// ---------------------------------------------------------------------------
// tangent_flow_substrate
// ---------------------------------------------------------------------------

#[test]
fn tangent_flow_substrate_identity_at_ref_tau() {
    // tau_obs == tau_obs_ref → ratio=1, log_ratio=0, pow_ratio=1 → unchanged.
    let (chit, gamma) = tangent_flow_substrate(1.5, 2.5, 0.3, 0.7, 4.0, 4.0);
    assert!(approx_eq(chit, 1.5, TOL));
    assert!(approx_eq(gamma, 2.5, TOL));
}

#[test]
fn tangent_flow_substrate_identity_at_degenerate_tau() {
    // tau_obs <= 0 falls through to identity branch.
    let (chit, gamma) = tangent_flow_substrate(1.5, 2.5, 0.3, 0.7, 0.0, 4.0);
    assert_eq!(chit, 1.5);
    assert_eq!(gamma, 2.5);
    let (chit, gamma) = tangent_flow_substrate(1.5, 2.5, 0.3, 0.7, 4.0, 0.0);
    assert_eq!(chit, 1.5);
    assert_eq!(gamma, 2.5);
}

#[test]
fn tangent_flow_substrate_closed_form() {
    // tau/tau_ref = 2; log_ratio = ln(2); pow_ratio = 2^delta_gamma.
    let (chit, gamma) = tangent_flow_substrate(1.0, 2.0, 0.5, 0.3, 2.0, 1.0);
    assert!(approx_eq(chit, 1.0 + 0.5 * 2f64.ln(), TOL));
    assert!(approx_eq(gamma, 2.0 * 2f64.powf(0.3), TOL));
}

// ---------------------------------------------------------------------------
// tangent_flow_canonical_inverse — round-trip identity
// ---------------------------------------------------------------------------

#[test]
fn tangent_flow_round_trip_identity() {
    // Forward then inverse should recover the canonical pair exactly
    // (closed-form analytic inverse — no float-iteration drift).
    let (chit_0, gamma_0) = (1.234, 2.345);
    let (delta_chit, delta_gamma) = (0.4, 0.6);
    let (tau_obs, tau_ref) = (3.7, 1.0);

    let (s_chit, s_gamma) =
        tangent_flow_substrate(chit_0, gamma_0, delta_chit, delta_gamma, tau_obs, tau_ref);
    let (r_chit, r_gamma) = tangent_flow_canonical_inverse(
        s_chit,
        s_gamma,
        delta_chit,
        delta_gamma,
        tau_obs,
        tau_ref,
    );
    assert!(approx_eq(r_chit, chit_0, TOL));
    assert!(approx_eq(r_gamma, gamma_0, TOL));
}

#[test]
fn tangent_flow_inversion_residual_zero_at_map() {
    // Plug the MAP candidate (= the canonical that generated the target)
    // back into the residual: it should be exactly zero.
    let (chit_0, gamma_0) = (1.234, 2.345);
    let (delta_chit, delta_gamma) = (0.4, 0.6);
    let (tau_obs, tau_ref) = (3.7, 1.0);

    let (s_chit, s_gamma) =
        tangent_flow_substrate(chit_0, gamma_0, delta_chit, delta_gamma, tau_obs, tau_ref);
    let residual = tangent_flow_inversion_residual(
        chit_0,
        gamma_0,
        s_chit,
        s_gamma,
        delta_chit,
        delta_gamma,
        tau_obs,
        tau_ref,
    );
    assert_eq!(residual, 0.0);
}

// ---------------------------------------------------------------------------
// banach_state
// ---------------------------------------------------------------------------

#[test]
fn banach_state_at_nu_zero_is_initial() {
    let (chit, gamma) = banach_state(1.5, 2.5, 0.1, 0.2, 0.0);
    assert!(approx_eq(chit, 1.5, TOL));
    assert!(approx_eq(gamma, 2.5, TOL));
}

#[test]
fn banach_state_exponential_decay_rate() {
    // chit(nu) / chit(0) = exp(-lambda * nu)
    let (chit, _) = banach_state(1.0, 1.0, 0.3, 0.4, 2.0);
    assert!(approx_eq(chit, (-0.3 * 2.0_f64).exp(), TOL));
}

// ---------------------------------------------------------------------------
// caputo_flow
// ---------------------------------------------------------------------------

#[test]
fn caputo_flow_single_term_matches_banach() {
    // prony = [(1.0, 1.0)] is the v2.4 Markovian degenerate case:
    // caputo_flow == banach_state, BLOCK_IN §v2.4 acceptance.
    let amps = [1.0];
    let decays = [1.0];
    let (caputo_chit, caputo_gamma) =
        caputo_flow(1.0, 2.0, 0.3, 0.4, 1.5, &amps, &decays);
    let (banach_chit, banach_gamma) = banach_state(1.0, 2.0, 0.3, 0.4, 1.5);
    assert_eq!(caputo_chit, banach_chit);
    assert_eq!(caputo_gamma, banach_gamma);
}

#[test]
fn caputo_flow_decomposition_sums_correctly() {
    // Two-term prony: result should equal the sum of single-term contributions.
    let amps = [0.4, 0.6];
    let decays = [1.0, 2.5];
    let (chit, _) = caputo_flow(1.0, 1.0, 0.3, 0.4, 1.5, &amps, &decays);
    let expected = 0.4 * (-1.0_f64 * 0.3 * 1.5).exp()
        + 0.6 * (-2.5_f64 * 0.3 * 1.5).exp();
    assert!(approx_eq(chit, expected, TOL));
}

// ---------------------------------------------------------------------------
// lookup_squared_distance
// ---------------------------------------------------------------------------

#[test]
fn lookup_squared_distance_no_tau_term() {
    // Two rules, neither carries tau — the third term collapses to 0.
    let d = lookup_squared_distance(
        0.0,
        0.0,
        &[3.0, 0.0],
        &[4.0, 0.0],
        &[1.0, 1.0],
        &[false, false],
        1.0,
        1.0,
    );
    assert!(approx_eq(d[0], 9.0 + 16.0, TOL));
    assert_eq!(d[1], 0.0);
}

#[test]
fn lookup_squared_distance_with_tau_term() {
    // Single rule carries tau: extra term = weight * (log(rule_tau) - log(query_tau))^2.
    let d = lookup_squared_distance(
        0.0,
        0.0,
        &[0.0],
        &[0.0],
        &[2.0],
        &[true],
        1.0,
        3.0,
    );
    let expected_tau_term = 3.0 * (2f64.ln() - 1f64.ln()).powi(2);
    assert!(approx_eq(d[0], expected_tau_term, TOL));
}

// ---------------------------------------------------------------------------
// 2x2 inverse + Laplace covariance
// ---------------------------------------------------------------------------

#[test]
fn inv_2x2_times_self_is_identity() {
    let m = [[2.0, 1.0], [1.0, 3.0]];
    let inv = inv_2x2(&m).expect("non-singular");
    let prod: [[f64; 2]; 2] = [
        [
            m[0][0] * inv[0][0] + m[0][1] * inv[1][0],
            m[0][0] * inv[0][1] + m[0][1] * inv[1][1],
        ],
        [
            m[1][0] * inv[0][0] + m[1][1] * inv[1][0],
            m[1][0] * inv[0][1] + m[1][1] * inv[1][1],
        ],
    ];
    assert!(approx_eq(prod[0][0], 1.0, TOL));
    assert!(approx_eq(prod[1][1], 1.0, TOL));
    assert!(prod[0][1].abs() < TOL);
    assert!(prod[1][0].abs() < TOL);
}

#[test]
fn inv_2x2_singular_returns_none() {
    assert!(inv_2x2(&[[1.0, 2.0], [2.0, 4.0]]).is_none());
}

#[test]
fn laplace_covariance_isotropic_jacobian() {
    // J = identity columns (per-obs 2-vectors (1,0) and (0,1))
    // → J^T J = I → covariance = sigma^2 * I.
    let j: Vec<[f64; 2]> = vec![[1.0, 0.0], [0.0, 1.0]];
    let cov = laplace_covariance_from_jacobian(&j, 0.25).expect("non-singular");
    assert!(approx_eq(cov[0][0], 0.25, TOL));
    assert!(approx_eq(cov[1][1], 0.25, TOL));
    assert!(cov[0][1].abs() < TOL);
    assert!(cov[1][0].abs() < TOL);
}

// ---------------------------------------------------------------------------
// MLP forward
// ---------------------------------------------------------------------------

#[test]
fn mlp_forward_single_linear_layer_is_affine() {
    // Single layer = linear (the output layer is always linear).
    let layer = MlpLayer {
        w: vec![vec![2.0, 0.0], vec![0.0, 3.0]],
        b: vec![0.5, -0.5],
    };
    let y = mpa_scale_solver::math::mlp_forward(&[1.0, 1.0], &[layer], Activation::Tanh);
    assert!(approx_eq(y[0], 2.0 * 1.0 + 0.5, TOL));
    assert!(approx_eq(y[1], 3.0 * 1.0 - 0.5, TOL));
}

#[test]
fn mlp_forward_two_layer_tanh() {
    // Hidden tanh, linear output.
    let hidden = MlpLayer {
        w: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        b: vec![0.0, 0.0],
    };
    let out = MlpLayer {
        w: vec![vec![1.0, 1.0]],
        b: vec![0.0],
    };
    let y = mpa_scale_solver::math::mlp_forward(&[0.5, 0.5], &[hidden, out], Activation::Tanh);
    let expected = 0.5_f64.tanh() + 0.5_f64.tanh();
    assert!(approx_eq(y[0], expected, TOL));
}

// ---------------------------------------------------------------------------
// learned_field_substrate
// ---------------------------------------------------------------------------

#[test]
fn learned_field_log_ratio_clamped_at_degenerate_tau() {
    // Identity-passthrough layer on (chit, gamma, log_ratio).
    // At tau_obs <= 0, log_ratio is clamped to 0 — the output should NOT
    // include any log-ratio contribution.
    let identity_layer = MlpLayer {
        w: vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 1.0]],
        b: vec![0.0, 0.0],
    };
    let (chit, gamma) =
        learned_field_substrate(1.5, 2.5, 0.0, 4.0, &[identity_layer], Activation::Tanh);
    assert!(approx_eq(chit, 1.5, TOL));
    // gamma channel had log_ratio added; clamped log_ratio = 0 → unchanged.
    assert!(approx_eq(gamma, 2.5, TOL));
}
