//! Math primitives — port of `mpa_scale_solver/jax_core.py`.
//!
//! The Python source is the canonical reference. Bit-identity (or
//! within a documented libm tolerance) per the v6 BLOCK_IN acceptance.
//! All scalars are `f64` to match `jax_enable_x64`.
//!
//! `jnp.where`-guarded branches in the Python (needed for JAX
//! tracing) collapse to plain `if`/`else` here — same math, no tracer.

// ---------------------------------------------------------------------------
// Tangent-flow forward map  (mirror of jax_core.tangent_flow_substrate)
// ---------------------------------------------------------------------------

/// Forward-map canonical to substrate via the ScalingRule closed form.
///
///     scaled_chit  = chit + delta_chit  * log(tau_obs / tau_obs_ref)
///     scaled_gamma = gamma_AB * (tau_obs / tau_obs_ref) ** delta_gamma
///
/// For `tau_obs <= 0` or `tau_obs_ref <= 0` returns the canonical values
/// unmodified (identity at degenerate tau_obs — matches the v0 branch).
pub fn tangent_flow_substrate(
    chit: f64,
    gamma_ab: f64,
    delta_chit: f64,
    delta_gamma: f64,
    tau_obs: f64,
    tau_obs_ref: f64,
) -> (f64, f64) {
    if tau_obs > 0.0 && tau_obs_ref > 0.0 {
        let ratio = tau_obs / tau_obs_ref;
        let log_ratio = ratio.ln();
        let pow_ratio = ratio.powf(delta_gamma);
        (chit + delta_chit * log_ratio, gamma_ab * pow_ratio)
    } else {
        (chit, gamma_ab)
    }
}

// ---------------------------------------------------------------------------
// Banach analytical canonical state  (mirror of jax_core.banach_state)
// ---------------------------------------------------------------------------

/// Canonical state at depth nu under Banach exponential decay.
///
///     chit(nu)     = chit_0     * exp(-lambda_chit  * nu)
///     gamma_AB(nu) = gamma_AB_0 * exp(-lambda_gamma * nu)
pub fn banach_state(
    chit_0: f64,
    gamma_ab_0: f64,
    lambda_chit: f64,
    lambda_gamma: f64,
    nu: f64,
) -> (f64, f64) {
    (
        chit_0 * (-lambda_chit * nu).exp(),
        gamma_ab_0 * (-lambda_gamma * nu).exp(),
    )
}

// ---------------------------------------------------------------------------
// Generic tangent flow  (mirror of jax_core.tangent_flow_canonical)
// ---------------------------------------------------------------------------

/// Continuous-form canonical flow under a ScalingRule treating nu as tau_obs.
/// Delegates to `tangent_flow_substrate` — the math is identical, just
/// with `nu` in the tau_obs slot.
pub fn tangent_flow_canonical(
    chit_0: f64,
    gamma_ab_0: f64,
    delta_chit: f64,
    delta_gamma: f64,
    nu: f64,
    tau_obs_ref: f64,
) -> (f64, f64) {
    tangent_flow_substrate(chit_0, gamma_ab_0, delta_chit, delta_gamma, nu, tau_obs_ref)
}

// ---------------------------------------------------------------------------
// Lookup-table squared distance  (mirror of jax_core.lookup_squared_distance)
// ---------------------------------------------------------------------------

/// Per-rule squared L2 distance with the log-tau term for tau-carrying rules.
///
/// Returns the per-rule `d2` array; `argmin(d2)` selects the nearest rule.
/// All field arrays must have the same length.
pub fn lookup_squared_distance(
    query_chit: f64,
    query_gamma: f64,
    field_chits: &[f64],
    field_gammas: &[f64],
    field_taus: &[f64],
    has_tau: &[bool],
    tau_obs: f64,
    tau_obs_weight: f64,
) -> Vec<f64> {
    let n = field_chits.len();
    debug_assert_eq!(field_gammas.len(), n);
    debug_assert_eq!(field_taus.len(), n);
    debug_assert_eq!(has_tau.len(), n);

    let log_tau_q = if tau_obs > 0.0 { tau_obs.ln() } else { 0.0 };

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let d_chit = field_chits[i] - query_chit;
        let d_gamma = field_gammas[i] - query_gamma;
        let d2 = d_chit * d_chit + d_gamma * d_gamma;
        let d_tau = if has_tau[i] {
            field_taus[i].ln() - log_tau_q
        } else {
            0.0
        };
        out.push(d2 + tau_obs_weight * d_tau * d_tau);
    }
    out
}

// ---------------------------------------------------------------------------
// Analytical inverse of the tangent-flow forward map
// (mirror of jax_core.tangent_flow_canonical_inverse)
// ---------------------------------------------------------------------------

/// Exact closed-form inverse of `tangent_flow_substrate`.
///
///     canonical_chit     = substrate_chit  - delta_chit * log(tau / tau_ref)
///     canonical_gamma_AB = substrate_gamma / (tau / tau_ref) ** delta_gamma
///
/// Identity at degenerate `tau_obs <= 0` or `tau_obs_ref <= 0`.
pub fn tangent_flow_canonical_inverse(
    substrate_chit: f64,
    substrate_gamma_ab: f64,
    delta_chit: f64,
    delta_gamma: f64,
    tau_obs: f64,
    tau_obs_ref: f64,
) -> (f64, f64) {
    if tau_obs > 0.0 && tau_obs_ref > 0.0 {
        let ratio = tau_obs / tau_obs_ref;
        let log_ratio = ratio.ln();
        let pow_ratio = ratio.powf(delta_gamma);
        (
            substrate_chit - delta_chit * log_ratio,
            substrate_gamma_ab / pow_ratio,
        )
    } else {
        (substrate_chit, substrate_gamma_ab)
    }
}

// ---------------------------------------------------------------------------
// Jacobian of the tangent-flow forward map
// (mirror of jax_ops.tangent_flow_forward_jacobian)
// ---------------------------------------------------------------------------

/// 2x2 Jacobian of `tangent_flow_substrate` w.r.t. `(chit, gamma_AB)` at
/// the given canonical state.
///
/// The forward map is component-wise: `scaled_chit` depends only on
/// `chit` (additively, via `delta_chit * log(ratio)`); `scaled_gamma`
/// depends only on `gamma_AB` (multiplicatively, via `ratio^delta_gamma`).
/// The Jacobian is therefore diagonal:
///
///     [[ 1, 0 ],
///      [ 0, (tau_obs/tau_obs_ref)^delta_gamma ]]
///
/// At degenerate `tau_obs <= 0` or `tau_obs_ref <= 0` the forward map
/// is identity, so the Jacobian is identity too.
///
/// Bit-identity contract: matches `jax_ops.tangent_flow_forward_jacobian`
/// (the JAX `jacfwd` path) within `LIBM` ULPs — the only non-trivial entry
/// is `ratio.powf(delta_gamma)`, same libm call as Python's `**`.
pub fn tangent_flow_forward_jacobian(
    delta_gamma: f64,
    tau_obs: f64,
    tau_obs_ref: f64,
) -> Mat2 {
    if tau_obs > 0.0 && tau_obs_ref > 0.0 {
        let pow_ratio = (tau_obs / tau_obs_ref).powf(delta_gamma);
        [[1.0, 0.0], [0.0, pow_ratio]]
    } else {
        [[1.0, 0.0], [0.0, 1.0]]
    }
}

// ---------------------------------------------------------------------------
// Inversion residual  (mirror of jax_core.tangent_flow_inversion_residual)
// ---------------------------------------------------------------------------

/// Scalar squared-residual of the tangent-flow forward map at a candidate.
pub fn tangent_flow_inversion_residual(
    candidate_chit: f64,
    candidate_gamma: f64,
    target_substrate_chit: f64,
    target_substrate_gamma: f64,
    delta_chit: f64,
    delta_gamma: f64,
    tau_obs: f64,
    tau_obs_ref: f64,
) -> f64 {
    let (predicted_chit, predicted_gamma) = tangent_flow_substrate(
        candidate_chit,
        candidate_gamma,
        delta_chit,
        delta_gamma,
        tau_obs,
        tau_obs_ref,
    );
    let d_chit = predicted_chit - target_substrate_chit;
    let d_gamma = predicted_gamma - target_substrate_gamma;
    d_chit * d_chit + d_gamma * d_gamma
}

// ---------------------------------------------------------------------------
// 2x2 matrix helpers (closed-form inverse for the Laplace primitives)
// ---------------------------------------------------------------------------

pub type Mat2 = [[f64; 2]; 2];

/// Closed-form inverse of a 2x2 matrix. Returns `None` on singular.
pub fn inv_2x2(m: &Mat2) -> Option<Mat2> {
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    if det == 0.0 {
        return None;
    }
    let inv_det = 1.0 / det;
    Some([
        [m[1][1] * inv_det, -m[0][1] * inv_det],
        [-m[1][0] * inv_det, m[0][0] * inv_det],
    ])
}

/// `log |det M|` and `sign(det M)` — mirrors `jnp.linalg.slogdet` for 2x2.
pub fn slogdet_2x2(m: &Mat2) -> (f64, f64) {
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    if det == 0.0 {
        (0.0, f64::NEG_INFINITY)
    } else {
        (det.signum(), det.abs().ln())
    }
}

/// Scale a 2x2 matrix elementwise.
pub fn scale_2x2(m: &Mat2, s: f64) -> Mat2 {
    [[m[0][0] * s, m[0][1] * s], [m[1][0] * s, m[1][1] * s]]
}

// ---------------------------------------------------------------------------
// Laplace-approximation posterior
// (mirror of jax_core.laplace_covariance_from_jacobian / _from_hessian)
// ---------------------------------------------------------------------------

/// Posterior covariance under a Gaussian likelihood with isotropic noise.
///
///     Σ_post = sigma^2 (J^T J)^-1
///
/// For a 2-parameter canonical state (chit, gamma_AB) the Jacobian is
/// `(n_obs, 2)`. We compute `J^T J` as a 2x2 matrix in closed form.
/// Returns `None` if `J^T J` is singular.
pub fn laplace_covariance_from_jacobian(jacobian: &[[f64; 2]], noise_variance: f64) -> Option<Mat2> {
    let mut jtj: Mat2 = [[0.0; 2]; 2];
    for row in jacobian {
        jtj[0][0] += row[0] * row[0];
        jtj[0][1] += row[0] * row[1];
        jtj[1][0] += row[1] * row[0];
        jtj[1][1] += row[1] * row[1];
    }
    inv_2x2(&jtj).map(|inv| scale_2x2(&inv, noise_variance))
}

/// Posterior covariance from the full residual Hessian at MAP.
///
///     Σ_post = sigma^2 * H_R(MAP)^-1
///
/// Returns `None` if the Hessian is singular.
pub fn laplace_covariance_from_hessian(hessian: &Mat2, noise_variance: f64) -> Option<Mat2> {
    inv_2x2(hessian).map(|inv| scale_2x2(&inv, noise_variance))
}

/// Log-marginal-likelihood under the Laplace approximation (2x2 Hessian).
pub fn laplace_log_evidence(
    residual_at_map: f64,
    hessian: &Mat2,
    noise_variance: f64,
    n_obs: usize,
) -> f64 {
    let dim_c = 2.0;
    let precision = scale_2x2(hessian, 1.0 / noise_variance);
    let (_sign, log_det_precision) = slogdet_2x2(&precision);
    let two_pi_sigma2 = 2.0 * std::f64::consts::PI * noise_variance;
    -0.5 * residual_at_map / noise_variance
        - 0.5 * (n_obs as f64) * two_pi_sigma2.ln()
        + 0.5 * dim_c * (2.0 * std::f64::consts::PI).ln()
        - 0.5 * log_det_precision
}

// ---------------------------------------------------------------------------
// Non-Markovian Caputo flow via Prony sum-of-exponentials
// (mirror of jax_core.caputo_flow)
// ---------------------------------------------------------------------------

/// Prony sum-of-exponentials approximation of the Mittag-Leffler kernel.
///
///     chit(ν)     = chit_0     * Σ_k a_k exp(-b_k λ_chit  ν)
///     gamma_AB(ν) = gamma_AB_0 * Σ_k a_k exp(-b_k λ_gamma ν)
///
/// For β = 1 with prony_terms = [(1.0, 1.0)] the kernel reduces to
/// `exp(-λν)` — matches the v1 Markovian Banach exponential branch.
///
/// Naive left-to-right sum. JAX's `jnp.sum` uses pairwise reduction;
/// for prony lengths in the typical 4-16 range the last-bit divergence
/// is documented as platform tolerance per v6 BLOCK_IN acceptance.
pub fn caputo_flow(
    chit_0: f64,
    gamma_ab_0: f64,
    lambda_chit: f64,
    lambda_gamma: f64,
    nu: f64,
    prony_amplitudes: &[f64],
    prony_decays: &[f64],
) -> (f64, f64) {
    debug_assert_eq!(prony_amplitudes.len(), prony_decays.len());
    let mut chit_kernel = 0.0;
    let mut gamma_kernel = 0.0;
    for (a, b) in prony_amplitudes.iter().zip(prony_decays.iter()) {
        chit_kernel += a * (-b * lambda_chit * nu).exp();
        gamma_kernel += a * (-b * lambda_gamma * nu).exp();
    }
    (chit_0 * chit_kernel, gamma_ab_0 * gamma_kernel)
}

// ---------------------------------------------------------------------------
// Small MLP forward pass (mirror of jax_core.mlp_forward)
// ---------------------------------------------------------------------------

/// One layer of a small MLP. `w` shape is `(out_dim, in_dim)`; `b` is `(out_dim,)`.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MlpLayer {
    pub w: Vec<Vec<f64>>,
    pub b: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Activation {
    Tanh,
    Relu,
}

fn apply_activation(z: f64, act: Activation) -> f64 {
    match act {
        Activation::Tanh => z.tanh(),
        Activation::Relu => z.max(0.0),
    }
}

/// Forward pass through a small MLP with linear output layer.
///
/// Hidden layers apply the chosen elementwise nonlinearity; the output
/// layer is linear. Panics on empty `weights` (caller's bug, mirror of
/// the Python `len(weights)` precondition).
pub fn mlp_forward(x: &[f64], weights: &[MlpLayer], activation: Activation) -> Vec<f64> {
    assert!(!weights.is_empty(), "mlp_forward: empty weights");
    let n_layers = weights.len();
    let mut h: Vec<f64> = x.to_vec();
    for (i, layer) in weights.iter().enumerate() {
        let out_dim = layer.w.len();
        let mut next = Vec::with_capacity(out_dim);
        for j in 0..out_dim {
            debug_assert_eq!(layer.w[j].len(), h.len());
            let mut acc = layer.b[j];
            for k in 0..h.len() {
                acc += layer.w[j][k] * h[k];
            }
            next.push(acc);
        }
        if i < n_layers - 1 {
            for v in next.iter_mut() {
                *v = apply_activation(*v, activation);
            }
        }
        h = next;
    }
    h
}

// ---------------------------------------------------------------------------
// Learned translation field (mirror of jax_core.learned_field_substrate)
// ---------------------------------------------------------------------------

/// Forward map (canonical -> substrate) for a learned translation field.
///
/// Input to the MLP is `(chit, gamma_AB, log(tau/tau_ref))`. The log-ratio
/// is clamped to 0 at degenerate tau (identity in the tau direction).
/// Returns `(substrate_chit, substrate_gamma_AB)`.
pub fn learned_field_substrate(
    chit: f64,
    gamma_ab: f64,
    tau_obs: f64,
    tau_obs_ref: f64,
    weights: &[MlpLayer],
    activation: Activation,
) -> (f64, f64) {
    let log_ratio = if tau_obs > 0.0 && tau_obs_ref > 0.0 {
        (tau_obs / tau_obs_ref).ln()
    } else {
        0.0
    };
    let x = [chit, gamma_ab, log_ratio];
    let y = mlp_forward(&x, weights, activation);
    debug_assert!(y.len() >= 2);
    (y[0], y[1])
}
