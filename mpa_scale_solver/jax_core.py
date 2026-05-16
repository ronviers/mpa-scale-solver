"""Pure JAX math primitives — the canonical differentiable reference.

This module is the single math source the v2+ differentiable surface
reads. The v0/v1 closed forms in `operations._apply_tangent_flow`,
`flow._flow_tangent`, `banach.BanachSubstrate.state_at`, and
`operations.TranslationFieldIndex.nearest` keep their `math.*` / numpy
implementations unchanged (the byte-identity contract for the v0/v1
fixture suite). This module re-states the same math in `jax.numpy` so
the differentiable variants in `jax_ops.py` can autograd through it.

Discipline:
  - All functions are pure, JIT-able, differentiable.
  - Inputs and outputs are JAX arrays (or scalars convertible to them).
  - Float64 enabled at import (the v6 native port targets float64
    semantics; float32 would drift from the math.* / numpy contract).
  - `jnp.where`-guarded branches replace Python `if/else` so the
    functions stay traceable under `jax.jit` and `jax.grad`.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

# Float64 is required for parity with the math.* / numpy contract that
# the v0/v1 wrappers commit to. JAX defaults to float32; enabling x64
# is a global process-level setting (the standard scientific JAX
# pattern). Imported once at module load.
jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Tangent-flow forward map (mirror of operations._apply_tangent_flow)
# ---------------------------------------------------------------------------

def tangent_flow_substrate(
    chit: jnp.ndarray,
    gamma_AB: jnp.ndarray,
    delta_chit: jnp.ndarray,
    delta_gamma: jnp.ndarray,
    tau_obs: jnp.ndarray,
    tau_obs_ref: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Forward-map canonical to substrate via the ScalingRule closed form.

    Matches `operations._apply_tangent_flow`:

        scaled_chit  = chit + delta_chit  * log(tau_obs / tau_obs_ref)
        scaled_gamma = gamma_AB * (tau_obs / tau_obs_ref) ** delta_gamma

    For `tau_obs <= 0` or `tau_obs_ref <= 0` returns the canonical values
    unmodified (identity at degenerate tau_obs — matches the v0 branch).
    """
    safe_tau = jnp.where(tau_obs > 0.0, tau_obs, 1.0)
    safe_ref = jnp.where(tau_obs_ref > 0.0, tau_obs_ref, 1.0)
    ratio = safe_tau / safe_ref
    log_ratio = jnp.log(ratio)
    pow_ratio = ratio ** delta_gamma

    scaled_chit = chit + delta_chit * log_ratio
    scaled_gamma = gamma_AB * pow_ratio

    use_scaling = (tau_obs > 0.0) & (tau_obs_ref > 0.0)
    out_chit = jnp.where(use_scaling, scaled_chit, chit)
    out_gamma = jnp.where(use_scaling, scaled_gamma, gamma_AB)
    return out_chit, out_gamma


# ---------------------------------------------------------------------------
# Banach analytical canonical state (mirror of BanachSubstrate.state_at
# and flow._flow_tangent's `banach_exponential` branch)
# ---------------------------------------------------------------------------

def banach_state(
    chit_0: jnp.ndarray,
    gamma_AB_0: jnp.ndarray,
    lambda_chit: jnp.ndarray,
    lambda_gamma: jnp.ndarray,
    nu: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Canonical state at depth nu under Banach exponential decay.

        chit(nu)     = chit_0     * exp(-lambda_chit  * nu)
        gamma_AB(nu) = gamma_AB_0 * exp(-lambda_gamma * nu)

    Matches `BanachSubstrate.state_at` (Q1 v1 normalization).
    """
    return (
        chit_0 * jnp.exp(-lambda_chit * nu),
        gamma_AB_0 * jnp.exp(-lambda_gamma * nu),
    )


# ---------------------------------------------------------------------------
# Generic tangent flow (mirror of flow._flow_tangent's generic branch)
# ---------------------------------------------------------------------------

def tangent_flow_canonical(
    chit_0: jnp.ndarray,
    gamma_AB_0: jnp.ndarray,
    delta_chit: jnp.ndarray,
    delta_gamma: jnp.ndarray,
    nu: jnp.ndarray,
    tau_obs_ref: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Continuous-form canonical flow under a ScalingRule treating nu as tau_obs.

    Matches `flow._flow_tangent`'s generic (non-`banach_exponential`) branch:

        chit(nu)     = chit_0    + delta_chit  * log(nu / tau_obs_ref)
        gamma_AB(nu) = gamma_AB_0 * (nu / tau_obs_ref) ** delta_gamma

    Identity at `nu <= 0` or `tau_obs_ref <= 0` (matches v1 branch).
    """
    return tangent_flow_substrate(
        chit_0, gamma_AB_0,
        delta_chit, delta_gamma,
        nu, tau_obs_ref,
    )


# ---------------------------------------------------------------------------
# Lookup-table squared distance (mirror of TranslationFieldIndex.nearest)
# ---------------------------------------------------------------------------

def lookup_squared_distance(
    query_chit: jnp.ndarray,
    query_gamma: jnp.ndarray,
    field_chits: jnp.ndarray,
    field_gammas: jnp.ndarray,
    field_taus: jnp.ndarray,
    has_tau: jnp.ndarray,
    tau_obs: jnp.ndarray,
    tau_obs_weight: jnp.ndarray,
) -> jnp.ndarray:
    """Per-rule squared L2 distance with the log-tau term for tau-carrying rules.

    Matches `TranslationFieldIndex.nearest`'s `d2` computation. Returns the
    full per-rule `d2` array; `argmin(d2)` selects the nearest rule.

    Note: `argmin` is non-differentiable through the rule-index choice.
    This function is differentiable in (query_chit, query_gamma) and is
    the building block consumers compose into smoothed surrogates
    (softmin, Gumbel-softmax, etc.) when they need a smooth lookup.
    """
    d_chit = field_chits - query_chit
    d_gamma = field_gammas - query_gamma
    d2 = d_chit * d_chit + d_gamma * d_gamma

    log_tau_q = jnp.where(tau_obs > 0.0, jnp.log(jnp.where(tau_obs > 0.0, tau_obs, 1.0)), 0.0)
    safe_field_tau = jnp.where(has_tau, field_taus, 1.0)
    d_tau = jnp.where(has_tau, jnp.log(safe_field_tau) - log_tau_q, 0.0)
    return d2 + tau_obs_weight * d_tau * d_tau


# ---------------------------------------------------------------------------
# Analytical inverse of the tangent-flow forward map (exact, differentiable)
# ---------------------------------------------------------------------------

def tangent_flow_canonical_inverse(
    substrate_chit: jnp.ndarray,
    substrate_gamma_AB: jnp.ndarray,
    delta_chit: jnp.ndarray,
    delta_gamma: jnp.ndarray,
    tau_obs: jnp.ndarray,
    tau_obs_ref: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Exact closed-form inverse of `tangent_flow_substrate`.

    The forward map is monotonic and invertible everywhere `tau_obs > 0`
    and `tau_obs_ref > 0`:

        canonical_chit     = substrate_chit  - delta_chit * log(tau / tau_ref)
        canonical_gamma_AB = substrate_gamma / (tau / tau_ref) ** delta_gamma

    At degenerate `tau_obs <= 0` or `tau_obs_ref <= 0` the forward map
    is identity, so the inverse is identity too (returns the substrate
    values unmodified).

    Differentiable in (substrate_chit, substrate_gamma_AB, delta_chit,
    delta_gamma, tau_obs, tau_obs_ref) — composition under
    `jax.grad`/`jax.jacobian` gives the inversion sensitivity v2.1's
    Laplace approximation will compose against.
    """
    safe_tau = jnp.where(tau_obs > 0.0, tau_obs, 1.0)
    safe_ref = jnp.where(tau_obs_ref > 0.0, tau_obs_ref, 1.0)
    ratio = safe_tau / safe_ref
    log_ratio = jnp.log(ratio)
    pow_ratio = ratio ** delta_gamma

    inv_chit = substrate_chit - delta_chit * log_ratio
    inv_gamma = substrate_gamma_AB / pow_ratio

    use_scaling = (tau_obs > 0.0) & (tau_obs_ref > 0.0)
    out_chit = jnp.where(use_scaling, inv_chit, substrate_chit)
    out_gamma = jnp.where(use_scaling, inv_gamma, substrate_gamma_AB)
    return out_chit, out_gamma


# ---------------------------------------------------------------------------
# Inversion residual surface (forward-search scoring; differentiable surrogate)
# ---------------------------------------------------------------------------

def tangent_flow_inversion_residual(
    candidate_chit: jnp.ndarray,
    candidate_gamma: jnp.ndarray,
    target_substrate_chit: jnp.ndarray,
    target_substrate_gamma: jnp.ndarray,
    delta_chit: jnp.ndarray,
    delta_gamma: jnp.ndarray,
    tau_obs: jnp.ndarray,
    tau_obs_ref: jnp.ndarray,
) -> jnp.ndarray:
    """Scalar squared-residual of the tangent-flow forward map at a candidate.

    The inversion `forward_sweep_invert` minimizes the analogous score
    (L2 over shared observable keys). For a tangent-flow field the two
    relevant keys are `substrate_chit` / `substrate_gamma_AB`; this
    function returns that score in closed form so `jax.grad` /
    `jax.scipy.optimize.minimize` can drive a gradient-based inversion
    (replacing the brute-force grid in monotonic regions per BLOCK_IN
    §v5; the v2 differentiable layer here is what v5's gradient-based
    inversion will compose).
    """
    predicted_chit, predicted_gamma = tangent_flow_substrate(
        candidate_chit, candidate_gamma,
        delta_chit, delta_gamma,
        tau_obs, tau_obs_ref,
    )
    d_chit = predicted_chit - target_substrate_chit
    d_gamma = predicted_gamma - target_substrate_gamma
    return d_chit * d_chit + d_gamma * d_gamma


# ---------------------------------------------------------------------------
# Laplace-approximation posterior (v2.1 — BLOCK_IN cut b)
# ---------------------------------------------------------------------------

def laplace_covariance_from_jacobian(
    jacobian: jnp.ndarray,
    noise_variance: jnp.ndarray,
) -> jnp.ndarray:
    """Posterior covariance under a Gaussian likelihood with isotropic noise.

    For a forward map `y = f(c)` with Gaussian observation noise
    `y_obs = y + ε`, `ε ~ N(0, sigma^2 I)`, the Laplace-approximation
    posterior covariance over `c` evaluated at the MAP point is

        Σ_post = sigma^2 (J^T J)^-1

    where `J = ∂f/∂c` at the MAP. This holds exactly when the residual
    at MAP is zero (the closed-form tangent-flow inverse case) and is
    the leading-order approximation otherwise. The full Hessian
    `H = J^T J - residual ⊗ ∂J/∂c` is computed via `jax.hessian` in
    callers that need the higher-order correction.

    Singular `J^T J` (rank-deficient Jacobian — e.g. a degenerate
    parameter direction) raises `numpy.linalg.LinAlgError` through
    `jnp.linalg.inv`. The Banach identity field at `tau_obs = 1` is the
    documented well-conditioned case.
    """
    jtj = jacobian.T @ jacobian
    return noise_variance * jnp.linalg.inv(jtj)


def laplace_covariance_from_hessian(
    hessian: jnp.ndarray,
    noise_variance: jnp.ndarray,
) -> jnp.ndarray:
    """Posterior covariance from the full Hessian of the residual.

    For the squared-residual cost `R(c) = ||y - f(c)||^2`, the
    negative-log-likelihood under isotropic Gaussian noise is
    `(1/2sigma^2) R(c)`. Its Hessian at MAP is `(1/sigma^2) * H_R(MAP)`
    where `H_R` is the residual Hessian. The Laplace posterior
    covariance is

        Σ_post = sigma^2 * H_R(MAP)^-1

    Use this form when the residual at MAP is non-zero (lookup-table
    inversion, learned-field inversion); use
    `laplace_covariance_from_jacobian` for the zero-residual fast path.
    """
    return noise_variance * jnp.linalg.inv(hessian)


# ---------------------------------------------------------------------------
# Non-Markovian Caputo flow via Prony sum-of-exponentials (v2.4 — cut e)
# ---------------------------------------------------------------------------

def caputo_flow(
    chit_0: jnp.ndarray,
    gamma_AB_0: jnp.ndarray,
    lambda_chit: jnp.ndarray,
    lambda_gamma: jnp.ndarray,
    nu: jnp.ndarray,
    prony_amplitudes: jnp.ndarray,
    prony_decays: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Non-Markovian Caputo flow via a Prony sum-of-exponentials kernel.

    The Mittag-Leffler kernel `E_β(-λν)` for `β < 1` is the canonical
    non-Markovian flow generator (v9_receipts §RG closure substrate-
    scope note). The curator pre-fits a Prony sum

        E_β(-x) ≈ Σ_k a_k exp(-b_k x)

    and ships `(a_k, b_k)` on the `ScalingRule.refinement` dict as
    `prony_terms`. This primitive computes per-axis

        chit(ν)     = chit_0     * Σ_k a_k exp(-b_k λ_chit  ν)
        gamma_AB(ν) = gamma_AB_0 * Σ_k a_k exp(-b_k λ_gamma ν)

    For `β = 1` with `prony_terms = [(1.0, 1.0)]`, the kernel reduces to
    `exp(-λν)` and the result is byte-identical to the v1 Markovian
    Banach exponential branch in `flow._flow_tangent`.

    Differentiable in all parameters (chit_0, gamma_AB_0, lambdas, ν,
    prony amplitudes and decays). Composes under `jax.grad` /
    `jax.jacobian`; v5's sensitivity backprop will compose against the
    same surface.

    `prony_amplitudes` and `prony_decays` are 1-D arrays of equal length;
    a length-mismatch is the caller's bug (no defensive check — the
    jax broadcast would fail with a shape error in any case).
    """
    chit_kernel = jnp.sum(prony_amplitudes * jnp.exp(-prony_decays * lambda_chit * nu))
    gamma_kernel = jnp.sum(prony_amplitudes * jnp.exp(-prony_decays * lambda_gamma * nu))
    return chit_0 * chit_kernel, gamma_AB_0 * gamma_kernel


# ---------------------------------------------------------------------------
# Laplace log evidence
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Learned translation field (v3 — BLOCK_IN §v3): small MLP forward map
# ---------------------------------------------------------------------------

def mlp_forward(
    x: jnp.ndarray,
    weights: Tuple[Tuple[jnp.ndarray, jnp.ndarray], ...],
    activation: str = "tanh",
) -> jnp.ndarray:
    """Forward pass through a small MLP with linear output layer.

    `weights` is a tuple of per-layer `(W, b)` pairs where `W` has shape
    `(out_dim, in_dim)` (matrix-vector product `W @ x + b`). Hidden layers
    apply the chosen elementwise nonlinearity (tanh or relu); the output
    layer is linear.

    Pure / JIT-able / differentiable in (x, weights). The v3 learned
    translation-field shape composes this primitive through the
    `(chit, gamma_AB, log(tau/tau_ref))` -> `(substrate_chit,
    substrate_gamma_AB)` forward map.

    Empty `weights` is an error (no architecture); the caller's bug.
    """
    if activation == "tanh":
        act = jnp.tanh
    elif activation == "relu":
        act = lambda z: jnp.maximum(z, 0.0)  # noqa: E731
    else:
        raise ValueError(f"unsupported activation: {activation!r}")

    n_layers = len(weights)
    h = x
    for i, (W, b) in enumerate(weights):
        h = W @ h + b
        if i < n_layers - 1:
            h = act(h)
    return h


def learned_field_substrate(
    chit: jnp.ndarray,
    gamma_AB: jnp.ndarray,
    tau_obs: jnp.ndarray,
    tau_obs_ref: jnp.ndarray,
    weights: Tuple[Tuple[jnp.ndarray, jnp.ndarray], ...],
    activation: str = "tanh",
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Forward map (canonical -> substrate) for a learned translation field.

    Input to the MLP is the 3-vector `(chit, gamma_AB, log(tau/tau_ref))`.
    The log-ratio coordinate matches the tangent-flow parametrization
    (`scaled_chit = chit + delta_chit * log(ratio)`), letting curators
    train against the same tau-scaling structure other field shapes use.

    At degenerate `tau_obs <= 0` or `tau_obs_ref <= 0` the log-ratio is
    clamped to 0 (identity in the tau direction; mirrors the
    `tangent_flow_substrate` branch).

    Differentiable in all parameters via `jax.grad` / `jax.jacobian`.
    """
    safe_tau = jnp.where(tau_obs > 0.0, tau_obs, 1.0)
    safe_ref = jnp.where(tau_obs_ref > 0.0, tau_obs_ref, 1.0)
    log_ratio = jnp.where(
        (tau_obs > 0.0) & (tau_obs_ref > 0.0),
        jnp.log(safe_tau / safe_ref),
        0.0,
    )
    x = jnp.stack([chit, gamma_AB, log_ratio])
    y = mlp_forward(x, weights, activation=activation)
    return y[0], y[1]


def laplace_log_evidence(
    residual_at_map: jnp.ndarray,
    hessian: jnp.ndarray,
    noise_variance: jnp.ndarray,
    n_obs: int,
) -> jnp.ndarray:
    """Log-marginal-likelihood under the Laplace approximation.

        log p(y) ≈ -0.5 * R(MAP) / sigma^2
                   - 0.5 * n_obs * log(2*pi*sigma^2)
                   + 0.5 * dim_c * log(2*pi)
                   - 0.5 * log det((1/sigma^2) * H_R(MAP))

    For Bayesian model comparison between competing driver profiles or
    competing intent maps; downstream consumers in v3's active learning
    will weight candidate measurements by expected information gain
    derived from this surface.
    """
    dim_c = hessian.shape[0]
    log_det_precision = jnp.linalg.slogdet(hessian / noise_variance)[1]
    return (
        -0.5 * residual_at_map / noise_variance
        - 0.5 * n_obs * jnp.log(2.0 * jnp.pi * noise_variance)
        + 0.5 * dim_c * jnp.log(2.0 * jnp.pi)
        - 0.5 * log_det_precision
    )
