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
