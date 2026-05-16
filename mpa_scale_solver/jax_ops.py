"""Differentiable v2 operation surface.

These functions are the v2-new entry points consumers call when they
need gradients. They take the same dataclasses as the v0/v1 surface
(CanonicalState / TangentFlowField / BanachSubstrate) but return JAX
arrays instead of dataclasses, so the result is differentiable under
`jax.grad` / `jax.jacobian` / `jax.hessian`.

The seven-operation API in `operations.py` is unchanged; this module
adds a parallel surface that v2+ consumers opt into.

Top-level entries:
  - `tangent_flow_substrate_diff` — forward map (canonical -> substrate)
    for tangent-flow fields. Differentiable in canonical coordinates
    and in field parameters.
  - `flow_diff` — continuous-form flow on tangent-flow fields. Banach
    exponential and generic tangent-flow branches mirror the v1
    `flow.flow()` dispatch.
  - `tangent_flow_forward_jacobian` — 2x2 Jacobian of the forward map
    at the given canonical state.
  - `forward_sweep_invert_diff` — exact closed-form analytical inverse
    for tangent-flow fields (differentiable through the target).
  - `banach_state_diff` — differentiable Banach analytical canonical
    state.
  - `tangent_flow_posterior` / `lookup_table_posterior` — v2.1 Laplace
    approximation posteriors over canonical states (BLOCK_IN cut b).

All entries import `jax_pytree` for CanonicalState PyTree registration
(side-effect on first import; idempotent).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from . import jax_pytree  # noqa: F401 — side-effect: PyTree registration
from .banach import BanachSubstrate
from .jax_core import (
    banach_state,
    caputo_flow,
    laplace_covariance_from_jacobian,
    learned_field_substrate,
    tangent_flow_canonical,
    tangent_flow_canonical_inverse,
    tangent_flow_substrate,
)
from .types import (
    AnyTranslationField,
    CanonicalState,
    LearnedField,
    SubstrateState,
    TangentFlowField,
    TranslationField,
)


# ---------------------------------------------------------------------------
# Forward map (canonical -> substrate) — tangent_flow only
# ---------------------------------------------------------------------------

def tangent_flow_substrate_diff(
    canonical: CanonicalState,
    field: TangentFlowField,
    tau_obs: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Differentiable forward map for a tangent-flow translation field.

    Returns `(substrate_chit, substrate_gamma_AB)` as JAX 0-d arrays.
    Differentiable in `canonical.chit`, `canonical.gamma_AB`, and the
    field's `delta_chit` / `delta_gamma` / `tau_obs_ref` parameters.

    `lookup_table` fields are not supported here — the v0 nearest-neighbor
    dispatch is non-differentiable through the rule choice. Consumers
    that need gradients on lookup-table outputs use the smooth surrogate
    composition in `jax_core.lookup_squared_distance`.
    """
    rule = field.scaling
    return tangent_flow_substrate(
        chit=jnp.asarray(canonical.chit, dtype=jnp.float64),
        gamma_AB=jnp.asarray(canonical.gamma_AB, dtype=jnp.float64),
        delta_chit=jnp.asarray(rule.delta_chit, dtype=jnp.float64),
        delta_gamma=jnp.asarray(rule.delta_gamma, dtype=jnp.float64),
        tau_obs=jnp.asarray(tau_obs, dtype=jnp.float64),
        tau_obs_ref=jnp.asarray(rule.tau_obs_ref, dtype=jnp.float64),
    )


# ---------------------------------------------------------------------------
# Continuous-form flow — tangent_flow only
# ---------------------------------------------------------------------------

def flow_diff(
    canonical_initial: CanonicalState,
    nu: float,
    field: AnyTranslationField,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Differentiable continuous-form flow.

    Mirrors `flow.flow()` dispatch:
      - tangent_flow + `beta_mem < 1` -> v2.4 Caputo (Prony sum).
      - tangent_flow + `flow_kind == 'banach_exponential'` -> Banach decay.
      - tangent_flow + generic -> ScalingRule with nu as tau_obs.
      - lookup_table -> NotImplementedError.

    Returns `(chit, gamma_AB)` as JAX 0-d arrays.
    """
    if isinstance(field, TranslationField):
        raise NotImplementedError(
            "flow_diff() on lookup_table fields: lookup tables sample the "
            "flow; no explicit generator. Use a tangent_flow field "
            "(generic, banach_exponential, or Caputo) for differentiable flow."
        )
    if not isinstance(field, TangentFlowField):
        raise TypeError(f"unsupported translation field type: {type(field).__name__}")

    refinement = field.scaling.refinement or {}
    beta_mem = float(refinement.get("beta_mem", 1.0))
    flow_kind = refinement.get("flow_kind") if isinstance(refinement, dict) else None
    nu_jax = jnp.asarray(nu, dtype=jnp.float64)
    chit_0 = jnp.asarray(canonical_initial.chit, dtype=jnp.float64)
    gamma_0 = jnp.asarray(canonical_initial.gamma_AB, dtype=jnp.float64)

    if beta_mem < 1.0:
        prony_terms = refinement.get("prony_terms")
        if not prony_terms:
            raise ValueError(
                "beta_mem < 1.0 requires prony_terms in refinement "
                "(curator-supplied Mittag-Leffler approximation)."
            )
        amps = jnp.asarray(
            [float(a) for a, _ in prony_terms], dtype=jnp.float64,
        )
        decays = jnp.asarray(
            [float(b) for _, b in prony_terms], dtype=jnp.float64,
        )
        lambda_chit = jnp.asarray(refinement.get("lambda_chit", 1.0), dtype=jnp.float64)
        lambda_gamma = jnp.asarray(refinement.get("lambda_gamma", 1.0), dtype=jnp.float64)
        return caputo_flow(
            chit_0=chit_0, gamma_AB_0=gamma_0,
            lambda_chit=lambda_chit, lambda_gamma=lambda_gamma,
            nu=nu_jax,
            prony_amplitudes=amps, prony_decays=decays,
        )

    if flow_kind == "banach_exponential":
        lambda_chit = jnp.asarray(refinement.get("lambda_chit", 1.0), dtype=jnp.float64)
        lambda_gamma = jnp.asarray(refinement.get("lambda_gamma", 1.0), dtype=jnp.float64)
        return banach_state(chit_0, gamma_0, lambda_chit, lambda_gamma, nu_jax)

    rule = field.scaling
    return tangent_flow_canonical(
        chit_0=chit_0,
        gamma_AB_0=gamma_0,
        delta_chit=jnp.asarray(rule.delta_chit, dtype=jnp.float64),
        delta_gamma=jnp.asarray(rule.delta_gamma, dtype=jnp.float64),
        nu=nu_jax,
        tau_obs_ref=jnp.asarray(rule.tau_obs_ref, dtype=jnp.float64),
    )


# ---------------------------------------------------------------------------
# Jacobian of the forward map (∂substrate / ∂canonical) — sensitivity exposure
# ---------------------------------------------------------------------------

def tangent_flow_forward_jacobian(
    canonical: CanonicalState,
    field: TangentFlowField,
    tau_obs: float,
) -> jnp.ndarray:
    """Jacobian of the tangent-flow forward map at the given canonical state.

    Returns a 2x2 JAX array:

        [[ ∂substrate_chit  / ∂canonical_chit,    ∂substrate_chit  / ∂canonical_gamma_AB ],
         [ ∂substrate_gamma / ∂canonical_chit,    ∂substrate_gamma / ∂canonical_gamma_AB ]]

    The free sensitivity surface that v5's `sensitivity_backprop` and
    v2.1's Laplace approximation will compose against.
    """
    rule = field.scaling
    delta_chit = jnp.asarray(rule.delta_chit, dtype=jnp.float64)
    delta_gamma = jnp.asarray(rule.delta_gamma, dtype=jnp.float64)
    tau_obs_jax = jnp.asarray(tau_obs, dtype=jnp.float64)
    tau_obs_ref = jnp.asarray(rule.tau_obs_ref, dtype=jnp.float64)

    def forward(params: jnp.ndarray) -> jnp.ndarray:
        s_chit, s_gamma = tangent_flow_substrate(
            params[0], params[1],
            delta_chit, delta_gamma,
            tau_obs_jax, tau_obs_ref,
        )
        return jnp.stack([s_chit, s_gamma])

    canonical_vec = jnp.array(
        [canonical.chit, canonical.gamma_AB], dtype=jnp.float64,
    )
    return jax.jacfwd(forward)(canonical_vec)


# ---------------------------------------------------------------------------
# Inversion — exact closed-form for tangent_flow
# ---------------------------------------------------------------------------

def forward_sweep_invert_diff(
    target_substrate: SubstrateState,
    field: TangentFlowField,
    tau_obs: float,
    *,
    k_frust: bool = False,
) -> CanonicalState:
    """Exact closed-form inverse of the tangent-flow forward map.

    The tangent-flow scaling rule is monotonic and analytically
    invertible everywhere `tau_obs > 0` and `tau_obs_ref > 0`:

        canonical_chit     = substrate_chit  - delta_chit * log(tau / tau_ref)
        canonical_gamma_AB = substrate_gamma / (tau / tau_ref) ** delta_gamma

    For the differentiable consumer surface, this routes through
    `jax_core.tangent_flow_canonical_inverse` so gradients flow through
    the inversion (the sensitivity surface v2.1's Laplace approximation
    will compose against). Recovery is exact at float64 precision for
    any target generated by the forward map — no iterative solver
    needed.

    Generic-differentiable inversion (BFGS / L-BFGS on arbitrary
    forward maps — e.g. v3's learned translation-field) lands in v5
    per BLOCK_IN §v5. `lookup_table` fields stay on the v0 brute-force
    grid in `operations.forward_sweep_invert` (argmin is
    non-differentiable through the rule choice).

    Target substrate keys consumed: `observables['substrate_chit']` and
    `observables['substrate_gamma_AB']` — the tangent-flow forward
    map's output keys.
    """
    obs = target_substrate.observables
    if "substrate_chit" not in obs or "substrate_gamma_AB" not in obs:
        raise ValueError(
            "forward_sweep_invert_diff requires the target SubstrateState's "
            "observables to carry 'substrate_chit' and 'substrate_gamma_AB' "
            "(the tangent-flow forward-map output keys)."
        )
    rule = field.scaling
    inv_chit, inv_gamma = tangent_flow_canonical_inverse(
        substrate_chit=jnp.asarray(obs["substrate_chit"], dtype=jnp.float64),
        substrate_gamma_AB=jnp.asarray(obs["substrate_gamma_AB"], dtype=jnp.float64),
        delta_chit=jnp.asarray(rule.delta_chit, dtype=jnp.float64),
        delta_gamma=jnp.asarray(rule.delta_gamma, dtype=jnp.float64),
        tau_obs=jnp.asarray(tau_obs, dtype=jnp.float64),
        tau_obs_ref=jnp.asarray(rule.tau_obs_ref, dtype=jnp.float64),
    )
    return CanonicalState(
        chit=float(inv_chit),
        gamma_AB=float(inv_gamma),
        k_frust=k_frust,
    )


# ---------------------------------------------------------------------------
# Posterior (Laplace approximation) — v2.1 BLOCK_IN cut b
# ---------------------------------------------------------------------------

def tangent_flow_posterior(
    target_substrate: SubstrateState,
    field: TangentFlowField,
    tau_obs: float,
    *,
    noise_variance: float = 1.0,
    k_frust: bool = False,
):
    """Laplace-approximation posterior for tangent-flow inversion.

    Fast-path (closed-form):
      - MAP = `forward_sweep_invert_diff` (exact analytical inverse)
      - Residual at MAP = 0 (exact)
      - Hessian at MAP = (1/sigma^2) J^T J where J is the forward-map
        Jacobian
      - Posterior covariance = sigma^2 * inv(J^T J)
      - Log-evidence reduces to the noise-prior-only normalizer (since
        the residual term vanishes)

    Returns a `Posterior` (imported lazily to avoid cycles).
    """
    from .types import Posterior

    map_estimate = forward_sweep_invert_diff(
        target_substrate, field, tau_obs, k_frust=k_frust,
    )
    jac = tangent_flow_forward_jacobian(map_estimate, field, tau_obs)
    cov = laplace_covariance_from_jacobian(
        jac, jnp.asarray(noise_variance, dtype=jnp.float64),
    )
    cov_np = np.asarray(cov)
    cov_tuple = (
        (float(cov_np[0, 0]), float(cov_np[0, 1])),
        (float(cov_np[1, 0]), float(cov_np[1, 1])),
    )

    # Log evidence at zero residual:
    #   log p(y) = -0.5 * dim_y * log(2*pi*sigma^2)
    #              + 0.5 * dim_c * log(2*pi)
    #              - 0.5 * log det((1/sigma^2) J^T J)
    dim_y = 2  # substrate_chit, substrate_gamma_AB
    dim_c = 2
    jtj = jac.T @ jac
    log_det_precision = float(jnp.linalg.slogdet(jtj / noise_variance)[1])
    log_evidence = float(
        -0.5 * dim_y * math.log(2.0 * math.pi * noise_variance)
        + 0.5 * dim_c * math.log(2.0 * math.pi)
        - 0.5 * log_det_precision
    )

    return Posterior(
        mean=map_estimate,
        covariance=cov_tuple,
        noise_variance=float(noise_variance),
        log_evidence=log_evidence,
        modes=(),
        notes=("laplace_from_closed_form_jacobian",),
    )


def lookup_table_posterior(
    target_substrate: SubstrateState,
    field: TranslationField,
    tau_obs: float,
    canonical_grid,
    *,
    noise_variance: float = 1.0,
    k_frust: bool = False,
    score_fn=None,
    top_k: int = 5,
):
    """Weighted-moment posterior for lookup-table inversion.

    Discrete grids don't have a meaningful Hessian (the residual is a
    step function over candidates), so the Laplace formula doesn't
    apply directly. Instead we treat the residual field as defining an
    unnormalized log-posterior

        log p(c | y) ∝ -0.5 * R(c) / sigma^2

    and report the moments of the resulting discrete distribution.
    Concentrating on the `top_k` lowest-residual candidates keeps the
    estimate insensitive to far-tail candidates that contribute
    essentially nothing.

    MAP = argmin candidate. Mean / covariance computed from the
    softmax-weighted moments over the top_k candidates. For k=1 this
    degenerates to a delta posterior at MAP with zero covariance (we
    add `noise_variance / 1.0` to the diagonal as a minimum-resolution
    proxy in that case so the result remains usable as a covariance).
    """
    from .operations import forward_sweep_invert
    from .types import Posterior

    map_state, _residual, residual_field = forward_sweep_invert(
        target_substrate, field, tau_obs, canonical_grid,
        score_fn=score_fn, return_residual_field=True,
    )
    map_with_kfrust = CanonicalState(
        chit=map_state.chit, gamma_AB=map_state.gamma_AB, k_frust=k_frust,
    )

    # Top-k indices by residual
    n = residual_field.shape[0]
    k = max(1, min(int(top_k), n))
    order = np.argsort(residual_field)
    idx = order[:k]
    top_residuals = residual_field[idx]
    top_points = canonical_grid[idx]

    if k == 1:
        # Degenerate: single candidate. Return MAP with noise-floor
        # covariance as a minimum-resolution proxy.
        cov_tuple = (
            (float(noise_variance), 0.0),
            (0.0, float(noise_variance)),
        )
        return Posterior(
            mean=map_with_kfrust,
            covariance=cov_tuple,
            noise_variance=float(noise_variance),
            log_evidence=None,
            modes=(),
            notes=("lookup_table_grid_top_k=1_delta_with_noise_floor",),
        )

    # log-weights: shift by min for numerical stability before exp
    log_weights = -0.5 * top_residuals / max(noise_variance, 1e-300)
    log_weights = log_weights - log_weights.max()
    weights = np.exp(log_weights)
    weights = weights / weights.sum()

    mean_chit = float(np.sum(weights * top_points[:, 0]))
    mean_gamma = float(np.sum(weights * top_points[:, 1]))

    dchit = top_points[:, 0] - mean_chit
    dgamma = top_points[:, 1] - mean_gamma
    cov_cc = float(np.sum(weights * dchit * dchit))
    cov_cg = float(np.sum(weights * dchit * dgamma))
    cov_gg = float(np.sum(weights * dgamma * dgamma))
    cov_tuple = ((cov_cc, cov_cg), (cov_cg, cov_gg))

    posterior_mean = CanonicalState(
        chit=mean_chit, gamma_AB=mean_gamma, k_frust=k_frust,
    )

    return Posterior(
        mean=posterior_mean,
        covariance=cov_tuple,
        noise_variance=float(noise_variance),
        log_evidence=None,
        modes=(map_with_kfrust,) if (map_with_kfrust.chit, map_with_kfrust.gamma_AB) != (mean_chit, mean_gamma) else (),
        notes=(f"lookup_table_weighted_moments_top_k={k}",),
    )


# ---------------------------------------------------------------------------
# Banach analytical state — differentiable in (chit_0, gamma_0, lambda_*, nu)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Learned translation field (v3 — BLOCK_IN §v3): differentiable forward map
# ---------------------------------------------------------------------------

def _learned_weights_as_jax(field: "LearnedField"):
    """Convert the nested-tuple weight storage into JAX arrays per layer."""
    return tuple(
        (
            jnp.asarray(W, dtype=jnp.float64),
            jnp.asarray(b, dtype=jnp.float64),
        )
        for W, b in field.weights
    )


def learned_field_substrate_diff(
    canonical: CanonicalState,
    field: "LearnedField",
    tau_obs: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Differentiable forward map for a learned translation field.

    Returns `(substrate_chit, substrate_gamma_AB)` as JAX 0-d arrays.
    Differentiable in `canonical.chit`, `canonical.gamma_AB`, and the
    MLP weights. The training surface is curator-side (mpa-conform);
    here the solver evaluates against curator-shipped weights.

    Input vector to the MLP is `(chit, gamma_AB, log(tau / tau_ref))`;
    output is `(substrate_chit, substrate_gamma_AB)`. The activation is
    the field's declared activation; the output layer is linear.
    """
    weights_jax = _learned_weights_as_jax(field)
    return learned_field_substrate(
        chit=jnp.asarray(canonical.chit, dtype=jnp.float64),
        gamma_AB=jnp.asarray(canonical.gamma_AB, dtype=jnp.float64),
        tau_obs=jnp.asarray(tau_obs, dtype=jnp.float64),
        tau_obs_ref=jnp.asarray(field.tau_obs_ref, dtype=jnp.float64),
        weights=weights_jax,
        activation=field.activation,
    )


def banach_state_diff(
    substrate: BanachSubstrate,
    nu: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Differentiable Banach analytical canonical state.

    Mirrors `BanachSubstrate.state_at(nu)` but returns JAX arrays so
    consumers can differentiate w.r.t. `chit_0`, `gamma_AB_0`,
    `lambda_chit`, `lambda_gamma`, or `nu`. The differentiable surface
    behind v2.1's Bayesian inversion on Banach observations.
    """
    return banach_state(
        chit_0=jnp.asarray(substrate.chit_0, dtype=jnp.float64),
        gamma_AB_0=jnp.asarray(substrate.gamma_AB_0, dtype=jnp.float64),
        lambda_chit=jnp.asarray(substrate.lambda_chit, dtype=jnp.float64),
        lambda_gamma=jnp.asarray(substrate.lambda_gamma, dtype=jnp.float64),
        nu=jnp.asarray(nu, dtype=jnp.float64),
    )
