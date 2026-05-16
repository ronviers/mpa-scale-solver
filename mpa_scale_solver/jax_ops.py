"""Differentiable v2 operation surface.

These functions are the v2-new entry points consumers call when they
need gradients. They take the same dataclasses as the v0/v1 surface
(CanonicalState / TangentFlowField / BanachSubstrate) but return JAX
arrays instead of dataclasses, so the result is differentiable under
`jax.grad` / `jax.jacobian` / `jax.hessian`.

The seven-operation API in `operations.py` is unchanged; this module
adds a parallel `*_diff` surface that v2+ consumers opt into. Per
BLOCK_IN.md §v2 cut (a), these are the JAX-foundation primitives that
v2.1's Bayesian inversion (Laplace around MAP) and v5's gradient-based
forward_sweep_invert will compose on top of.

Three top-level entries:
  - `tangent_flow_substrate_diff` — forward map (canonical -> substrate)
    for tangent-flow fields. Differentiable in canonical coordinates
    and in field parameters.
  - `flow_diff` — continuous-form flow on tangent-flow fields. Banach
    exponential and generic tangent-flow branches mirror the v1
    `flow.flow()` dispatch.
  - `forward_sweep_invert_diff` — gradient-based inversion for tangent-
    flow fields via L-BFGS, with the brute-force grid retained for
    lookup_table fields (argmin is non-differentiable through the rule
    choice; gradient inversion does not apply).

All entries import `jax_pytree` for CanonicalState PyTree registration
(side-effect on first import; idempotent).
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from . import jax_pytree  # noqa: F401 — side-effect: PyTree registration
from .banach import BanachSubstrate
from .jax_core import (
    banach_state,
    tangent_flow_canonical,
    tangent_flow_canonical_inverse,
    tangent_flow_substrate,
)
from .types import (
    AnyTranslationField,
    CanonicalState,
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

    Mirrors `flow.flow()` dispatch on `field.shape` / `flow_kind`:
      - tangent_flow + `flow_kind == 'banach_exponential'` -> Banach decay.
      - tangent_flow + generic -> ScalingRule with nu as tau_obs.
      - lookup_table -> NotImplementedError (matches v1).

    Returns `(chit, gamma_AB)` as JAX 0-d arrays.
    """
    if isinstance(field, TranslationField):
        raise NotImplementedError(
            "flow_diff() on lookup_table fields is deferred to v2.x "
            "(no explicit generator at v1; fractional-RG generalization is v2.4)."
        )
    if not isinstance(field, TangentFlowField):
        raise TypeError(f"unsupported translation field type: {type(field).__name__}")

    refinement = field.scaling.refinement or {}
    flow_kind = refinement.get("flow_kind") if isinstance(refinement, dict) else None
    nu_jax = jnp.asarray(nu, dtype=jnp.float64)
    chit_0 = jnp.asarray(canonical_initial.chit, dtype=jnp.float64)
    gamma_0 = jnp.asarray(canonical_initial.gamma_AB, dtype=jnp.float64)

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
# Banach analytical state — differentiable in (chit_0, gamma_0, lambda_*, nu)
# ---------------------------------------------------------------------------

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
