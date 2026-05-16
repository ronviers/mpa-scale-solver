"""Continuous-form flow `C^nu = exp(nu * ln C)` (handoff §C.1).

Grounded by v9_receipts §RG closure (Wilson-Kadanoff structural equivalence,
closed by composition) in Markovian scope (beta_mem = 1). The Banach
substrate sits exactly at the Markovian boundary by construction; that is
where continuous flow is in proven scope at v1.

v1 dispatch is on `field.shape`:

- `tangent_flow`: closed-form via the ScalingRule (and the refinement
  dict's optional `flow_kind` for substrate-specific closed forms like
  the Banach exponential decay).
- `lookup_table`: deferred to v2 (lookup tables provide samples, not an
  explicit generator; reconstructing the flow from a table requires the
  fractional-RG generalization that lands at v2 alongside JAX).

Non-Markovian Caputo (`beta_mem < 1`) is v2 (fractional-RG generalization
per v9_receipts §RG closure substrate-scope note).
"""

from __future__ import annotations

import math
from typing import Union

from .types import (
    AnyTranslationField,
    CanonicalState,
    TangentFlowField,
    TranslationField,
)


def flow(
    canonical_initial: CanonicalState,
    nu: float,
    field: AnyTranslationField,
) -> CanonicalState:
    """Continuous-form flow: canonical state at depth nu.

    Returns the canonical state reached by evolving `canonical_initial`
    under the field's flow generator for depth `nu`. For integer `nu = N`
    this is equivalent to N successive applications of the discrete map;
    for real `nu` it is the closed-form continuous flow.

    Dispatch:
      - `TangentFlowField` whose `scaling.refinement['flow_kind']` is
        `'banach_exponential'`: closed-form exp decay (the Banach
        substrate's canonical flow per Q1 of the v1 build session;
        `lambda_chit`, `lambda_gamma` in the refinement dict; defaults
        to 1.0).
      - `TangentFlowField` without a `flow_kind`: scales the canonical
        state via the ScalingRule treating `nu` as `tau_obs`.
      - `TranslationField` (lookup_table): raises NotImplementedError;
        defer to v2.
    """
    if isinstance(field, TangentFlowField):
        return _flow_tangent(canonical_initial, nu, field)
    if isinstance(field, TranslationField):
        raise NotImplementedError(
            "flow() on lookup_table fields is deferred to v2 "
            "(lookup tables sample the flow; no explicit generator at v1)."
        )
    raise TypeError(f"unsupported translation field type: {type(field).__name__}")


def _flow_tangent(
    canonical_initial: CanonicalState,
    nu: float,
    field: TangentFlowField,
) -> CanonicalState:
    """Tangent-flow closed form.

    Banach-exponential refinement uses the exp-decay form spelled out in
    Q1 of the v1 build session:

        chit(nu)    = chit_0 * exp(-lambda_chit * nu)
        gamma_AB(nu) = gamma_AB_0 * exp(-lambda_gamma * nu)

    Default lambdas = 1.0 correspond to the v1 normalization (spectral-gap
    eigenvalue exp(-1) of ln C; v2 derives these from `flow_spectrum`).

    Generic tangent-flow without a refinement uses the scaling rule
    treating nu as tau_obs. For the canonical default delta_gamma =
    delta_chit = 0, this is identity (the canonical state does not flow
    under the translation; the migration happens entirely in the substrate
    projection).
    """
    refinement = field.scaling.refinement or {}
    flow_kind = refinement.get("flow_kind") if isinstance(refinement, dict) else None

    if flow_kind == "banach_exponential":
        lambda_chit = float(refinement.get("lambda_chit", 1.0))
        lambda_gamma = float(refinement.get("lambda_gamma", 1.0))
        return CanonicalState(
            chit=canonical_initial.chit * math.exp(-lambda_chit * nu),
            gamma_AB=canonical_initial.gamma_AB * math.exp(-lambda_gamma * nu),
            k_frust=canonical_initial.k_frust,
        )

    # Generic tangent-flow: apply the ScalingRule with nu as tau_obs.
    rule = field.scaling
    if nu <= 0.0 or rule.tau_obs_ref <= 0.0:
        # Below or at the reference point the scaling rule is identity.
        return canonical_initial
    ratio = nu / rule.tau_obs_ref
    return CanonicalState(
        chit=canonical_initial.chit + rule.delta_chit * math.log(ratio),
        gamma_AB=canonical_initial.gamma_AB * (ratio ** rule.delta_gamma),
        k_frust=canonical_initial.k_frust,
    )
