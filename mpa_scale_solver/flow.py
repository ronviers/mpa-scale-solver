"""Continuous-form flow `C^nu = exp(nu * ln C)` (handoff §C.1).

Grounded by v9_receipts §RG closure (Wilson-Kadanoff structural equivalence,
closed by composition) in Markovian scope (beta_mem = 1). The Banach
substrate sits exactly at the Markovian boundary by construction.

Dispatch on `field.shape` and the refinement dict:

- `tangent_flow` + `beta_mem < 1`: v2.4 non-Markovian Caputo path. The
  flow generator is the Mittag-Leffler kernel `E_β(-λν)` approximated by
  the curator-supplied Prony sum `Σ_k a_k exp(-b_k x)` riding the
  refinement dict as `prony_terms`. β = 1 with single-term Prony
  `[(1.0, 1.0)]` reduces byte-identically to the Markovian path.
- `tangent_flow` + `flow_kind == 'banach_exponential'`: v1 Markovian
  exponential decay (Banach Q1 normalization).
- `tangent_flow` + generic: v1 generic tangent-flow (ScalingRule with
  nu treated as tau_obs).
- `lookup_table`: NotImplementedError (lookup tables provide samples,
  not an explicit generator).
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

    v2.4 Caputo branch fires when `refinement['beta_mem'] < 1.0`; the
    curator-supplied `prony_terms` approximate the Mittag-Leffler kernel.
    Per-axis lambdas (`lambda_chit`, `lambda_gamma`) scale the Prony
    decay rates. Otherwise the v1 branches apply:

      - `flow_kind == 'banach_exponential'`: exp-decay form (Banach Q1
        normalization).
      - generic: ScalingRule with `nu` treated as `tau_obs`.
    """
    refinement = field.scaling.refinement or {}
    beta_mem = float(refinement.get("beta_mem", 1.0))
    flow_kind = refinement.get("flow_kind") if isinstance(refinement, dict) else None

    if beta_mem < 1.0:
        return _flow_caputo(canonical_initial, nu, refinement)

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


def _flow_caputo(
    canonical_initial: CanonicalState,
    nu: float,
    refinement: dict,
) -> CanonicalState:
    """v2.4 non-Markovian Caputo flow via Prony sum-of-exponentials.

        chit(ν)     = chit_0     * Σ_k a_k exp(-b_k λ_chit  ν)
        gamma_AB(ν) = gamma_AB_0 * Σ_k a_k exp(-b_k λ_gamma ν)

    `prony_terms` is a list of `(amplitude, decay-rate)` tuples
    pre-fit by the curator (mpa-conform's curator path). The v2.4 solver
    consumes them; on-the-fly Mittag-Leffler fitting is a curator-path
    job.

    Missing `prony_terms` raises ValueError — `beta_mem < 1` without an
    accompanying kernel is a malformed refinement.
    """
    prony_terms = refinement.get("prony_terms")
    if not prony_terms:
        raise ValueError(
            "beta_mem < 1.0 requires prony_terms in refinement "
            "(curator-supplied Mittag-Leffler approximation)."
        )
    lambda_chit = float(refinement.get("lambda_chit", 1.0))
    lambda_gamma = float(refinement.get("lambda_gamma", 1.0))
    chit_kernel = sum(a * math.exp(-b * lambda_chit * nu) for a, b in prony_terms)
    gamma_kernel = sum(a * math.exp(-b * lambda_gamma * nu) for a, b in prony_terms)
    return CanonicalState(
        chit=canonical_initial.chit * chit_kernel,
        gamma_AB=canonical_initial.gamma_AB * gamma_kernel,
        k_frust=canonical_initial.k_frust,
    )
