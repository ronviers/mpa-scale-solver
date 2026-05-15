"""Synthetic parametric rules used ONLY by the camera test.

NOT the production translation-field shape (handoff §C.2 step 1). The v2.0
schema is forward-only / lookup_table; these closed-form rules exist to (a)
furnish analytical truth the camera test validates against, and (b) supply
a substrate-side observable a window-averaged signal can be compared to.

Public-API consumers do not import this module. It is exposed for tests only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .operations import apply_translation
from .types import (
    CanonicalPoint,
    CanonicalState,
    OperatingPoint,
    SubstrateState,
    TranslationField,
    TranslationRule,
)


@dataclass(frozen=True)
class AgingLogParams:
    """Closed-form synthetic aging rule (handoff §C.2 step 3 reference).

    substrate_chit  = canonical_chit + a * log(1 + tau_obs / tau_aging)
    substrate_gamma = canonical_gamma_AB * (1 + b * log(1 + tau_obs / tau_aging))

    For a fixed substrate observation S:
        canonical_chit(tau_obs)  = S_chit - a * log(1 + tau_obs / tau_aging)
        canonical_gamma(tau_obs) = S_gamma / (1 + b * log(1 + tau_obs / tau_aging))

    This is the analytical truth the camera test scores against.
    """

    chit_aging_coeff: float = 1.0
    tau_aging: float = 1.0
    gamma_aging_coeff: float = 0.0


def aging_log_forward(
    canonical: CanonicalState,
    tau_obs: float,
    params: AgingLogParams,
) -> SubstrateState:
    """Analytical forward map for the aging_log fixture.

    Returns a SubstrateState whose `observables` carry the closed-form
    substrate-side prediction. Used directly as a `forward_map` callable
    by forward_sweep_invert in the camera test.
    """
    drift = math.log1p(tau_obs / params.tau_aging)
    sub_chit = canonical.chit + params.chit_aging_coeff * drift
    sub_gamma = canonical.gamma_AB * (1.0 + params.gamma_aging_coeff * drift)
    return SubstrateState(
        tau_obs=tau_obs,
        label=None,
        axes={},
        observables={"substrate_chit": sub_chit, "substrate_gamma_AB": sub_gamma},
    )


def analytical_canonical_chit(
    substrate_chit: float,
    tau_obs: float,
    params: AgingLogParams,
) -> float:
    """Analytical canonical chit at tau_obs for fixed substrate observation."""
    return substrate_chit - params.chit_aging_coeff * math.log1p(tau_obs / params.tau_aging)


def analytical_canonical_gamma(
    substrate_gamma: float,
    tau_obs: float,
    params: AgingLogParams,
) -> float:
    """Analytical canonical gamma_AB at tau_obs for fixed substrate."""
    drift = math.log1p(tau_obs / params.tau_aging)
    return substrate_gamma / (1.0 + params.gamma_aging_coeff * drift)


def make_aging_log_lookup_table(
    chit_grid: np.ndarray,
    gamma_AB_grid: np.ndarray,
    tau_obs_grid: np.ndarray,
    params: AgingLogParams,
    *,
    method_str: Optional[str] = None,
) -> TranslationField:
    """Sample the analytical aging_log rule on a (chit, gamma_AB, tau_obs) grid
    and assemble it as a lookup_table TranslationField.

    Per handoff §C.2 step 3: this gives the camera test a production-shape
    field to exercise apply_translation against, while leaving the
    analytical truth available for residual scoring.
    """
    method = method_str or (
        f"aging_log fixture (a={params.chit_aging_coeff}, "
        f"tau_aging={params.tau_aging}, b={params.gamma_aging_coeff})"
    )
    rules: list[TranslationRule] = []
    for chit in chit_grid:
        for gamma in gamma_AB_grid:
            for tau in tau_obs_grid:
                drift = math.log1p(float(tau) / params.tau_aging)
                sub_chit = float(chit) + params.chit_aging_coeff * drift
                sub_gamma = float(gamma) * (1.0 + params.gamma_aging_coeff * drift)
                op = OperatingPoint(
                    label=f"chit={float(chit):+.4f} gamma={float(gamma):+.4f} tau={float(tau):.4g}",
                    gt="s",  # placeholder; the camera test does not consult gt
                    axes={
                        "tau_obs": float(tau),
                        "substrate_chit": sub_chit,
                        "substrate_gamma_AB": sub_gamma,
                    },
                )
                canonical = CanonicalPoint(
                    chit=float(chit),
                    gamma_AB=float(gamma),
                    k_frust=False,
                    method=method,
                )
                rules.append(TranslationRule(
                    operating_point=op,
                    xdot_choice="synthetic",
                    canonical=canonical,
                ))
    return TranslationField(
        direction="forward",
        shape="lookup_table",
        rule=rules,
        description="Synthetic aging_log lookup-table fixture (camera test).",
    )
