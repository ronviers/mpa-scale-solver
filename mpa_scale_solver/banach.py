"""Banach substrate — calibration reference (handoff §C.3).

The Banach substrate is the canonical reference instance of the cdv1
universal two-mode kernel with parameters at framework-default values
and identity translation field. The camera-test fixture for v1.

The canonical RG flow is the closed-form exponential decay agreed in Q1
of the v1 build session:

    chit(nu)    = chit_0 * exp(-lambda_chit * nu)
    gamma_AB(nu) = gamma_AB_0 * exp(-lambda_gamma * nu)

Default rates `lambda_chit = lambda_gamma = 1.0` are the v1 normalization
(equivalent to spectral-gap eigenvalue exp(-1) of `ln C`; v2 derives these
from `flow_spectrum` via the closed Wilson-Kadanoff construction in
v9_receipts §RG closure).

The substrate's translation field is identity (substrate-native equivalent
to canonical at every nu); the RG flow happens entirely in canonical
space and is read off by `state_at`. See `docs/BANACH_SUBSTRATE.md`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .types import (
    CanonicalPoint,
    CanonicalState,
    InverseLookupSidecar,
    OperatingPoint,
    ScalingRule,
    SubstrateState,
    TangentFlowField,
    TranslationRule,
)


# Default Banach kinetic rates (Q1 v1 normalization).
DEFAULT_LAMBDA_CHIT: float = 1.0
DEFAULT_LAMBDA_GAMMA: float = 1.0

# Canonical-initial defaults: c-band start, cooperative gamma, so the
# trajectory traverses the full c -> s -> r migration interior as nu sweeps.
DEFAULT_CHIT_0: float = 1.5
DEFAULT_GAMMA_AB_0: float = -0.5


@dataclass(frozen=True)
class BanachSubstrate:
    """Banach calibration substrate.

    Plain frozen dataclass. Methods are pure functions of the stored
    parameters — no mutable state, no global state. Compatible with the
    handoff §A.3 stateless commitment.
    """

    chit_0: float = DEFAULT_CHIT_0
    gamma_AB_0: float = DEFAULT_GAMMA_AB_0
    lambda_chit: float = DEFAULT_LAMBDA_CHIT
    lambda_gamma: float = DEFAULT_LAMBDA_GAMMA
    tau_obs_ref: float = 1.0
    k_frust: bool = False

    # -----------------------------------------------------------------
    # Canonical surface
    # -----------------------------------------------------------------

    def canonical_initial(self) -> CanonicalState:
        """Canonical state at nu=0."""
        return CanonicalState(
            chit=self.chit_0,
            gamma_AB=self.gamma_AB_0,
            k_frust=self.k_frust,
        )

    def state_at(self, nu: float) -> CanonicalState:
        """Analytical canonical state at depth nu (framework truth).

        Closed-form exponential decay agreed in Q1 of the v1 build
        session. Asymptotic-Closure-compliant: chit / gamma_AB approach
        0 as nu -> infinity but never reach it at any finite nu.
        """
        return CanonicalState(
            chit=self.chit_0 * math.exp(-self.lambda_chit * nu),
            gamma_AB=self.gamma_AB_0 * math.exp(-self.lambda_gamma * nu),
            k_frust=self.k_frust,
        )

    # -----------------------------------------------------------------
    # Translation field
    # -----------------------------------------------------------------

    def translation_field(self) -> TangentFlowField:
        """Tangent-flow field with identity scaling + banach_exponential
        refinement.

        Identity at the translation step (`delta_gamma = delta_chit = 0`):
        substrate-native equivalent to canonical, no per-frame drift.
        The refinement dict carries the closed-form flow generator so
        `flow()` returns the analytical truth.
        """
        ref_state = CanonicalState(
            chit=self.chit_0,
            gamma_AB=self.gamma_AB_0,
            k_frust=self.k_frust,
        )
        rule_at_origin = TranslationRule(
            operating_point=OperatingPoint(
                label="banach_origin",
                gt="c" if self.chit_0 >= 0.2 else ("s" if self.chit_0 > -0.2 else "r"),
                axes={"tau_obs": self.tau_obs_ref},
            ),
            xdot_choice="identity",
            canonical=CanonicalPoint(
                chit=self.chit_0,
                gamma_AB=self.gamma_AB_0,
                k_frust=self.k_frust,
                method="banach_canonical",
            ),
        )
        scaling = ScalingRule(
            tau_obs_ref=self.tau_obs_ref,
            delta_gamma=0.0,
            delta_chit=0.0,
            refinement={
                "flow_kind": "banach_exponential",
                "lambda_chit": self.lambda_chit,
                "lambda_gamma": self.lambda_gamma,
            },
        )
        return TangentFlowField(
            direction="forward",
            shape="tangent_flow",
            rule_at_origin=rule_at_origin,
            scaling=scaling,
            description=(
                "Banach substrate: identity translation + exponential "
                "canonical flow (Q1 v1 normalization)."
            ),
        )

    # -----------------------------------------------------------------
    # Substrate-side observable (for camera test)
    # -----------------------------------------------------------------

    def substrate_at(self, nu: float) -> SubstrateState:
        """Substrate observation at depth nu.

        Identity translation: substrate.observables = state_at(nu)'s
        canonical values. This is what the camera test passes as the
        per-frame target.
        """
        s = self.state_at(nu)
        return SubstrateState(
            tau_obs=nu,
            label="banach",
            axes={"tau_obs": nu},
            observables={
                "substrate_chit": s.chit,
                "substrate_gamma_AB": s.gamma_AB,
                "chit": s.chit,
                "gamma_AB": s.gamma_AB,
            },
        )

    def trajectory(self, tau_obs_grid: np.ndarray) -> list[CanonicalState]:
        """Analytical canonical trajectory across the tau_obs grid."""
        return [self.state_at(float(nu)) for nu in tau_obs_grid]

    # -----------------------------------------------------------------
    # Sidecar builder
    # -----------------------------------------------------------------

    def build_sidecar(
        self,
        tau_obs_grid: np.ndarray,
        *,
        version: str = "1.0.0",
        rounding_decimals: int = 6,
    ) -> InverseLookupSidecar:
        """Construct an InverseLookupSidecar from the Banach analytical truth.

        Curator-path production of sidecars is mpa-conform's job; this
        helper exists so the v1 Banach camera test can exercise the
        table-first path without leaving the solver.

        Forward keys are (canonical.chit, canonical.gamma_AB, tau_obs)
        rounded to `rounding_decimals` decimals; inverse keys are
        (substrate observable chit, gamma, tau_obs) — same numbers
        because the translation is identity.
        """
        from .sidecar import round_key  # local import to avoid cycle

        nus = [float(t) for t in tau_obs_grid]
        canonicals = [self.state_at(nu) for nu in nus]
        substrates = [self.substrate_at(nu) for nu in nus]

        forward_lookup: dict[tuple[float, float, float], SubstrateState] = {}
        inverse_lookup: dict[tuple[float, float, float], CanonicalState] = {}
        for nu, c, s in zip(nus, canonicals, substrates):
            fkey = round_key((c.chit, c.gamma_AB, nu), rounding_decimals)
            forward_lookup[fkey] = s
            ikey = round_key(
                (s.observables["substrate_chit"],
                 s.observables["substrate_gamma_AB"],
                 nu),
                rounding_decimals,
            )
            inverse_lookup[ikey] = c

        return InverseLookupSidecar(
            version=version,
            driver_profile_id="banach",
            driver_profile_version="1.0.0",
            tau_obs_grid=tuple(nus),
            substrate_grid=tuple(substrates),
            canonical_grid=tuple(canonicals),
            forward_lookup=forward_lookup,
            inverse_lookup=inverse_lookup,
            ambiguity_regions=(),
        )


def build_sidecar_for_banach(
    substrate: Optional[BanachSubstrate] = None,
    *,
    tau_obs_grid: Optional[np.ndarray] = None,
    version: str = "1.0.0",
) -> InverseLookupSidecar:
    """Convenience wrapper: build a Banach sidecar on a default tau_obs grid.

    Defaults: BanachSubstrate() and `numpy.logspace(-2, 4, 80)`.
    """
    substrate = substrate or BanachSubstrate()
    grid = tau_obs_grid if tau_obs_grid is not None else np.logspace(-2, 4, 80)
    return substrate.build_sidecar(grid, version=version)
