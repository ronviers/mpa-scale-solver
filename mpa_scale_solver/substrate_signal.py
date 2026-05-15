"""Synthetic substrate-side K(t) generator + per-frame window-average.

Used only by the camera test (handoff §A.5: the EXR-encoding camera test
is a test harness, not a solver operation). The substrate K(t) here is
the synthetic time-series whose window-averaged observable, viewed through
a tau_obs camera, yields the c -> s -> r migration the framework predicts.

Per handoff §C.1: the window average is per-frame. Each tau_obs gets its
own window. The function name `window_average_at_tau_obs` makes this
explicit so callers cannot accidentally treat it as a single global
operation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .types import SubstrateState


@dataclass(frozen=True)
class AgingSignalParams:
    """Synthetic substrate-side aging signal.

    K(t) is constant in time at value `substrate_chit` plus a small fluctuation;
    `substrate_chit` is what the windowed observation reports at every frame
    (the synthetic-aging substrate's "raw" observable is stationary, but the
    canonical state changes because the tau_obs camera changes).

    For the camera test the substrate signal is built so that the windowed
    observable equals the aging_log forward-projection of the reference
    canonical state at the reference tau_obs.
    """

    substrate_chit: float
    substrate_gamma_AB: float
    fluctuation_amplitude: float = 0.0  # zero by default for deterministic tests
    fluctuation_seed: int = 0


def window_average_at_tau_obs(
    signal: AgingSignalParams,
    tau_obs: float,
) -> SubstrateState:
    """Produce the substrate observation at a single tau_obs frame.

    Per handoff §C.1: this is a SINGLE-FRAME operation. The s -> r migration
    test calls it once per tau_obs in a sweep, NOT once for the whole
    substrate. The name is deliberate.

    For the synthetic aging signal the window-averaged observable is
    independent of tau_obs (the raw signal is stationary). A non-stationary
    substrate would have tau_obs-dependent window means; that's how the
    real curator-path uses this function.
    """
    # Optional reproducible fluctuation
    if signal.fluctuation_amplitude > 0.0:
        rng = np.random.default_rng(signal.fluctuation_seed + int(tau_obs * 1e6))
        chit_obs = signal.substrate_chit + signal.fluctuation_amplitude * rng.normal()
        gamma_obs = signal.substrate_gamma_AB + signal.fluctuation_amplitude * rng.normal()
    else:
        chit_obs = signal.substrate_chit
        gamma_obs = signal.substrate_gamma_AB

    return SubstrateState(
        tau_obs=tau_obs,
        label=None,
        axes={"tau_obs": tau_obs},
        observables={
            "substrate_chit": float(chit_obs),
            "substrate_gamma_AB": float(gamma_obs),
        },
    )


def sweep_window_average(
    signal: AgingSignalParams,
    tau_obs_grid: np.ndarray,
) -> list[SubstrateState]:
    """Per-frame window averages across a tau_obs sweep (handoff §B.2)."""
    return [window_average_at_tau_obs(signal, float(t)) for t in tau_obs_grid]
