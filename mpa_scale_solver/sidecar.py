"""Inverse-lookup-table sidecar dispatch (handoff §C.4).

The sidecar is a curator-precomputed table that lets `forward_sweep_invert`
short-circuit the brute-force grid search when the (substrate, tau_obs)
pair is in the table. This module provides the dispatch helpers; sidecar
*production* is mpa-conform's curator-path responsibility. The Banach
sidecar (`banach.BanachSubstrate.build_sidecar`) is the v1 reference
producer used by the camera test.

Dispatch contract:

- Round `(chit, gamma_AB, tau_obs)` keys to a fixed decimal precision so
  lookups survive float churn. Default = 6 decimals; producers and
  consumers must agree on the precision (carried implicitly by the
  sidecar's construction).
- On a hit, return the canonical state recorded for that key.
- On a miss, return None — the caller falls through to the compute path.
"""

from __future__ import annotations

from typing import Optional

from .types import CanonicalState, InverseLookupSidecar, SubstrateState


# Default key-rounding precision. Producers and consumers must agree;
# `BanachSubstrate.build_sidecar` uses this value.
DEFAULT_ROUNDING_DECIMALS: int = 6


def round_key(
    key: tuple[float, float, float],
    decimals: int = DEFAULT_ROUNDING_DECIMALS,
) -> tuple[float, float, float]:
    """Round a 3-tuple key to the agreed precision."""
    return (round(key[0], decimals), round(key[1], decimals), round(key[2], decimals))


def lookup_inverse(
    sidecar: InverseLookupSidecar,
    substrate: SubstrateState,
    tau_obs: float,
    *,
    decimals: int = DEFAULT_ROUNDING_DECIMALS,
) -> Optional[CanonicalState]:
    """Table-first inverse lookup.

    Returns the recorded canonical state if `(substrate, tau_obs)` is in
    the sidecar's inverse table; None on miss. Callers fall through to
    the compute path on None.

    Substrate-side keying uses `observables['substrate_chit']` /
    `observables['substrate_gamma_AB']` — the canonical curator
    convention. Substrates without those keys are a guaranteed miss.
    """
    obs = substrate.observables
    chit = obs.get("substrate_chit")
    gamma = obs.get("substrate_gamma_AB")
    if chit is None or gamma is None:
        return None
    key = round_key((float(chit), float(gamma), float(tau_obs)), decimals)
    return sidecar.inverse_lookup.get(key)


def lookup_forward(
    sidecar: InverseLookupSidecar,
    canonical: CanonicalState,
    tau_obs: float,
    *,
    decimals: int = DEFAULT_ROUNDING_DECIMALS,
) -> Optional[SubstrateState]:
    """Table-first forward lookup."""
    key = round_key(
        (float(canonical.chit), float(canonical.gamma_AB), float(tau_obs)),
        decimals,
    )
    return sidecar.forward_lookup.get(key)
