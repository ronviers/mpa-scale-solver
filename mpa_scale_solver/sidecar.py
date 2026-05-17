"""Inverse-lookup-table sidecar dispatch (handoff §C.4).

The sidecar is a curator-precomputed table that lets `forward_sweep_invert`
short-circuit the brute-force grid search when the (substrate, tau_obs)
pair is in the table. This module provides the dispatch helpers + the
JSON wire-format encode/decode helpers (the cross-language artifact
contract); sidecar *production* is mpa-conform's curator-path
responsibility. The Banach sidecar (`banach.BanachSubstrate.build_sidecar`)
is the v1 reference producer used by the camera test.

Dispatch contract:

- Round `(chit, gamma_AB, tau_obs)` keys to a fixed decimal precision so
  lookups survive float churn. Default = 6 decimals; producers and
  consumers must agree on the precision (carried on the sidecar's
  `rounding_decimals` field).
- Rounding algorithm is `float(np.rint(x * 10**n)) / 10**n` — IEEE-754
  `roundTiesToEven` on the scaled value. This is bit-deterministic and
  matches the Rust port's `(x * 10^n).round_ties_even() / 10^n` exactly,
  so cross-language sidecar round-trips are bit-identical (see
  `docs/SIDECAR_FORMAT.md`). The previous `round(x, n)` Python builtin
  used dtoa-based decimal rounding which could diverge from Rust on
  `.x5`-decimal binary halfway cases; the np.rint form fixes that.
- On a hit, return the canonical state recorded for that key.
- On a miss, return None — the caller falls through to the compute path.

JSON wire format (see SIDECAR_FORMAT.md for the authoritative spec):

- Top-level: `{wire_version, version, driver_profile_id,
  driver_profile_version, rounding_decimals, tau_obs_grid,
  substrate_grid, canonical_grid, forward_lookup, inverse_lookup,
  ambiguity_regions}`.
- Lookup-dict keys are `':'`-joined `u64` decimal strings of
  `f64::to_bits` of the rounded (chit, gamma_AB, tau_obs) floats — the
  bit-exact lossless serialization of the rounded value.
- `encode_sidecar_to_json` produces this shape from an in-memory
  `InverseLookupSidecar`; `decode_sidecar_from_json` is the inverse.
"""

from __future__ import annotations

import struct
from typing import Any, Optional

import numpy as np

from .types import (
    CanonicalState,
    InverseLookupSidecar,
    SubstrateState,
)


# Default key-rounding precision. Producers and consumers must agree;
# the active precision rides on `InverseLookupSidecar.rounding_decimals`
# so consumers can recover it without out-of-band assumptions.
DEFAULT_ROUNDING_DECIMALS: int = 6

# Current wire-format version. Bump on any incompatible JSON-shape
# change (e.g., renamed top-level fields). Additive changes (new
# optional fields with serde defaults on the Rust side) do NOT bump
# this string; consumers ignore unknown additive keys.
WIRE_VERSION: str = "1.0"


# ---------------------------------------------------------------------------
# Rounding
# ---------------------------------------------------------------------------


def round_key(
    key: tuple[float, float, float],
    decimals: int = DEFAULT_ROUNDING_DECIMALS,
) -> tuple[float, float, float]:
    """Round a 3-tuple key to the agreed precision.

    Uses `float(np.rint(x * 10**n)) / 10**n` — IEEE-754
    `roundTiesToEven` on the scaled value. Bit-deterministic; matches
    the Rust port's `(x * 10^n).round_ties_even() / 10^n` exactly so
    Python→Rust and Rust→Python sidecar round-trips are bit-identical.
    """
    return (
        _round_ties_even_decimal(key[0], decimals),
        _round_ties_even_decimal(key[1], decimals),
        _round_ties_even_decimal(key[2], decimals),
    )


def _round_ties_even_decimal(x: float, decimals: int) -> float:
    if not np.isfinite(x):
        return float(x)
    scale = 10.0 ** decimals
    return float(np.rint(x * scale)) / scale


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------


def lookup_inverse(
    sidecar: InverseLookupSidecar,
    substrate: SubstrateState,
    tau_obs: float,
    *,
    decimals: Optional[int] = None,
) -> Optional[CanonicalState]:
    """Table-first inverse lookup.

    Returns the recorded canonical state if `(substrate, tau_obs)` is in
    the sidecar's inverse table; None on miss. Callers fall through to
    the compute path on None.

    Substrate-side keying uses `observables['substrate_chit']` /
    `observables['substrate_gamma_AB']` — the canonical curator
    convention. Substrates without those keys are a guaranteed miss.

    `decimals` defaults to the sidecar's recorded `rounding_decimals`
    so the consumer rounds at the same precision the producer used.
    """
    obs = substrate.observables
    chit = obs.get("substrate_chit")
    gamma = obs.get("substrate_gamma_AB")
    if chit is None or gamma is None:
        return None
    n = sidecar.rounding_decimals if decimals is None else decimals
    key = round_key((float(chit), float(gamma), float(tau_obs)), n)
    return sidecar.inverse_lookup.get(key)


def lookup_forward(
    sidecar: InverseLookupSidecar,
    canonical: CanonicalState,
    tau_obs: float,
    *,
    decimals: Optional[int] = None,
) -> Optional[SubstrateState]:
    """Table-first forward lookup. See `lookup_inverse` for the
    `decimals` semantics."""
    n = sidecar.rounding_decimals if decimals is None else decimals
    key = round_key(
        (float(canonical.chit), float(canonical.gamma_AB), float(tau_obs)),
        n,
    )
    return sidecar.forward_lookup.get(key)


# ---------------------------------------------------------------------------
# Wire-format encode / decode — JSON parity with Rust
# ---------------------------------------------------------------------------


def _float_to_bits(x: float) -> int:
    """`f64.to_bits` — IEEE-754 binary representation as an unsigned int."""
    return struct.unpack("<Q", struct.pack("<d", float(x)))[0]


def _bits_to_float(b: int) -> float:
    """Inverse of `_float_to_bits`."""
    return struct.unpack("<d", struct.pack("<Q", int(b)))[0]


def encode_sidecar_key(chit: float, gamma_AB: float, tau_obs: float) -> str:
    """Encode a 3-tuple of (rounded) floats as the wire-format key string.

    Format: `'<chit_bits>:<gamma_AB_bits>:<tau_obs_bits>'` where each
    `*_bits` is the decimal string of `f64::to_bits(x)`. Matches
    `rust/src/types.rs::SidecarKey::serialize` exactly.

    Callers MUST pass already-rounded floats. `round_key` is the
    canonical rounding step.
    """
    return f"{_float_to_bits(chit)}:{_float_to_bits(gamma_AB)}:{_float_to_bits(tau_obs)}"


def decode_sidecar_key(s: str) -> tuple[float, float, float]:
    """Inverse of `encode_sidecar_key`. Recovers the rounded float
    values losslessly."""
    parts = s.split(":")
    if len(parts) != 3:
        raise ValueError(
            f"SidecarKey: expected 3 ':'-separated u64 parts, got {len(parts)}"
        )
    return (_bits_to_float(int(parts[0])),
            _bits_to_float(int(parts[1])),
            _bits_to_float(int(parts[2])))


def _encode_canonical(c: CanonicalState) -> dict[str, Any]:
    return {"chit": c.chit, "gamma_AB": c.gamma_AB, "k_frust": c.k_frust}


def _decode_canonical(d: dict[str, Any]) -> CanonicalState:
    return CanonicalState(
        chit=float(d["chit"]),
        gamma_AB=float(d["gamma_AB"]),
        k_frust=bool(d.get("k_frust", False)),
    )


def _encode_substrate(s: SubstrateState) -> dict[str, Any]:
    return {
        "tau_obs": s.tau_obs,
        "label": s.label,
        "axes": dict(s.axes),
        "observables": dict(s.observables),
    }


def _decode_substrate(d: dict[str, Any]) -> SubstrateState:
    return SubstrateState(
        tau_obs=float(d["tau_obs"]),
        label=d.get("label"),
        axes=dict(d.get("axes", {})),
        observables={k: float(v) for k, v in dict(d.get("observables", {})).items()},
    )


def encode_sidecar_to_json(sidecar: InverseLookupSidecar) -> dict[str, Any]:
    """Encode an `InverseLookupSidecar` as a JSON-ready dict per the
    `docs/SIDECAR_FORMAT.md` wire-format spec.

    The lookup dicts emit keys via `encode_sidecar_key` (`':'`-joined
    `f64::to_bits` decimal strings). The result is `json.dumps`-ready
    and parses byte-losslessly back into `InverseLookupSidecar` via
    `decode_sidecar_from_json`. The Rust port's
    `serde_json::from_str::<InverseLookupSidecar>` consumes the same
    shape with `wire_version` / `rounding_decimals` serde defaults
    matching the constants here.
    """
    def k_str(key: tuple[float, float, float]) -> str:
        return encode_sidecar_key(key[0], key[1], key[2])

    return {
        "wire_version": sidecar.wire_version,
        "version": sidecar.version,
        "driver_profile_id": sidecar.driver_profile_id,
        "driver_profile_version": sidecar.driver_profile_version,
        "rounding_decimals": sidecar.rounding_decimals,
        "tau_obs_grid": list(sidecar.tau_obs_grid),
        "substrate_grid": [_encode_substrate(s) for s in sidecar.substrate_grid],
        "canonical_grid": [_encode_canonical(c) for c in sidecar.canonical_grid],
        "forward_lookup": {
            k_str(k): _encode_substrate(v) for k, v in sidecar.forward_lookup.items()
        },
        "inverse_lookup": {
            k_str(k): _encode_canonical(v) for k, v in sidecar.inverse_lookup.items()
        },
        "ambiguity_regions": [dict(r) for r in sidecar.ambiguity_regions],
    }


def decode_sidecar_from_json(d: dict[str, Any]) -> InverseLookupSidecar:
    """Inverse of `encode_sidecar_to_json`. Reads a JSON-shaped dict
    and reconstructs the in-memory `InverseLookupSidecar`.

    Unknown / missing optional fields default per the spec:
      - `wire_version` defaults to `WIRE_VERSION` (currently "1.0").
      - `rounding_decimals` defaults to `DEFAULT_ROUNDING_DECIMALS` (6).
      - `ambiguity_regions` defaults to empty.

    Unknown extra keys are ignored — this is forward-compat: a future
    additive wire version can land new optional fields without
    breaking the present consumer.
    """
    wire_version = str(d.get("wire_version", WIRE_VERSION))

    forward_lookup: dict[tuple[float, float, float], SubstrateState] = {}
    for k, v in dict(d.get("forward_lookup", {})).items():
        key = decode_sidecar_key(k)
        forward_lookup[key] = _decode_substrate(v)

    inverse_lookup: dict[tuple[float, float, float], CanonicalState] = {}
    for k, v in dict(d.get("inverse_lookup", {})).items():
        key = decode_sidecar_key(k)
        inverse_lookup[key] = _decode_canonical(v)

    return InverseLookupSidecar(
        version=str(d["version"]),
        driver_profile_id=str(d["driver_profile_id"]),
        driver_profile_version=str(d["driver_profile_version"]),
        tau_obs_grid=tuple(float(x) for x in d.get("tau_obs_grid", ())),
        substrate_grid=tuple(_decode_substrate(s) for s in d.get("substrate_grid", ())),
        canonical_grid=tuple(_decode_canonical(c) for c in d.get("canonical_grid", ())),
        forward_lookup=forward_lookup,
        inverse_lookup=inverse_lookup,
        ambiguity_regions=tuple(dict(r) for r in d.get("ambiguity_regions", ())),
        wire_version=wire_version,
        rounding_decimals=int(d.get("rounding_decimals", DEFAULT_ROUNDING_DECIMALS)),
    )
