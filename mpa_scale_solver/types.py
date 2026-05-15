"""Plain dataclasses for the scale-solver.

Per §A.3 of the build handoff: stateless free functions on plain dataclasses.
Type hints everywhere; numpy arrays only for vectorized ops, never for stored
state. Direct-port shape for the eventual Rust/C++ build.

Two pairs of canonical types live here, distinguished by where they ride:

- `CanonicalPoint` / `OperatingPoint` / `TranslationRule` / `TranslationField`
  mirror the driver-profile.v2.0 JSON schema. They are the on-disk schema
  cells the curator path emits and this solver consumes.
- `CanonicalState` / `SubstrateState` are the runtime working states. The
  state pair carries no tau_obs (tau_obs is a free argument on every
  operation, never embedded), matching the §A.3 commitment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional


# ---------------------------------------------------------------------------
# Runtime working states (§A.3)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CanonicalState:
    """Canonical-frame state at the call-site's tau_obs.

    tau_obs is NOT stored on the state. Every operation that needs it takes
    it as an explicit argument. This keeps the state pair substrate-neutral
    and matches the handoff §A.3 stateless commitment.
    """

    chit: float
    gamma_AB: float
    k_frust: bool = False


@dataclass(frozen=True)
class SubstrateState:
    """Substrate-native observation at one tau_obs frame.

    Driver-profile-conditional shape. The lookup_table production path
    populates `label` (matched operating-point label) and `axes` (the
    operating-point's substrate-side axis values). The synthetic parametric
    test path populates `observables` (scalar chit/gamma estimates with an
    analytical forward map). Both populate `tau_obs` for trace-ability.
    """

    tau_obs: float
    label: Optional[str] = None
    axes: dict[str, Any] = field(default_factory=dict)
    observables: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Schema dataclasses (driver-profile.v2.0)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CanonicalPoint:
    """Canonical-coordinate target a TranslationRule projects to.

    Mirrors driver-profile.v2.0 #/$defs/canonical_point. The four named
    fields are co-required by the seed-corpus convention. Additional cdv1
    API-slot values (per mpa-auditor §Q6) ride `extras`.
    """

    chit: float
    gamma_AB: float
    k_frust: bool
    method: str
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OperatingPoint:
    """A cell in the substrate's operating envelope.

    Mirrors driver-profile.v2.0 #/$defs/operating_point. `label` and `gt`
    are cross-substrate; substrate-specific axes (T, p_base, h_field,
    scenario, ...) ride `axes`. Union shape with nulls is valid; omitting
    irrelevant keys is also valid.
    """

    label: str
    gt: Literal["c", "s", "r", "k"]
    axes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TranslationRule:
    operating_point: OperatingPoint
    xdot_choice: str
    canonical: CanonicalPoint


@dataclass(frozen=True)
class TranslationField:
    """Forward-only canonical -> substrate-native map (RFC-S §4, §Q13).

    `direction` and `shape` are Literal type pins, not enums with branches.
    Forward-only is §Q13 (the backward map is structurally ill-posed).
    lookup_table is RFC-S Appendix B item 1 deferral (tangent-flow form is
    v3, not v2).
    """

    direction: Literal["forward"]
    shape: Literal["lookup_table"]
    rule: list[TranslationRule]
    description: Optional[str] = None


# ---------------------------------------------------------------------------
# Gamut / regime
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GamutSpec:
    """Substrate gamut: image of the RG trajectory in canonical space (RFC-S §2).

    chit_range and gamma_AB_range describe the canonical-space envelope the
    declared driver profile covers. tau_obs_range bounds the camera frames
    the profile is valid for. out_of_scope_residual_threshold is the
    per-class miss tolerance (replaces the auditor's M6 stand-in).
    """

    chit_range: tuple[float, float]
    gamma_AB_range: tuple[float, float]
    tau_obs_range: Optional[tuple[float, float]] = None
    out_of_scope_residual_threshold: float = 0.05


RegimeLabel = Literal["deep_c", "c_near_s", "s_critical", "r_near_s", "deep_r"]
DisplayBand = Literal["c", "s", "r"]


@dataclass(frozen=True)
class RegimeReading:
    """Five-bucket vertex regime at a tau_obs frame (handoff §C.4).

    The 5-bucket label is the canonical classifier (matches the auditor's
    gfdr-model.js). `regime_display_band` in operations collapses to the
    3-bucket display-only band when the renderer needs it.
    """

    regime: RegimeLabel
    k_frust: bool = False
