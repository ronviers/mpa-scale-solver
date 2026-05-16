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

v1 additions (handoff §B.4, §C.4, §C.5, §C.6):

- `TangentFlowField` + `ScalingRule` add the second translation-field shape
  (RFC-S Appendix B item 1). `TranslationField` remains the lookup-table
  dataclass for back-compat; `LookupTableField` is an alias for the
  handoff-spelled name; `AnyTranslationField` is the union type the v1
  operations accept.
- `OperationOutput[T]` / `ValidationReport` / `Provenance` / `DispatchPath`
  ride with the wrapped-variant operations (`*_wrapped`).
- `InverseLookupSidecar` is the curator-precomputed dispatch fast-path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, Literal, Optional, TypeVar, Union


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
    """Lookup-table translation field (RFC-S §4, §Q13).

    `direction` and `shape` are Literal type pins. Forward-only is §Q13
    (the backward map is structurally ill-posed). The lookup-table form is
    the v0 production shape; v1 adds the tangent-flow form as a sibling.

    The v0 name `TranslationField` is preserved verbatim for back-compat.
    The handoff-spelled name `LookupTableField` is exposed below as an
    alias. The Union accepted by v1 operations is `AnyTranslationField`.
    """

    direction: Literal["forward"]
    shape: Literal["lookup_table"]
    rule: list[TranslationRule]
    description: Optional[str] = None


# v1: handoff-spelled name. Same class, two names — keeps v0 constructors
# working (`TranslationField(direction="forward", shape="lookup_table", ...)`)
# while introducing the disambiguated form for new code.
LookupTableField = TranslationField


@dataclass(frozen=True)
class ScalingRule:
    """Banach-canonical leading-order tangent-flow rule (handoff §B.4).

        gamma(tau_obs) = gamma_initial * (tau_obs / tau_obs_ref) ** delta_gamma
        chit(tau_obs)  = chit_initial + delta_chit * ln(tau_obs / tau_obs_ref)

    Default delta_gamma = delta_chit = 0.0 is identity scaling (the Banach
    substrate's translation-field defaults). Substrate-conditional
    refinements ride in `refinement`.
    """

    tau_obs_ref: float
    delta_gamma: float = 0.0
    delta_chit: float = 0.0
    refinement: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class TangentFlowField:
    """Tangent-flow translation field (handoff §B.4 / §C.2).

    Closed-form sibling of `TranslationField` (lookup_table). The scaling
    rule carries the leading-order auto-remap derivatives at the reference
    point; `rule_at_origin` pins the substrate-side mapping at the
    reference point so the substrate identity is well-defined.
    """

    direction: Literal["forward"]
    shape: Literal["tangent_flow"]
    rule_at_origin: TranslationRule
    scaling: ScalingRule
    description: Optional[str] = None


# v1: the union the operations accept. Per the handoff §A.4 the
# seven-operation API surface still takes "a translation field" — both
# shapes ride the same callsites, dispatched on `field.shape`.
AnyTranslationField = Union[TranslationField, TangentFlowField]


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


# ---------------------------------------------------------------------------
# v1: provenance + dispatch-path
# ---------------------------------------------------------------------------

class DispatchPath(str, Enum):
    """Which path the operation took (handoff §C.6).

    Recorded on every `Provenance` so consumers (mpa-conform's audit
    record, mpa-auditor's display layer) can distinguish table-hits from
    compute-fallbacks without re-running the operation.
    """

    TABLE_HIT = "table_hit"          # sidecar lookup succeeded
    COMPUTE_FALLBACK = "compute_fallback"   # sidecar missed; brute force ran
    DIRECT_COMPUTE = "direct_compute"       # no sidecar consulted


@dataclass(frozen=True)
class Provenance:
    """Per-call provenance trail (handoff §C.6).

    Populated by each `*_wrapped` operation and consumed by mpa-conform
    when assembling the bundle's audit record. Fields are intentionally
    primitive types so the record serializes cleanly into the bundle.
    """

    solver_version: str
    operation: str
    timestamp_ns: int
    dispatch_path: DispatchPath
    table_version: Optional[str] = None
    notes: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# v1: per-call validation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ValidationReport:
    """Per-call validation flags (handoff §C.5).

    Flags are reported, not raised. Consumers decide whether to trust
    borderline outputs. The default-True convention means an operation
    that has no constraint on a given flag still reports True (the
    constraint is vacuously satisfied), distinct from a flag that fired.
    """

    asymptotic_closure_compliant: bool = True
    k_frust_invariant: bool = True
    round_trip_residual: Optional[float] = None
    notes: tuple[str, ...] = ()


T = TypeVar("T")


@dataclass(frozen=True)
class OperationOutput(Generic[T]):
    """Wrapped operation result (handoff §A.2 / §C.5 / §C.6).

    Returned by every `*_wrapped` operation. The unwrapped v0 functions
    keep their raw return types for back-compat; v1 consumers that want
    validation + provenance call the wrapped variants.
    """

    value: T
    validation: ValidationReport
    provenance: Provenance


# ---------------------------------------------------------------------------
# v1: inverse-lookup-table sidecar
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InverseLookupSidecar:
    """Curator-precomputed inverse-lookup table (handoff §C.4).

    Sidecar production lives in mpa-conform's curator path. This solver
    consumes the sidecar via `forward_sweep_invert`'s optional `sidecar`
    kwarg: when present, table-first; on miss, compute-fallback.

    `forward_lookup` and `inverse_lookup` are keyed on rounded
    (chit, gamma_AB, tau_obs) tuples — see `sidecar.py` for the rounding
    contract. `ambiguity_regions` records multi-valued inverse zones so
    consumers can opt to fall through to the compute path even on a hit.
    """

    version: str
    driver_profile_id: str
    driver_profile_version: str
    tau_obs_grid: tuple[float, ...]
    substrate_grid: tuple[SubstrateState, ...]
    canonical_grid: tuple[CanonicalState, ...]
    forward_lookup: dict[tuple[float, float, float], SubstrateState]
    inverse_lookup: dict[tuple[float, float, float], CanonicalState]
    ambiguity_regions: tuple[dict[str, Any], ...] = ()
