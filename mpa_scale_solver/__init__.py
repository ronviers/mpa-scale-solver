"""mpa-scale-solver — MPA scale-management kernel (Python v1.0.0).

Per the v1 build handoff at
H:/mpa-conform/docs/mpa-scale-solver-v1-handoff.md:

- This Python is the v0/v1 SHIPPING artifact; native port is v6.
- Forward-only architecture (mpa-auditor §Q13).
- Two translation-field shapes: `lookup_table` (v0) and `tangent_flow`
  (v1, RFC-S Appendix B item 1, Banach canonical leading-order).
- Five-bucket regime classifier (handoff §C.4 / gfdr_model.js).
- Continuous-form flow `C^nu = exp(nu * ln C)` in Markovian scope.
- Banach substrate calibration reference + analytical `state_at(nu)`.
- Inverse-lookup-table sidecar dispatch (curator-produced in mpa-conform).
- Per-call self-validation + provenance trail on the wrapped variants.

The seven operations of handoff §A.4 are unchanged in surface; v1 adds
their `*_wrapped` variants for consumers that want validation +
provenance riding on `OperationOutput[T]`. Observable extraction lives
in mpa-solver, bundle orchestration in mpa-conform, display in mpa-auditor.
"""

from ._version import __version__
from .types import (
    # v0 unchanged
    CanonicalPoint,
    CanonicalState,
    DisplayBand,
    GamutSpec,
    OperatingPoint,
    RegimeLabel,
    RegimeReading,
    SubstrateState,
    TranslationField,
    TranslationRule,
    # v1 additions
    AnyTranslationField,
    DispatchPath,
    InverseLookupSidecar,
    LookupTableField,
    OperationOutput,
    Provenance,
    ScalingRule,
    TangentFlowField,
    ValidationReport,
)
from .operations import (
    # v0 sigs unchanged
    apply_translation,
    forward_sweep_invert,
    tau_obs_sweep,
    regime_at,
    regime_display_band,
    gamut_classify,
    intent_map,
    validate_driver_profile,
    parse_translation_field,
    parse_gamut,
    # v1 wrapped variants
    apply_translation_wrapped,
    forward_sweep_invert_wrapped,
    tau_obs_sweep_wrapped,
    regime_at_wrapped,
    gamut_classify_wrapped,
    intent_map_wrapped,
    validate_driver_profile_wrapped,
)
from .flow import flow
from .banach import BanachSubstrate, build_sidecar_for_banach
from .sidecar import lookup_forward, lookup_inverse, round_key
from .provenance import make_provenance, provenance_hash
from .validation import validation_flags_bitfield
from .gfdr_model import (
    vertex_regime,
    alpha_s,
    plateau_height,
    generate_locus,
    interp_locus,
    locus_residual,
)


__all__ = [
    "__version__",
    # types — v0
    "CanonicalPoint",
    "CanonicalState",
    "DisplayBand",
    "GamutSpec",
    "OperatingPoint",
    "RegimeLabel",
    "RegimeReading",
    "SubstrateState",
    "TranslationField",
    "TranslationRule",
    # types — v1
    "AnyTranslationField",
    "DispatchPath",
    "InverseLookupSidecar",
    "LookupTableField",
    "OperationOutput",
    "Provenance",
    "ScalingRule",
    "TangentFlowField",
    "ValidationReport",
    # operations — v0
    "apply_translation",
    "forward_sweep_invert",
    "tau_obs_sweep",
    "regime_at",
    "regime_display_band",
    "gamut_classify",
    "intent_map",
    "validate_driver_profile",
    "parse_translation_field",
    "parse_gamut",
    # operations — v1
    "apply_translation_wrapped",
    "forward_sweep_invert_wrapped",
    "tau_obs_sweep_wrapped",
    "regime_at_wrapped",
    "gamut_classify_wrapped",
    "intent_map_wrapped",
    "validate_driver_profile_wrapped",
    # v1 new functions / modules
    "flow",
    "BanachSubstrate",
    "build_sidecar_for_banach",
    "lookup_forward",
    "lookup_inverse",
    "round_key",
    "make_provenance",
    "provenance_hash",
    "validation_flags_bitfield",
    # gfdr model port
    "vertex_regime",
    "alpha_s",
    "plateau_height",
    "generate_locus",
    "interp_locus",
    "locus_residual",
]
