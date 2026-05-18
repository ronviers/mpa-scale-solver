"""mpa-scale-solver — MPA scale-management kernel.

- This Python is the v0/v1/v2 SHIPPING artifact; native port is v6.
- Forward-only architecture (mpa-auditor §Q13).
- Two translation-field shapes: `lookup_table` (v0) and `tangent_flow`
  (v1, RFC-S Appendix B item 1, Banach canonical leading-order).
- Five-bucket regime classifier (gfdr_model.js parity).
- Continuous-form flow `C^nu = exp(nu * ln C)` in Markovian scope.
- Banach substrate calibration reference + analytical `state_at(nu)`.
- Inverse-lookup-table sidecar dispatch (curator-produced in mpa-conform).
- Per-call self-validation + provenance trail on the wrapped variants.
- v2 (cut (a)): JAX foundation under `jax_core` / `jax_ops` — pure
  differentiable forward maps for the tangent-flow surface, Banach
  analytical state, gradient-based inversion via BFGS, and
  `CanonicalState` registered as a JAX PyTree. The v0/v1 unwrapped
  signatures keep their `math.*` / numpy implementations unchanged
  (fixture byte-identity contract); the JAX surface is parallel and
  opt-in.

The seven operations are unchanged in surface; v1 added their
`*_wrapped` variants. Observable extraction lives in mpa-solver, bundle
orchestration in mpa-conform, display in mpa-auditor.
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
    # v2.1 addition
    Posterior,
    # v3 addition
    LearnedField,
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
    # v2.1 Bayesian inversion
    forward_sweep_invert_posterior,
    forward_sweep_invert_posterior_wrapped,
    # v2.3 intent composition
    intent_compose,
    intent_compose_wrapped,
)
from .cross_substrate import (
    # v3 cross-substrate compositions (BLOCK_IN §v3)
    canonical_distance,
    canonical_distance_wrapped,
    gamut_overlap,
    gamut_overlap_wrapped,
    universality_agreement,
    universality_agreement_wrapped,
)
from .active_learning import (
    # v3 active learning (BLOCK_IN §v3)
    MeasurementCandidate,
    suggest_measurements,
    suggest_measurements_wrapped,
)
from .streaming import (
    # v4 streaming surface (BLOCK_IN §v4)
    InversionResult,
    forward_sweep_invert_stream,
    from_iterable,
    from_stdin,
)
from .self_test import (
    # v5 continuous Banach self-test cadence (BLOCK_IN §v5)
    BanachDriftReport,
    DRIFT_TOLERANCE,
    SelfTestCadence,
    run_banach_self_test,
)
from .sensitivity import (
    # v5 sensitivity backprop (BLOCK_IN §v5)
    driver_profile_loss_grad,
    field_parameter_sensitivity,
    inversion_sensitivity,
    trajectory_substrate_diff,
    trajectory_substrate_jacobian,
)
from .symbolic_query import (
    # v4 symbolic query DSL (BLOCK_IN §v4)
    QueryParseError,
    QueryResult,
    query,
    supported_patterns,
)
from . import plotting  # noqa: F401 — v4 default plot hooks (lazy backends)
from .flow import flow
from .banach import BanachSubstrate, build_sidecar_for_banach
from .sidecar import (
    DEFAULT_ROUNDING_DECIMALS,
    WIRE_VERSION,
    decode_sidecar_from_json,
    decode_sidecar_key,
    encode_sidecar_key,
    encode_sidecar_to_json,
    lookup_forward,
    lookup_inverse,
    round_key,
)
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

# v2 JAX surface (BLOCK_IN §v2 cut (a) — JAX foundation + differentiability).
# Imported as modules so consumers can pick the surface they need; the
# `jax_pytree` import has the side effect of registering CanonicalState
# as a JAX PyTree (idempotent).
from . import jax_core, jax_ops, jax_pytree  # noqa: F401


# ---------------------------------------------------------------------------
# v6 — native (Rust) acceleration via `_mpa_scale_solver_native`.
#
# When the pyo3 wheel produced by `cd rust && maturin build --features
# python` is installed, the 9 wrapped variants + raw
# `validate_driver_profile` + `flow` re-bind here to thin Python shims
# (see `_native_shim.py`) that route through the native module and
# reconstruct the typed Python dataclasses on the way back. The pure-
# Python implementations above remain the executable reference; the
# shim's typed-dataclass return preserves the consumer surface
# (`out.value.chit`, `out.provenance.dispatch_path ==
# DispatchPath.DIRECT_COMPUTE`, etc.).
#
# Wire-level parity proven by `tests/test_rust_parity.py` across all
# 9 wrapped variants + raw + flow.
try:
    from ._native_shim import (
        apply_translation_wrapped,  # noqa: F811
        flow,  # noqa: F811
        forward_sweep_invert_posterior_wrapped,  # noqa: F811
        forward_sweep_invert_wrapped,  # noqa: F811
        gamut_classify_wrapped,  # noqa: F811
        intent_compose_wrapped,  # noqa: F811
        intent_map_wrapped,  # noqa: F811
        regime_at_wrapped,  # noqa: F811
        tau_obs_sweep_wrapped,  # noqa: F811
        validate_driver_profile,  # noqa: F811
        validate_driver_profile_wrapped,  # noqa: F811
    )
    NATIVE_AVAILABLE = True
except ImportError:
    # No native wheel installed for this platform / Python version —
    # the pure-Python implementations imported above remain bound.
    NATIVE_AVAILABLE = False


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
    # types — v2.1
    "Posterior",
    # types — v3
    "LearnedField",
    "MeasurementCandidate",
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
    # operations — v2.1 (Bayesian inversion)
    "forward_sweep_invert_posterior",
    "forward_sweep_invert_posterior_wrapped",
    # operations — v2.3 (intent composition)
    "intent_compose",
    "intent_compose_wrapped",
    # operations — v3 (cross-substrate)
    "canonical_distance",
    "canonical_distance_wrapped",
    "gamut_overlap",
    "gamut_overlap_wrapped",
    "universality_agreement",
    "universality_agreement_wrapped",
    # operations — v3 (active learning)
    "suggest_measurements",
    "suggest_measurements_wrapped",
    # v4 streaming surface
    "InversionResult",
    "forward_sweep_invert_stream",
    "from_iterable",
    "from_stdin",
    # v4 symbolic query
    "QueryParseError",
    "QueryResult",
    "query",
    "supported_patterns",
    # v5 continuous self-test cadence
    "BanachDriftReport",
    "DRIFT_TOLERANCE",
    "SelfTestCadence",
    "run_banach_self_test",
    # v5 sensitivity backprop
    "driver_profile_loss_grad",
    "field_parameter_sensitivity",
    "inversion_sensitivity",
    "trajectory_substrate_diff",
    "trajectory_substrate_jacobian",
    # v1 new functions / modules
    "flow",
    "BanachSubstrate",
    "build_sidecar_for_banach",
    "lookup_forward",
    "lookup_inverse",
    "round_key",
    # v1 sidecar JSON wire format (cross-language artifact contract)
    "encode_sidecar_to_json",
    "decode_sidecar_from_json",
    "encode_sidecar_key",
    "decode_sidecar_key",
    "WIRE_VERSION",
    "DEFAULT_ROUNDING_DECIMALS",
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
