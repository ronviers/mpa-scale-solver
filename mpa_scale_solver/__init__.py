"""mpa-scale-solver — MPA scale-management kernel (Python v0.1.0).

Per the build handoff at
H:/mpa-conform/docs/mpa-scale-solver-python-build-handoff.md:
- This Python is the v0 SHIPPING artifact, not a reference.
- A native (Rust/C++) port comes later, byte-identical to this Python.
- Forward-only architecture (mpa-auditor §Q13).
- lookup_table translation-field shape (driver-profile.v2.0).
- Five-bucket regime classifier (handoff §C.4 / gfdr_model.js).

The seven operations of handoff §A.4 are the entire v0 API surface. No
other operations are added at v0. Observable extraction lives in mpa-solver,
bundle orchestration in mpa-conform, display in mpa-auditor.
"""

from .types import (
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
)
from .operations import (
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
)
from .gfdr_model import (
    vertex_regime,
    alpha_s,
    plateau_height,
    generate_locus,
    interp_locus,
    locus_residual,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # types
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
    # operations
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
    # gfdr model port
    "vertex_regime",
    "alpha_s",
    "plateau_height",
    "generate_locus",
    "interp_locus",
    "locus_residual",
]
