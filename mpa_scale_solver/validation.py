"""Per-call self-validation (handoff §C.5).

Three flags are checked across the seven operations:

- **Asymptotic-Closure compliance** (v9 §Asymptotic closure): no framework-
  prediction observable attains exact 0 or 1 at non-asymptotic points.
  The Banach substrate is the documented exception — it sits at the
  asymptotic limits by construction. For real substrates an exact 0.0
  or 1.0 in a canonical / substrate float is a flag.
- **k_frust invariance** (v9 §Scale-relativity): the topological
  invariant is preserved across trajectory operations.
- **Round-trip residual**: optional; populated by inversion-side
  validators that recompose forward-then-back and report the gap.

Flags are reported, not raised. Consumers decide whether to trust
borderline outputs.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Optional, Sequence

from .types import (
    CanonicalState,
    SubstrateState,
    ValidationReport,
)


# Float literals that flag the Asymptotic-Closure Principle. Comparison is
# exact equality on purpose: floats that arrived at literal 0.0 or 1.0 are
# either inputs the framework forbids at non-asymptotic points, or a
# numerical degeneracy worth surfacing.
_ASYMPTOTIC_FLOATS: tuple[float, ...] = (0.0, 1.0)


def _is_asymptotic_literal(value: float) -> bool:
    return any(value == lit for lit in _ASYMPTOTIC_FLOATS)


def check_asymptotic_closure_canonical(
    canonical: CanonicalState,
) -> tuple[bool, list[str]]:
    """Asymptotic-Closure check on a CanonicalState's float channels."""
    notes: list[str] = []
    for name, val in (("chit", canonical.chit), ("gamma_AB", canonical.gamma_AB)):
        if _is_asymptotic_literal(float(val)):
            notes.append(
                f"canonical.{name} == {val} (asymptotic-closure flag; "
                f"Banach substrate is the documented exception)"
            )
    return (len(notes) == 0, notes)


def check_asymptotic_closure_substrate(
    substrate: SubstrateState,
    *,
    excluded_keys: Iterable[str] = (),
) -> tuple[bool, list[str]]:
    """Asymptotic-Closure check on substrate observables.

    `excluded_keys` carries the substrate's declared normalization
    conventions (e.g. a substrate whose unit interval is [0, 1] by
    construction). Keys in `excluded_keys` are skipped.
    """
    excluded = set(excluded_keys)
    notes: list[str] = []
    for k, v in substrate.observables.items():
        if k in excluded:
            continue
        if isinstance(v, bool):
            continue
        if not isinstance(v, (int, float)):
            continue
        if _is_asymptotic_literal(float(v)):
            notes.append(
                f"substrate.observables[{k!r}] == {v} (asymptotic-closure flag)"
            )
    return (len(notes) == 0, notes)


def check_k_frust_invariance(
    trajectory: Sequence[CanonicalState],
) -> tuple[bool, list[str]]:
    """k_frust must not flip across a trajectory (v9 §Scale-relativity)."""
    if not trajectory:
        return (True, [])
    initial = trajectory[0].k_frust
    flips: list[int] = []
    for i, state in enumerate(trajectory[1:], start=1):
        if state.k_frust != initial:
            flips.append(i)
    if flips:
        return (False, [
            f"k_frust flipped at frames {flips} (was {initial!r} initially)"
        ])
    return (True, [])


def report_for_apply_translation(
    canonical_in: CanonicalState,
    substrate_out: SubstrateState,
    *,
    excluded_substrate_keys: Iterable[str] = (),
) -> ValidationReport:
    notes: list[str] = []
    ac_c, n_c = check_asymptotic_closure_canonical(canonical_in)
    ac_s, n_s = check_asymptotic_closure_substrate(
        substrate_out, excluded_keys=excluded_substrate_keys,
    )
    notes.extend(n_c)
    notes.extend(n_s)
    return ValidationReport(
        asymptotic_closure_compliant=ac_c and ac_s,
        k_frust_invariant=True,  # vacuously: apply_translation is per-frame
        round_trip_residual=None,
        notes=tuple(notes),
    )


def report_for_forward_sweep_invert(
    target: SubstrateState,
    recovered: CanonicalState,
    *,
    round_trip_residual: Optional[float] = None,
    excluded_substrate_keys: Iterable[str] = (),
) -> ValidationReport:
    notes: list[str] = []
    ac_c, n_c = check_asymptotic_closure_canonical(recovered)
    ac_s, n_s = check_asymptotic_closure_substrate(
        target, excluded_keys=excluded_substrate_keys,
    )
    notes.extend(n_c)
    notes.extend(n_s)
    return ValidationReport(
        asymptotic_closure_compliant=ac_c and ac_s,
        k_frust_invariant=True,
        round_trip_residual=round_trip_residual,
        notes=tuple(notes),
    )


def report_for_tau_obs_sweep(
    trajectory: Sequence[CanonicalState],
) -> ValidationReport:
    notes: list[str] = []
    inv_ok, inv_notes = check_k_frust_invariance(trajectory)
    notes.extend(inv_notes)
    # Per-frame asymptotic-closure: any frame at exact 0/1 flags.
    ac_ok = True
    for i, state in enumerate(trajectory):
        ok, n = check_asymptotic_closure_canonical(state)
        if not ok:
            ac_ok = False
            for line in n:
                notes.append(f"frame {i}: {line}")
    return ValidationReport(
        asymptotic_closure_compliant=ac_ok,
        k_frust_invariant=inv_ok,
        round_trip_residual=None,
        notes=tuple(notes),
    )


def report_for_regime_at(
    canonical: CanonicalState,
) -> ValidationReport:
    ac_c, n_c = check_asymptotic_closure_canonical(canonical)
    return ValidationReport(
        asymptotic_closure_compliant=ac_c,
        k_frust_invariant=True,
        round_trip_residual=None,
        notes=tuple(n_c),
    )


def report_for_gamut_classify(
    canonical: CanonicalState,
) -> ValidationReport:
    ac_c, n_c = check_asymptotic_closure_canonical(canonical)
    return ValidationReport(
        asymptotic_closure_compliant=ac_c,
        k_frust_invariant=True,
        round_trip_residual=None,
        notes=tuple(n_c),
    )


def report_for_intent_map(
    original: CanonicalState,
    mapped: CanonicalState,
    sacrifice: dict[str, Any],
) -> ValidationReport:
    """Intent invariance: the intent's named invariant flags `k_frust_invariant`.

    Reads `invariant_preserved` from the sacrifice dict (v2.3 uniform key);
    falls back to `regime_preserved` for v1-shaped I5 sacrifice dicts.
    The `k_frust_invariant` ValidationReport field is repurposed here as
    the intent-invariant pass/fail (v1 convention extended by v2.3).
    """
    notes: list[str] = []
    ac_orig, n_orig = check_asymptotic_closure_canonical(original)
    ac_map, n_map = check_asymptotic_closure_canonical(mapped)
    notes.extend(n_orig)
    notes.extend(n_map)
    invariant_preserved = bool(sacrifice.get(
        "invariant_preserved", sacrifice.get("regime_preserved", True),
    ))
    if not invariant_preserved:
        intent = sacrifice.get("intent", "?")
        invariant_name = sacrifice.get(
            "preserved_invariant",
            "regime_label" if intent == "I5" else "?",
        )
        if intent == "I5":
            notes.append(
                f"I5 regime not preserved: "
                f"{sacrifice.get('original_regime')} -> {sacrifice.get('mapped_regime')}"
            )
        else:
            notes.append(f"{intent} did not preserve {invariant_name}")
    return ValidationReport(
        asymptotic_closure_compliant=ac_orig and ac_map,
        k_frust_invariant=invariant_preserved,
        round_trip_residual=None,
        notes=tuple(notes),
    )


def report_for_intent_compose(
    original: CanonicalState,
    mapped: CanonicalState,
    sacrifices: Sequence[dict[str, Any]],
) -> ValidationReport:
    """Composition: `k_frust_invariant` True iff every intent preserved its invariant."""
    notes: list[str] = []
    ac_orig, n_orig = check_asymptotic_closure_canonical(original)
    ac_map, n_map = check_asymptotic_closure_canonical(mapped)
    notes.extend(n_orig)
    notes.extend(n_map)
    all_preserved = True
    for sac in sacrifices:
        ok = bool(sac.get(
            "invariant_preserved", sac.get("regime_preserved", True),
        ))
        if not ok:
            all_preserved = False
            intent = sac.get("intent", "?")
            invariant_name = sac.get(
                "preserved_invariant",
                "regime_label" if intent == "I5" else "?",
            )
            notes.append(f"{intent} did not preserve {invariant_name}")
    return ValidationReport(
        asymptotic_closure_compliant=ac_orig and ac_map,
        k_frust_invariant=all_preserved,
        round_trip_residual=None,
        notes=tuple(notes),
    )


def report_for_validate_driver_profile(
    summary: dict[str, Any],
) -> ValidationReport:
    rt_mean = float(summary.get("round_trip_mean", 0.0))
    regime_rate = float(summary.get("regime_agreement_rate", 1.0))
    notes: list[str] = []
    if rt_mean > 0.0 and not math.isfinite(rt_mean):
        notes.append(f"round_trip_mean non-finite: {rt_mean}")
    if regime_rate < 1.0:
        notes.append(
            f"regime_agreement_rate {regime_rate:.4f} < 1.0 "
            f"({summary.get('intent')} round-trip)"
        )
    return ValidationReport(
        asymptotic_closure_compliant=True,  # not checked at the summary level
        k_frust_invariant=regime_rate == 1.0,
        round_trip_residual=rt_mean,
        notes=tuple(notes),
    )


def validation_flags_bitfield(report: ValidationReport) -> float:
    """Float32-encoded bitfield of the report's pass/fail flags (handoff §A.5).

    Bit 0: asymptotic_closure_compliant
    Bit 1: k_frust_invariant
    Bit 2: round_trip_residual present (1) or None (0)

    Encoded as a small integer cast to float so it survives the EXR
    channel's float32 storage without quantization.
    """
    bits = 0
    if report.asymptotic_closure_compliant:
        bits |= 1 << 0
    if report.k_frust_invariant:
        bits |= 1 << 1
    if report.round_trip_residual is not None:
        bits |= 1 << 2
    return float(bits)
