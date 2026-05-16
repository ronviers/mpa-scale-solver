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


# ---------------------------------------------------------------------------
# Per-intent RFC-S §5 metrics (v3 — BLOCK_IN §v3)
# ---------------------------------------------------------------------------
#
# RFC-S §5 (v0.2) lists the metric each intent uses for both forward and
# round-trip comparison:
#
#   I1 regime-preserving   : Hamming on regime partition;
#                            agreement on edge-type partition (sign(γ))
#   I2 drive-faithful      : L² on drive vector (chit, γ_AB);
#                            max deviation on γ
#   I3 capacity-preserving : ‖Γ*‖ deviation (chit distance to the |chit|=0.7
#                            fixed-point boundary at the operational layer);
#                            structural-pattern similarity (k_frust match
#                            at the state level)
#   I4 persistence-preserv : sequence distance on {ε_n} (sign(γ_AB)
#                            contraction-ordering proxy at the state level);
#                            survival-declaration agreement (gamut residency)
#   I5 signature-preserving: universality-class agreement (5-bucket regime
#                            label is the universality class at the
#                            operational layer per cdv1 §gFDR);
#                            intra-class parameter L² for matching cells
#
# Each helper takes (original, recovered) CanonicalStates and returns a
# float (lower-is-better) plus an explanatory dict the summary aggregator
# can fold per-intent.


def _capacity_class(chit: float) -> str:
    return "deep" if abs(chit) >= 0.7 else "shallow"


def _sign(x: float) -> int:
    if x > 0.0:
        return 1
    if x < 0.0:
        return -1
    return 0


def per_intent_cell_metric(
    intent_id: str,
    original: CanonicalState,
    recovered: CanonicalState,
    *,
    in_gamut: Optional[bool] = None,
) -> dict[str, Any]:
    """Per-cell metric components for the named intent (RFC-S §5).

    Returns a dict whose keys are intent-specific. `validate_driver_profile`
    aggregates these into summary statistics.

    `in_gamut` (optional) supplies the cell's gamut residency for the I4
    survival-declaration component. None when the caller has not run
    `gamut_classify` for this cell.
    """
    # Import locally to avoid circular import with operations.regime_at.
    from .gfdr_model import vertex_regime

    if intent_id == "I1":
        # Hamming on regime partition + sign(gamma) agreement.
        regime_match = vertex_regime(original.chit) == vertex_regime(recovered.chit)
        sign_match = _sign(original.gamma_AB) == _sign(recovered.gamma_AB)
        k_frust_match = original.k_frust == recovered.k_frust
        return {
            "regime_match": regime_match,
            "edge_type_match": sign_match,
            "k_frust_match": k_frust_match,
            "hamming": 0 if (regime_match and sign_match) else 1,
        }

    if intent_id == "I2":
        # L² on drive vector (chit, gamma_AB); max deviation on gamma.
        d_chit = recovered.chit - original.chit
        d_gamma = recovered.gamma_AB - original.gamma_AB
        return {
            "l2_drive": math.sqrt(d_chit * d_chit + d_gamma * d_gamma),
            "gamma_deviation": abs(d_gamma),
        }

    if intent_id == "I3":
        # ‖Γ*‖ deviation — distance to the |chit|=0.7 fixed-point boundary
        # is the capacity-class signed distance at the operational layer;
        # ‖Γ*‖ deviation between original and recovered is the change in
        # that signed distance. Plus structural-pattern similarity (k_frust).
        gamma_star_original = abs(original.chit) - 0.7
        gamma_star_recovered = abs(recovered.chit) - 0.7
        return {
            "gamma_star_deviation": abs(gamma_star_recovered - gamma_star_original),
            "capacity_class_match": _capacity_class(original.chit) == _capacity_class(recovered.chit),
            "k_frust_match": original.k_frust == recovered.k_frust,
        }

    if intent_id == "I4":
        # Sequence distance on {ε_n} at state-level: sign(gamma_AB)
        # contraction-ordering proxy. Plus survival = in-gamut.
        sign_match = _sign(original.gamma_AB) == _sign(recovered.gamma_AB)
        return {
            "epsilon_sequence_distance": 0 if sign_match else 1,
            "survival": True if in_gamut is None else bool(in_gamut),
        }

    if intent_id == "I5":
        # Universality-class agreement (5-bucket regime label is the
        # universality class at the operational layer per cdv1 §gFDR);
        # intra-class parameter distance (L² in canonical space).
        original_class = vertex_regime(original.chit)
        recovered_class = vertex_regime(recovered.chit)
        class_match = original_class == recovered_class
        d_chit = recovered.chit - original.chit
        d_gamma = recovered.gamma_AB - original.gamma_AB
        return {
            "universality_class_match": class_match,
            "intra_class_l2": math.sqrt(d_chit * d_chit + d_gamma * d_gamma) if class_match else None,
            "original_class": original_class,
            "recovered_class": recovered_class,
        }

    raise ValueError(f"unknown intent: {intent_id!r}")


def aggregate_per_intent_metrics(
    intent_id: str,
    cells: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-cell intent metrics into summary statistics.

    Returns the intent-specific summary block that rides next to the
    summary's `forward_residuals` / `round_trip_residuals` keys.
    """
    n = len(cells)
    if n == 0:
        return {"intent": intent_id, "n_cells": 0}

    if intent_id == "I1":
        hamming = sum(c["hamming"] for c in cells) / n
        return {
            "intent": "I1",
            "n_cells": n,
            "hamming_rate": hamming,
            "regime_match_rate": sum(1 for c in cells if c["regime_match"]) / n,
            "edge_type_match_rate": sum(1 for c in cells if c["edge_type_match"]) / n,
            "k_frust_match_rate": sum(1 for c in cells if c["k_frust_match"]) / n,
        }

    if intent_id == "I2":
        l2s = [c["l2_drive"] for c in cells]
        gdevs = [c["gamma_deviation"] for c in cells]
        return {
            "intent": "I2",
            "n_cells": n,
            "l2_drive_mean": sum(l2s) / n,
            "l2_drive_max": max(l2s),
            "gamma_deviation_max": max(gdevs),
            "gamma_deviation_mean": sum(gdevs) / n,
        }

    if intent_id == "I3":
        gsds = [c["gamma_star_deviation"] for c in cells]
        return {
            "intent": "I3",
            "n_cells": n,
            "gamma_star_deviation_mean": sum(gsds) / n,
            "gamma_star_deviation_max": max(gsds),
            "capacity_class_match_rate": sum(1 for c in cells if c["capacity_class_match"]) / n,
            "k_frust_match_rate": sum(1 for c in cells if c["k_frust_match"]) / n,
        }

    if intent_id == "I4":
        seq_dists = [c["epsilon_sequence_distance"] for c in cells]
        return {
            "intent": "I4",
            "n_cells": n,
            "epsilon_sequence_distance_mean": sum(seq_dists) / n,
            "survival_rate": sum(1 for c in cells if c["survival"]) / n,
        }

    if intent_id == "I5":
        matches = sum(1 for c in cells if c["universality_class_match"]) / n
        intra = [c["intra_class_l2"] for c in cells if c["intra_class_l2"] is not None]
        return {
            "intent": "I5",
            "n_cells": n,
            "universality_class_agreement_rate": matches,
            "intra_class_l2_mean": (sum(intra) / len(intra)) if intra else 0.0,
            "intra_class_l2_max": (max(intra) if intra else 0.0),
        }

    raise ValueError(f"unknown intent: {intent_id!r}")


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
