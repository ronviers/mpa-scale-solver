"""Cross-substrate operations (v3 — BLOCK_IN §v3).

Three cross-substrate compositions that the framework's primary
cross-substrate test (s→r migration per cdv1 §gFDR signatures) calls
out as first-class:

  - `gamut_overlap(gamut_a, gamut_b)` — intersection of two gamut
    specs in canonical (chit, gamma_AB) space, plus an optional
    tau_obs band intersection. Reports intersection extents and a
    Jaccard ratio.
  - `canonical_distance(state_a, state_b, metric)` — distance between
    two canonical states under one of several named metrics (l2, l1,
    regime, universality).
  - `universality_agreement(profile_a, profile_b, canonical_grid,
    tau_obs)` — over the canonical grid restricted to the gamut
    intersection, project each profile forward and re-invert through
    its own translation field, then compare 5-bucket regime labels
    (the universality class at the operational layer per cdv1 §gFDR).
    Reports agreement rate plus the per-class intra-class L².

These count against the seven-operation API as cross-substrate
compositions, not new fundamental operations (per CLAUDE.md). Each
also has a `*_wrapped` variant returning `OperationOutput[T]` with the
established validation + provenance pattern.

Substrate-class compatibility (e.g., comparing a substrate with
β_mem < 1 against one with β_mem = 1) is the caller's responsibility:
universality_agreement compares regimes at a shared tau_obs, not
matched-step-count, which is the correct comparison surface per
BLOCK_IN §v3 ("substrates with differing β_mem must be compared at
matched ν, not matched-step-count").
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence, Union

import numpy as np

from . import validation as _validation
from .gfdr_model import vertex_regime
from .operations import (
    apply_translation,
    forward_sweep_invert,
)
from .provenance import make_provenance
from .types import (
    AnyTranslationField,
    CanonicalState,
    DispatchPath,
    GamutSpec,
    OperationOutput,
)


# ---------------------------------------------------------------------------
# Op: gamut_overlap (cross-substrate gamut intersection in canonical space)
# ---------------------------------------------------------------------------


def _interval_intersection(
    a: tuple[float, float],
    b: tuple[float, float],
) -> Optional[tuple[float, float]]:
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    if lo > hi:
        return None
    return (lo, hi)


def _interval_extent(rng: Optional[tuple[float, float]]) -> float:
    if rng is None:
        return 0.0
    return rng[1] - rng[0]


def gamut_overlap(
    gamut_a: GamutSpec,
    gamut_b: GamutSpec,
) -> dict[str, Any]:
    """Intersection of two gamuts in canonical (+ optional tau_obs) space.

    Returns:
      {
        "chit_intersection": (lo, hi) | None,
        "gamma_AB_intersection": (lo, hi) | None,
        "tau_obs_intersection": (lo, hi) | None,
        "intersection_area": float,   # chit_extent * gamma_extent
        "union_area": float,
        "jaccard": float,             # intersection / union (canonical-plane)
        "compatible": bool,           # any 2D intersection at all
      }

    The tau_obs band is reported when both inputs carry one; the union
    area is the sum of areas minus the intersection.
    """
    chit_int = _interval_intersection(gamut_a.chit_range, gamut_b.chit_range)
    gamma_int = _interval_intersection(gamut_a.gamma_AB_range, gamut_b.gamma_AB_range)
    tau_int: Optional[tuple[float, float]] = None
    if gamut_a.tau_obs_range is not None and gamut_b.tau_obs_range is not None:
        tau_int = _interval_intersection(gamut_a.tau_obs_range, gamut_b.tau_obs_range)

    inter_area = _interval_extent(chit_int) * _interval_extent(gamma_int)
    area_a = (
        (gamut_a.chit_range[1] - gamut_a.chit_range[0])
        * (gamut_a.gamma_AB_range[1] - gamut_a.gamma_AB_range[0])
    )
    area_b = (
        (gamut_b.chit_range[1] - gamut_b.chit_range[0])
        * (gamut_b.gamma_AB_range[1] - gamut_b.gamma_AB_range[0])
    )
    union_area = area_a + area_b - inter_area
    jaccard = (inter_area / union_area) if union_area > 0.0 else 0.0
    compatible = chit_int is not None and gamma_int is not None
    return {
        "chit_intersection": chit_int,
        "gamma_AB_intersection": gamma_int,
        "tau_obs_intersection": tau_int,
        "intersection_area": float(inter_area),
        "union_area": float(union_area),
        "jaccard": float(jaccard),
        "compatible": bool(compatible),
    }


def gamut_overlap_wrapped(
    gamut_a: GamutSpec,
    gamut_b: GamutSpec,
) -> OperationOutput[dict[str, Any]]:
    result = gamut_overlap(gamut_a, gamut_b)
    notes: list[str] = []
    if not result["compatible"]:
        notes.append("gamut_a and gamut_b have no canonical-plane intersection")
    # Asymptotic-closure: gamut endpoints at exact 0/1 carry the same flag
    # as canonical states do (the gamut bound IS a canonical point).
    for axis_name, rng in (
        ("chit_a_low", gamut_a.chit_range[0]),
        ("chit_a_high", gamut_a.chit_range[1]),
        ("chit_b_low", gamut_b.chit_range[0]),
        ("chit_b_high", gamut_b.chit_range[1]),
    ):
        if rng == 0.0 or rng == 1.0:
            notes.append(f"asymptotic-closure flag: {axis_name}={rng}")
    from .types import ValidationReport

    report = ValidationReport(
        asymptotic_closure_compliant=not any("asymptotic-closure" in n for n in notes),
        k_frust_invariant=True,
        round_trip_residual=None,
        notes=tuple(notes),
    )
    prov = make_provenance("gamut_overlap")
    return OperationOutput(value=result, validation=report, provenance=prov)


# ---------------------------------------------------------------------------
# Op: canonical_distance (between two canonical states under named metric)
# ---------------------------------------------------------------------------


_CANONICAL_DISTANCE_METRICS = ("l2", "l1", "regime", "universality")


def canonical_distance(
    state_a: CanonicalState,
    state_b: CanonicalState,
    metric: str = "l2",
) -> float:
    """Distance between two canonical states under the named metric.

    Metrics:
      - `l2` (default): L² distance in (chit, gamma_AB) space.
      - `l1`: L¹ (Manhattan) distance in (chit, gamma_AB).
      - `regime`: 0 if the 5-bucket regimes match, 1 otherwise.
      - `universality`: 0 if regime AND sign(gamma_AB) AND k_frust all
        match (the I5+I1+I3 union-of-invariants at the state level),
        1 otherwise.
    """
    if metric == "l2":
        d_chit = state_b.chit - state_a.chit
        d_gamma = state_b.gamma_AB - state_a.gamma_AB
        return math.sqrt(d_chit * d_chit + d_gamma * d_gamma)
    if metric == "l1":
        return abs(state_b.chit - state_a.chit) + abs(state_b.gamma_AB - state_a.gamma_AB)
    if metric == "regime":
        return 0.0 if vertex_regime(state_a.chit) == vertex_regime(state_b.chit) else 1.0
    if metric == "universality":
        regime_match = vertex_regime(state_a.chit) == vertex_regime(state_b.chit)
        sign_match = _sign(state_a.gamma_AB) == _sign(state_b.gamma_AB)
        k_match = state_a.k_frust == state_b.k_frust
        return 0.0 if (regime_match and sign_match and k_match) else 1.0
    raise ValueError(
        f"unknown metric: {metric!r}; expected one of {_CANONICAL_DISTANCE_METRICS}"
    )


def canonical_distance_wrapped(
    state_a: CanonicalState,
    state_b: CanonicalState,
    metric: str = "l2",
) -> OperationOutput[float]:
    distance = canonical_distance(state_a, state_b, metric)
    from .types import ValidationReport

    notes: list[str] = []
    # Asymptotic-closure on either state propagates as a flag.
    ac_a, n_a = _validation.check_asymptotic_closure_canonical(state_a)
    ac_b, n_b = _validation.check_asymptotic_closure_canonical(state_b)
    notes.extend(n_a)
    notes.extend(n_b)
    report = ValidationReport(
        asymptotic_closure_compliant=ac_a and ac_b,
        k_frust_invariant=state_a.k_frust == state_b.k_frust,
        round_trip_residual=None,
        notes=tuple(notes),
    )
    prov = make_provenance(
        "canonical_distance",
        notes=(f"metric={metric}",),
    )
    return OperationOutput(value=distance, validation=report, provenance=prov)


def _sign(x: float) -> int:
    if x > 0.0:
        return 1
    if x < 0.0:
        return -1
    return 0


# ---------------------------------------------------------------------------
# Op: universality_agreement (cross-substrate s→r migration)
# ---------------------------------------------------------------------------


def universality_agreement(
    profile_a_field: AnyTranslationField,
    profile_a_gamut: GamutSpec,
    profile_b_field: AnyTranslationField,
    profile_b_gamut: GamutSpec,
    canonical_grid: np.ndarray,
    tau_obs: float,
    *,
    canonical_search_grid: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """Cross-substrate universality test (RFC-S §3 I5 metric, §5 §"agreement").

    For each point in `canonical_grid` that lies inside both gamuts:
      - Forward-project through profile A; invert through A's own field;
        record the recovered regime as the A-universality-class label.
      - Forward-project through profile B; invert through B's own field;
        record the recovered regime as the B-universality-class label.
      - Increment the agreement counter iff the two recovered regimes
        match.

    `canonical_search_grid` is the inversion grid for lookup-table fields.
    Defaults to `canonical_grid` itself (the same shape), which is the
    sensible default for matched curator-supplied grids.

    Returns:
      {
        "n_total": int,             # cells in canonical_grid
        "n_compared": int,          # cells in gamut_a ∩ gamut_b
        "n_agreed": int,            # cells where regimes match
        "agreement_rate": float,    # n_agreed / n_compared (0 if none)
        "intra_class_l2_mean": float,   # over agreed cells
        "intra_class_l2_max": float,    # over agreed cells
        "per_class_counts": {regime: {agree, disagree}},
      }

    The intra-class L² is computed in canonical space between the
    A-recovered and B-recovered points for agreed cells (RFC-S §5 I5's
    "intra-class parameter distance" metric).
    """
    if canonical_grid.ndim != 2 or canonical_grid.shape[1] != 2:
        raise ValueError(
            f"canonical_grid must have shape (N, 2); got {canonical_grid.shape}"
        )
    if canonical_search_grid is None:
        canonical_search_grid = canonical_grid

    n_total = canonical_grid.shape[0]
    n_compared = 0
    n_agreed = 0
    intra_class_l2s: list[float] = []
    per_class: dict[str, dict[str, int]] = {}

    for i in range(n_total):
        c = CanonicalState(
            chit=float(canonical_grid[i, 0]),
            gamma_AB=float(canonical_grid[i, 1]),
        )
        in_a = _in_gamut(c, profile_a_gamut)
        in_b = _in_gamut(c, profile_b_gamut)
        if not (in_a and in_b):
            continue
        n_compared += 1

        sub_a = apply_translation(c, profile_a_field, tau_obs)
        rec_a, _ = forward_sweep_invert(
            sub_a, profile_a_field, tau_obs, canonical_search_grid,
        )
        sub_b = apply_translation(c, profile_b_field, tau_obs)
        rec_b, _ = forward_sweep_invert(
            sub_b, profile_b_field, tau_obs, canonical_search_grid,
        )

        regime_a = vertex_regime(rec_a.chit)
        regime_b = vertex_regime(rec_b.chit)
        bucket = per_class.setdefault(regime_a, {"agree": 0, "disagree": 0})
        if regime_a == regime_b:
            n_agreed += 1
            bucket["agree"] += 1
            d_chit = rec_b.chit - rec_a.chit
            d_gamma = rec_b.gamma_AB - rec_a.gamma_AB
            intra_class_l2s.append(math.sqrt(d_chit * d_chit + d_gamma * d_gamma))
        else:
            bucket["disagree"] += 1

    agreement_rate = (n_agreed / n_compared) if n_compared > 0 else 0.0
    intra_mean = (sum(intra_class_l2s) / len(intra_class_l2s)) if intra_class_l2s else 0.0
    intra_max = max(intra_class_l2s) if intra_class_l2s else 0.0

    return {
        "n_total": int(n_total),
        "n_compared": int(n_compared),
        "n_agreed": int(n_agreed),
        "agreement_rate": float(agreement_rate),
        "intra_class_l2_mean": float(intra_mean),
        "intra_class_l2_max": float(intra_max),
        "per_class_counts": per_class,
    }


def universality_agreement_wrapped(
    profile_a_field: AnyTranslationField,
    profile_a_gamut: GamutSpec,
    profile_b_field: AnyTranslationField,
    profile_b_gamut: GamutSpec,
    canonical_grid: np.ndarray,
    tau_obs: float,
    *,
    canonical_search_grid: Optional[np.ndarray] = None,
) -> OperationOutput[dict[str, Any]]:
    result = universality_agreement(
        profile_a_field, profile_a_gamut,
        profile_b_field, profile_b_gamut,
        canonical_grid, tau_obs,
        canonical_search_grid=canonical_search_grid,
    )
    from .types import ValidationReport

    notes: list[str] = []
    if result["n_compared"] == 0:
        notes.append(
            "no cells in gamut intersection — agreement undefined"
        )
    report = ValidationReport(
        asymptotic_closure_compliant=True,
        k_frust_invariant=result["agreement_rate"] == 1.0 if result["n_compared"] > 0 else True,
        round_trip_residual=result["intra_class_l2_mean"],
        notes=tuple(notes),
    )
    prov = make_provenance(
        "universality_agreement",
        notes=(
            f"n_compared={result['n_compared']}/{result['n_total']}, "
            f"agreement_rate={result['agreement_rate']:.4f}",
        ),
    )
    return OperationOutput(value=result, validation=report, provenance=prov)


def _in_gamut(state: CanonicalState, gamut: GamutSpec) -> bool:
    return (
        gamut.chit_range[0] <= state.chit <= gamut.chit_range[1]
        and gamut.gamma_AB_range[0] <= state.gamma_AB <= gamut.gamma_AB_range[1]
    )
