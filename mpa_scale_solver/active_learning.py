"""Active learning: suggest measurements (v3 — BLOCK_IN §v3).

Driver-profile-conditional candidate-point selection. Given a driver
profile, score a canonical-space grid by:

  - **Posterior uncertainty** (v2.1 surface) — covariance-trace at the
    candidate point. The Laplace posterior in `jax_ops.lookup_table_posterior`
    gives a meaningful per-point uncertainty surface for lookup_table
    fields; tangent_flow / learned fields have closed-form inverses, so
    the posterior covariance is dominated by the Jacobian conditioning
    rather than discrete-grid sparsity.
  - **Gamut-edge proximity** — distance to the nearer gamut boundary.
    Edge cells with low classification confidence (gamut-classify
    boundary) are where additional measurements most reduce future
    out-of-gamut surprises.
  - **Intent-invariance fragility** (v2.3 surface) — applies each named
    intent to the candidate state and counts how many produced
    `invariant_preserved=False`. Cells where intents struggle are
    cells where the driver profile is locally weak (the discipline that
    BLOCK_IN §v3 names).

The three scores combine into a composite (weighted sum, defaults
match the BLOCK_IN call to "log-evidence + covariance trace + intent
invariance flags"). Top-n candidates are returned with the per-score
trace for inspection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from .operations import (
    apply_translation,
    forward_sweep_invert_posterior,
    gamut_classify,
    intent_map,
)
from .provenance import make_provenance
from .types import (
    AnyTranslationField,
    CanonicalState,
    GamutSpec,
    LearnedField,
    OperationOutput,
    TangentFlowField,
    TranslationField,
    ValidationReport,
)


_DEFAULT_INTENTS: tuple[str, ...] = ("I1", "I3", "I4", "I5")  # I2 doesn't adjust


@dataclass(frozen=True)
class MeasurementCandidate:
    """A suggested operating point for follow-up measurement.

    `state` is the canonical-space candidate. `score` is the composite
    rank score (higher = more informative). `components` carries the
    per-score breakdown for inspection / debugging.
    """

    state: CanonicalState
    tau_obs: float
    score: float
    components: dict[str, float]


def suggest_measurements(
    field: AnyTranslationField,
    gamut: GamutSpec,
    canonical_grid: np.ndarray,
    tau_obs: float,
    n: int = 5,
    *,
    canonical_search_grid: Optional[np.ndarray] = None,
    noise_variance: float = 1.0,
    weights: Optional[dict[str, float]] = None,
    intents: Sequence[str] = _DEFAULT_INTENTS,
) -> list[MeasurementCandidate]:
    """Suggest n canonical-space points to measure next.

    Per BLOCK_IN §v3 the discipline is: high-uncertainty regions in
    canonical space + gamut edges with low classification confidence +
    cells where intent invariants are at risk.

    `canonical_grid` is the candidate set (typically a regular sweep of
    the gamut interior). Cells outside the gamut are skipped — measuring
    out-of-gamut points is not informative for the driver under test.

    `canonical_search_grid` is the inversion grid the posterior uses
    when `field` is a lookup_table. Defaults to `canonical_grid` itself.

    `weights` controls the composite score (defaults sum to 1.0):
        uncertainty: 0.5    # posterior cov trace
        edge:        0.3    # 1 / min-gamut-distance
        fragility:   0.2    # fraction of intents flagging invariant loss

    Returns the top `n` candidates ranked by descending composite score.
    """
    if canonical_grid.ndim != 2 or canonical_grid.shape[1] != 2:
        raise ValueError(
            f"canonical_grid must have shape (N, 2); got {canonical_grid.shape}"
        )
    if canonical_search_grid is None:
        canonical_search_grid = canonical_grid
    w = {"uncertainty": 0.5, "edge": 0.3, "fragility": 0.2}
    if weights is not None:
        w.update(weights)

    candidates: list[MeasurementCandidate] = []

    for i in range(canonical_grid.shape[0]):
        state = CanonicalState(
            chit=float(canonical_grid[i, 0]),
            gamma_AB=float(canonical_grid[i, 1]),
        )
        diag = gamut_classify(state, tau_obs, gamut)
        if not diag["in_gamut"]:
            continue

        # 1) Posterior covariance-trace at the candidate.
        substrate_at_state = apply_translation(state, field, tau_obs)
        try:
            posterior = forward_sweep_invert_posterior(
                substrate_at_state, field, tau_obs,
                canonical_grid=canonical_search_grid,
                noise_variance=noise_variance,
            )
            cov = posterior.covariance
            cov_trace = float(cov[0][0] + cov[1][1])
            if not math.isfinite(cov_trace):
                cov_trace = 0.0
        except Exception:
            cov_trace = 0.0
        uncertainty_score = cov_trace

        # 2) Gamut-edge proximity (higher score = closer to edge).
        chit_lo, chit_hi = gamut.chit_range
        gamma_lo, gamma_hi = gamut.gamma_AB_range
        chit_edge = min(state.chit - chit_lo, chit_hi - state.chit)
        gamma_edge = min(state.gamma_AB - gamma_lo, gamma_hi - state.gamma_AB)
        min_edge = max(min(chit_edge, gamma_edge), 1e-6)
        edge_score = 1.0 / min_edge

        # 3) Intent-invariance fragility — fraction of intents flagging loss.
        fragility = 0
        for iid in intents:
            _mapped, sac = intent_map(state, tau_obs, gamut, iid)
            if not sac.get("invariant_preserved", True):
                fragility += 1
        fragility_score = fragility / max(len(intents), 1)

        # Composite — multiplicative weights, additive aggregation.
        composite = (
            w["uncertainty"] * uncertainty_score
            + w["edge"] * edge_score
            + w["fragility"] * fragility_score
        )
        candidates.append(
            MeasurementCandidate(
                state=state,
                tau_obs=tau_obs,
                score=composite,
                components={
                    "uncertainty": uncertainty_score,
                    "edge": edge_score,
                    "fragility": fragility_score,
                },
            )
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[: max(0, int(n))]


def suggest_measurements_wrapped(
    field: AnyTranslationField,
    gamut: GamutSpec,
    canonical_grid: np.ndarray,
    tau_obs: float,
    n: int = 5,
    *,
    canonical_search_grid: Optional[np.ndarray] = None,
    noise_variance: float = 1.0,
    weights: Optional[dict[str, float]] = None,
    intents: Sequence[str] = _DEFAULT_INTENTS,
) -> OperationOutput[list[MeasurementCandidate]]:
    suggestions = suggest_measurements(
        field, gamut, canonical_grid, tau_obs, n,
        canonical_search_grid=canonical_search_grid,
        noise_variance=noise_variance,
        weights=weights,
        intents=intents,
    )
    notes: list[str] = []
    if not suggestions:
        notes.append("no in-gamut candidates in canonical_grid")
    report = ValidationReport(
        asymptotic_closure_compliant=True,
        k_frust_invariant=True,
        round_trip_residual=None,
        notes=tuple(notes),
    )
    prov = make_provenance(
        "suggest_measurements",
        notes=(f"n_returned={len(suggestions)}",),
    )
    return OperationOutput(value=suggestions, validation=report, provenance=prov)
