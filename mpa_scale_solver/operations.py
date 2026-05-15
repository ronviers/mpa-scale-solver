"""The seven scale-solver operations (handoff §A.4).

All stateless free functions on plain dataclasses. Per §A.3, the operations
take tau_obs as an explicit argument rather than reading it off the state.

Per §C.2 the production translation-field shape is `lookup_table`. The
parametric path from the prior reference (aging_log, trivial_baseline) lives
in `_test_fixtures.py` and is used only by the camera test.

Per §C.4 the canonical regime classifier is the FIVE-bucket cut from
gfdr_model.vertex_regime. The three-bucket cut (`regime_display_band`) is a
display-only helper.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional, Union

import numpy as np

from .gfdr_model import vertex_regime
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


# Default L2-distance threshold beyond which apply_translation declares
# a canonical state outside the table's domain (handoff §C.2 step 2).
DEFAULT_DOMAIN_DISTANCE_THRESHOLD = 1e9  # effectively off; tables specify their own


# ---------------------------------------------------------------------------
# Internal: indexed view of a TranslationField for fast lookup
# ---------------------------------------------------------------------------

class TranslationFieldIndex:
    """Pre-cached numpy view of a TranslationField's rule canonicals.

    Idempotent and immutable after construction. Callers that need to apply
    the same field many times (the camera test, forward_sweep_invert at
    repeated tau_obs frames) build one index and reuse it.

    Not exported as part of the public API at v0 — internal optimization.
    """

    __slots__ = ("_field", "_chit", "_gamma_AB", "_tau_obs", "_has_tau", "_rules")

    def __init__(self, field: TranslationField) -> None:
        rules = field.rule
        n = len(rules)
        self._field = field
        self._rules = rules
        self._chit = np.empty(n, dtype=np.float64)
        self._gamma_AB = np.empty(n, dtype=np.float64)
        self._tau_obs = np.empty(n, dtype=np.float64)
        self._has_tau = np.zeros(n, dtype=bool)
        for i, r in enumerate(rules):
            self._chit[i] = r.canonical.chit
            self._gamma_AB[i] = r.canonical.gamma_AB
            tau_in_axes = r.operating_point.axes.get("tau_obs")
            if tau_in_axes is None:
                self._tau_obs[i] = np.nan
                self._has_tau[i] = False
            else:
                self._tau_obs[i] = float(tau_in_axes)
                self._has_tau[i] = True

    def nearest(
        self,
        chit: float,
        gamma_AB: float,
        tau_obs: float,
        *,
        tau_obs_weight: float = 1.0,
    ) -> tuple[int, float]:
        """Return (rule_index, squared_distance) for the nearest rule.

        Distance is L2 over (chit, gamma_AB) with an additional log-tau_obs
        term for rules carrying a tau_obs axis.
        """
        d_chit = self._chit - chit
        d_gamma = self._gamma_AB - gamma_AB
        d2 = d_chit * d_chit + d_gamma * d_gamma
        # log-tau_obs term for rules with a tau_obs axis
        if np.any(self._has_tau):
            log_tau_q = math.log(tau_obs) if tau_obs > 0 else 0.0
            d_tau = np.where(
                self._has_tau,
                np.log(np.where(self._has_tau, self._tau_obs, 1.0)) - log_tau_q,
                0.0,
            )
            d2 = d2 + tau_obs_weight * d_tau * d_tau
        idx = int(np.argmin(d2))
        return idx, float(d2[idx])


# ---------------------------------------------------------------------------
# Op 1: apply_translation (lookup-form forward map; §C.2)
# ---------------------------------------------------------------------------

def apply_translation(
    canonical: CanonicalState,
    field: Union[TranslationField, TranslationFieldIndex],
    tau_obs: float,
    *,
    domain_distance_threshold: float = DEFAULT_DOMAIN_DISTANCE_THRESHOLD,
    tau_obs_weight: float = 1.0,
) -> SubstrateState:
    """Forward map: canonical state -> substrate-native at tau_obs.

    Per §Q13 the only well-defined direction is forward; the backward map
    (substrate -> canonical) is handled by forward_sweep_invert.

    Per §C.2 the production shape is `lookup_table`. Given (canonical,
    tau_obs), find the rule whose (canonical.chit, canonical.gamma_AB) and
    operating-point tau_obs (if any) are L2-nearest. Return a SubstrateState
    carrying the matched rule's operating-point identity (label + axes).

    Raises ValueError if the nearest rule is beyond
    domain_distance_threshold — that is the curator-path's signal that the
    declared driver profile does not cover this substrate state (a gamut
    violation handled upstream).
    """
    index = field if isinstance(field, TranslationFieldIndex) else TranslationFieldIndex(field)
    if len(index._rules) == 0:
        raise ValueError("translation field has no rules")
    idx, d2 = index.nearest(
        canonical.chit, canonical.gamma_AB, tau_obs, tau_obs_weight=tau_obs_weight
    )
    if d2 > domain_distance_threshold * domain_distance_threshold:
        raise ValueError(
            f"canonical state outside translation field domain: "
            f"nearest rule distance {math.sqrt(d2):.4g} > threshold "
            f"{domain_distance_threshold:.4g}"
        )
    rule = index._rules[idx]
    return SubstrateState(
        tau_obs=tau_obs,
        label=rule.operating_point.label,
        axes=dict(rule.operating_point.axes),
        observables={
            "canonical_chit": rule.canonical.chit,
            "canonical_gamma_AB": rule.canonical.gamma_AB,
        },
    )


# ---------------------------------------------------------------------------
# Op 2: forward_sweep_invert (substrate -> canonical via forward search)
# ---------------------------------------------------------------------------

def _default_substrate_score(predicted: SubstrateState, target: SubstrateState) -> float:
    """Default L2 score over shared numeric keys (cross-category).

    Predicted and target each carry observables and axes; the score sees
    them as one flat bag of named scalars. Where the same key appears in
    both, the squared difference contributes. Where a key appears in only
    one side, it contributes nothing (treat as no constraint).

    Per the seed-corpus convention, lookup_table rules park their
    substrate-side measurements in operating_point.axes (returned via
    SubstrateState.axes), while curators or window-averagers may place
    the same scalar in SubstrateState.observables. Cross-category lets
    both shapes pair up without consumers having to know which side put
    a given measurement where.
    """
    p_all = {**predicted.observables, **predicted.axes}
    t_all = {**target.observables, **target.axes}
    score = 0.0
    for k in set(p_all.keys()) & set(t_all.keys()):
        pv, tv = p_all[k], t_all[k]
        if isinstance(pv, (int, float)) and not isinstance(pv, bool) and \
           isinstance(tv, (int, float)) and not isinstance(tv, bool):
            d = float(pv) - float(tv)
            score += d * d
    return score


def forward_sweep_invert(
    target_substrate: SubstrateState,
    field: TranslationField,
    tau_obs: float,
    canonical_grid: np.ndarray,
    *,
    score_fn: Optional[Callable[[SubstrateState, SubstrateState], float]] = None,
    forward_map: Optional[Callable[[CanonicalState, float], SubstrateState]] = None,
    return_residual_field: bool = False,
) -> Union[
    tuple[CanonicalState, float],
    tuple[CanonicalState, float, np.ndarray],
]:
    """Substrate observation -> canonical state at tau_obs, via forward search.

    Per §Q13: replaces the structurally ill-posed backward map. Each
    canonical_grid candidate is forward-projected through the translation
    field; the candidate with the smallest score against `target_substrate`
    wins.

    canonical_grid: numpy array of shape (N, 2), columns [chit, gamma_AB].

    score_fn: optional substrate-comparison callable. Defaults to
        L2 over shared numeric keys in observables + axes.

    forward_map: optional override for the canonical->substrate forward
        callable. Defaults to `apply_translation(...)` on the supplied
        field. Test fixtures pass an analytical forward map here (handoff
        §C.2 step 1).

    return_residual_field: if True, also return the per-candidate residual
        array. Consumers compute conditioning estimates from this (the
        bootstrap §7 caveat to ship the residual field).
    """
    if canonical_grid.ndim != 2 or canonical_grid.shape[1] != 2:
        raise ValueError(
            f"canonical_grid must have shape (N, 2); got {canonical_grid.shape}"
        )
    score_fn = score_fn or _default_substrate_score
    if forward_map is None:
        index = TranslationFieldIndex(field)
        def forward_map(c: CanonicalState, t: float) -> SubstrateState:  # type: ignore[misc]
            return apply_translation(c, index, t)

    n = canonical_grid.shape[0]
    residuals = np.empty(n, dtype=np.float64)
    for i in range(n):
        candidate = CanonicalState(
            chit=float(canonical_grid[i, 0]),
            gamma_AB=float(canonical_grid[i, 1]),
        )
        predicted = forward_map(candidate, tau_obs)
        residuals[i] = score_fn(predicted, target_substrate)

    best_idx = int(np.argmin(residuals))
    best_state = CanonicalState(
        chit=float(canonical_grid[best_idx, 0]),
        gamma_AB=float(canonical_grid[best_idx, 1]),
    )
    best_residual = float(math.sqrt(residuals[best_idx]))
    if return_residual_field:
        return best_state, best_residual, residuals
    return best_state, best_residual


# ---------------------------------------------------------------------------
# Op 3: tau_obs_sweep (per-frame fan-out; handoff §B.2 s->r traversal)
# ---------------------------------------------------------------------------

def tau_obs_sweep(
    target_substrates: Union[SubstrateState, list[SubstrateState]],
    field: TranslationField,
    tau_obs_grid: np.ndarray,
    canonical_search_grid: np.ndarray,
    *,
    score_fn: Optional[Callable[[SubstrateState, SubstrateState], float]] = None,
    forward_map: Optional[Callable[[CanonicalState, float], SubstrateState]] = None,
) -> list[CanonicalState]:
    """Walk the RG-flow trajectory across tau_obs.

    Per §B.1 the order constraint: tau_obs is declared before any projection.
    The traversal is a fan-out — one forward_sweep_invert per frame — not a
    pipeline applied once.

    `target_substrates` may be a single SubstrateState (the camera-test case
    where a single substrate observation is re-imaged through the tau_obs
    sweep) or a list-per-frame (the general case where each frame carries
    its own window-averaged observation).
    """
    if isinstance(target_substrates, SubstrateState):
        targets = [target_substrates] * len(tau_obs_grid)
    else:
        targets = list(target_substrates)
        if len(targets) != len(tau_obs_grid):
            raise ValueError(
                f"per-frame target list length {len(targets)} != "
                f"tau_obs_grid length {len(tau_obs_grid)}"
            )

    # Build the field index once; reuse across frames.
    index = TranslationFieldIndex(field) if forward_map is None else None

    trajectory: list[CanonicalState] = []
    for i, tau in enumerate(tau_obs_grid):
        if forward_map is None:
            assert index is not None
            def fm(c: CanonicalState, t: float, _idx=index) -> SubstrateState:
                return apply_translation(c, _idx, t)
            state, _ = forward_sweep_invert(
                targets[i], field, float(tau), canonical_search_grid,
                score_fn=score_fn, forward_map=fm,
            )
        else:
            state, _ = forward_sweep_invert(
                targets[i], field, float(tau), canonical_search_grid,
                score_fn=score_fn, forward_map=forward_map,
            )
        trajectory.append(state)
    return trajectory


# ---------------------------------------------------------------------------
# Op 4: regime_at (5-bucket per §C.4)
# ---------------------------------------------------------------------------

def regime_at(canonical: CanonicalState, tau_obs: float) -> RegimeReading:
    """Five-bucket vertex regime at this tau_obs frame (handoff §C.4).

    Per the auditor's gfdr_model.js canonical classifier:
        chit >= 0.7   : deep_c
        chit >= 0.2   : c_near_s
        |chit|  < 0.2 : s_critical
        chit > -0.7   : r_near_s
        else          : deep_r

    tau_obs argument is accepted for traceability and future tau-conditional
    classifiers (RFC-S Appendix B item 4 territory); v0 ignores it.
    """
    _ = tau_obs
    return RegimeReading(regime=vertex_regime(canonical.chit), k_frust=canonical.k_frust)


def regime_display_band(regime: RegimeLabel) -> DisplayBand:
    """Display-only collapse from 5-bucket to 3-bucket (handoff §C.4)."""
    if regime in ("deep_c", "c_near_s"):
        return "c"
    if regime == "s_critical":
        return "s"
    return "r"


# ---------------------------------------------------------------------------
# Op 5: gamut_classify
# ---------------------------------------------------------------------------

def gamut_classify(
    canonical: CanonicalState,
    tau_obs: float,
    gamut: GamutSpec,
) -> dict[str, Any]:
    """In-gamut / out-of-gamut diagnosis (RFC-S §2).

    Returns:
      {"in_gamut": bool, "diagnoses": [{axis, value, range, distance}, ...]}

    `diagnoses` is empty when in-gamut. Each entry names an axis where the
    canonical state lies outside the gamut range, with the distance to the
    nearer bound.
    """
    diagnoses: list[dict[str, Any]] = []

    def _diag(axis: str, value: float, rng: tuple[float, float]) -> dict[str, Any]:
        return {
            "axis": axis,
            "value": value,
            "range": rng,
            "distance": min(abs(value - rng[0]), abs(value - rng[1])),
        }

    if not (gamut.chit_range[0] <= canonical.chit <= gamut.chit_range[1]):
        diagnoses.append(_diag("chit", canonical.chit, gamut.chit_range))
    if not (gamut.gamma_AB_range[0] <= canonical.gamma_AB <= gamut.gamma_AB_range[1]):
        diagnoses.append(_diag("gamma_AB", canonical.gamma_AB, gamut.gamma_AB_range))
    if gamut.tau_obs_range is not None and not (
        gamut.tau_obs_range[0] <= tau_obs <= gamut.tau_obs_range[1]
    ):
        diagnoses.append(_diag("tau_obs", tau_obs, gamut.tau_obs_range))

    return {"in_gamut": len(diagnoses) == 0, "diagnoses": diagnoses}


# ---------------------------------------------------------------------------
# Op 6: intent_map (I5 only at v0)
# ---------------------------------------------------------------------------

def intent_map(
    out_of_gamut: CanonicalState,
    tau_obs: float,
    gamut: GamutSpec,
    intent_id: str,
) -> tuple[CanonicalState, dict[str, Any]]:
    """Map an out-of-gamut canonical state to in-gamut per the chosen intent.

    Per RFC-S §3: scale uniformly along the gamut to fit, preserving the
    named invariant. v0 implements only I5 (signature-preserving).

    Returns (mapped_state, sacrifice_record).
    """
    if intent_id not in {"I1", "I2", "I3", "I4", "I5"}:
        raise ValueError(f"unknown intent: {intent_id!r}")
    if intent_id != "I5":
        raise NotImplementedError(
            f"intent {intent_id} not implemented in v0 (I5-only)"
        )

    original_regime = regime_at(out_of_gamut, tau_obs).regime
    chit = float(np.clip(
        out_of_gamut.chit, gamut.chit_range[0], gamut.chit_range[1]
    ))
    gamma = float(np.clip(
        out_of_gamut.gamma_AB, gamut.gamma_AB_range[0], gamut.gamma_AB_range[1]
    ))
    mapped = CanonicalState(chit=chit, gamma_AB=gamma, k_frust=out_of_gamut.k_frust)
    mapped_regime = regime_at(mapped, tau_obs).regime

    sacrifice = {
        "intent": "I5",
        "delta_chit": chit - out_of_gamut.chit,
        "delta_gamma_AB": gamma - out_of_gamut.gamma_AB,
        "regime_preserved": original_regime == mapped_regime,
        "original_regime": original_regime,
        "mapped_regime": mapped_regime,
    }
    return mapped, sacrifice


# ---------------------------------------------------------------------------
# Op 7: validate_driver_profile (RFC-S §5 round-trip)
# ---------------------------------------------------------------------------

def validate_driver_profile(
    field: TranslationField,
    reference_dataset: list[dict[str, Any]],
    canonical_search_grid: np.ndarray,
    *,
    intent_id: str = "I5",
) -> dict[str, Any]:
    """RFC-S §5 round-trip validation.

    Each entry of `reference_dataset` is a dict with:
        canonical_state: CanonicalState (the truth)
        tau_obs: float
        expected_substrate: SubstrateState | None (optional; auto-computed
            by apply_translation when None)

    Returns a per-entry residual record plus aggregate stats. v0 reports I5
    (signature-preserving) regime agreement as the universality-class check.
    """
    if intent_id != "I5":
        raise NotImplementedError(
            f"validate_driver_profile intent {intent_id} not implemented in v0"
        )

    index = TranslationFieldIndex(field)
    forward_residuals: list[float] = []
    round_trip_residuals: list[float] = []
    regime_agreements: list[bool] = []

    for entry in reference_dataset:
        canonical: CanonicalState = entry["canonical_state"]
        tau_obs: float = float(entry["tau_obs"])
        expected: Optional[SubstrateState] = entry.get("expected_substrate")

        predicted = apply_translation(canonical, index, tau_obs)
        if expected is not None:
            fwd_err = _default_substrate_score(predicted, expected)
        else:
            fwd_err = 0.0
        forward_residuals.append(math.sqrt(fwd_err))

        recovered, _ = forward_sweep_invert(
            predicted, field, tau_obs, canonical_search_grid,
        )
        rt_err = math.sqrt(
            (recovered.chit - canonical.chit) ** 2
            + (recovered.gamma_AB - canonical.gamma_AB) ** 2
        )
        round_trip_residuals.append(rt_err)

        orig_r = regime_at(canonical, tau_obs).regime
        rec_r = regime_at(recovered, tau_obs).regime
        regime_agreements.append(orig_r == rec_r)

    return {
        "intent": intent_id,
        "forward_residuals": forward_residuals,
        "round_trip_residuals": round_trip_residuals,
        "regime_agreements": regime_agreements,
        "forward_mean": float(np.mean(forward_residuals)) if forward_residuals else 0.0,
        "round_trip_mean": float(np.mean(round_trip_residuals)) if round_trip_residuals else 0.0,
        "regime_agreement_rate": float(np.mean(regime_agreements)) if regime_agreements else 0.0,
    }


# ---------------------------------------------------------------------------
# Driver-profile JSON loader (consumer convenience)
# ---------------------------------------------------------------------------

def parse_translation_field(d: dict[str, Any]) -> TranslationField:
    """Parse a driver-profile.v2.0 `translation_field` block into types.

    Tolerant of the seed-corpus convention where operating-point axes ride
    `additionalProperties` (label, gt, plus arbitrary other keys).
    """
    rules: list[TranslationRule] = []
    for r in d.get("rule", []):
        op_d = r["operating_point"]
        known = {"label", "gt"}
        axes = {k: v for k, v in op_d.items() if k not in known}
        op = OperatingPoint(label=op_d["label"], gt=op_d["gt"], axes=axes)

        c_d = r["canonical"]
        c_known = {"chit", "gamma_AB", "k_frust", "method"}
        c_extras = {k: v for k, v in c_d.items() if k not in c_known}
        canonical = CanonicalPoint(
            chit=float(c_d["chit"]),
            gamma_AB=float(c_d["gamma_AB"]),
            k_frust=bool(c_d["k_frust"]),
            method=str(c_d["method"]),
            extras=c_extras,
        )
        rules.append(TranslationRule(
            operating_point=op,
            xdot_choice=str(r["xdot_choice"]),
            canonical=canonical,
        ))
    return TranslationField(
        direction=d.get("direction", "forward"),
        shape=d.get("shape", "lookup_table"),
        rule=rules,
        description=d.get("description"),
    )


def parse_gamut(d: dict[str, Any]) -> GamutSpec:
    """Parse a driver-profile.v2.0 `gamut` block into a GamutSpec.

    The seed-corpus profiles ship `chit_range` but not gamma_AB_range or
    tau_obs_range; we default gamma_AB_range to the seed convention
    [-1.0, 1.0] when absent.
    """
    chit_range = tuple(d.get("chit_range", [-1.0, 1.0]))
    gamma_range = tuple(d.get("gamma_AB_range", [-1.0, 1.0]))
    tau_range = d.get("tau_obs_range")
    if tau_range is not None:
        tau_range = tuple(tau_range)
    return GamutSpec(
        chit_range=(float(chit_range[0]), float(chit_range[1])),
        gamma_AB_range=(float(gamma_range[0]), float(gamma_range[1])),
        tau_obs_range=tau_range,
        out_of_scope_residual_threshold=float(
            d.get("out_of_scope_residual_threshold", 0.05)
        ),
    )
