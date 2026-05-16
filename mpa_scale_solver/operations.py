"""The seven scale-solver operations (handoff §A.4).

All stateless free functions on plain dataclasses. Per §A.3, the operations
take tau_obs as an explicit argument rather than reading it off the state.

v0 production translation-field shape is `lookup_table`. v1 adds the
`tangent_flow` shape via `TangentFlowField`; `apply_translation` dispatches
on `field.shape`. The parametric path from the prior reference (aging_log,
trivial_baseline) lives in `_test_fixtures.py` and is used only by the
camera test.

Per §C.4 the canonical regime classifier is the FIVE-bucket cut from
gfdr_model.vertex_regime. The three-bucket cut (`regime_display_band`) is a
display-only helper.

v1 adds seven `*_wrapped` variants returning `OperationOutput[T]` with
validation + provenance riding alongside the value. v0 signatures are
unchanged (handoff §A.2 back-compat).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional, Union

import numpy as np

from . import validation as _validation
from .gfdr_model import vertex_regime
from .provenance import make_provenance
from .sidecar import lookup_forward, lookup_inverse
from .types import (
    AnyTranslationField,
    CanonicalPoint,
    CanonicalState,
    DisplayBand,
    DispatchPath,
    GamutSpec,
    InverseLookupSidecar,
    OperatingPoint,
    OperationOutput,
    RegimeLabel,
    RegimeReading,
    SubstrateState,
    TangentFlowField,
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
    field: Union[TranslationField, TangentFlowField, "TranslationFieldIndex"],
    tau_obs: float,
    *,
    domain_distance_threshold: float = DEFAULT_DOMAIN_DISTANCE_THRESHOLD,
    tau_obs_weight: float = 1.0,
) -> SubstrateState:
    """Forward map: canonical state -> substrate-native at tau_obs.

    Per §Q13 the only well-defined direction is forward; the backward map
    (substrate -> canonical) is handled by forward_sweep_invert.

    Dispatch on `field.shape`:
      - `lookup_table` (v0): find the rule whose
        (canonical.chit, canonical.gamma_AB) and operating-point tau_obs
        (if any) are L2-nearest; return a SubstrateState carrying the
        matched rule's operating-point identity.
      - `tangent_flow` (v1): apply the ScalingRule closed form and project
        through `rule_at_origin`.

    Raises ValueError for lookup_table when the nearest rule is beyond
    `domain_distance_threshold` — the curator-path's signal that the
    declared driver profile does not cover this substrate state (a gamut
    violation handled upstream).
    """
    if isinstance(field, TangentFlowField):
        return _apply_tangent_flow(canonical, field, tau_obs)
    return _apply_lookup(
        canonical, field, tau_obs,
        domain_distance_threshold=domain_distance_threshold,
        tau_obs_weight=tau_obs_weight,
    )


def _apply_lookup(
    canonical: CanonicalState,
    field: Union[TranslationField, "TranslationFieldIndex"],
    tau_obs: float,
    *,
    domain_distance_threshold: float,
    tau_obs_weight: float,
) -> SubstrateState:
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


def _apply_tangent_flow(
    canonical: CanonicalState,
    field: TangentFlowField,
    tau_obs: float,
) -> SubstrateState:
    """Tangent-flow forward map (handoff §C.2).

    Scales the canonical state via the ScalingRule closed form, then
    packages it as a substrate observation labeled by `rule_at_origin`.
    For the Banach default (delta_chit = delta_gamma = 0) this is the
    identity translation: substrate observables equal canonical values.
    """
    rule = field.scaling
    if tau_obs <= 0.0 or rule.tau_obs_ref <= 0.0:
        scaled_chit = canonical.chit
        scaled_gamma = canonical.gamma_AB
    else:
        ratio = tau_obs / rule.tau_obs_ref
        scaled_chit = canonical.chit + rule.delta_chit * math.log(ratio)
        scaled_gamma = canonical.gamma_AB * (ratio ** rule.delta_gamma)
    origin = field.rule_at_origin
    axes = dict(origin.operating_point.axes)
    axes["tau_obs"] = tau_obs
    return SubstrateState(
        tau_obs=tau_obs,
        label=origin.operating_point.label,
        axes=axes,
        observables={
            "substrate_chit": scaled_chit,
            "substrate_gamma_AB": scaled_gamma,
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
        if isinstance(field, TangentFlowField):
            # Tangent-flow apply_translation is closed-form; no index.
            _field = field
            def forward_map(c: CanonicalState, t: float) -> SubstrateState:  # type: ignore[misc]
                return apply_translation(c, _field, t)
        else:
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

    # Build the field index once for lookup_table; reuse across frames.
    # Tangent-flow has no index — apply_translation dispatches directly.
    if forward_map is None and not isinstance(field, TangentFlowField):
        index: Optional[TranslationFieldIndex] = TranslationFieldIndex(field)
    else:
        index = None

    trajectory: list[CanonicalState] = []
    for i, tau in enumerate(tau_obs_grid):
        if forward_map is None:
            if index is not None:
                def fm(c: CanonicalState, t: float, _idx=index) -> SubstrateState:
                    return apply_translation(c, _idx, t)
            else:
                def fm(c: CanonicalState, t: float, _field=field) -> SubstrateState:
                    return apply_translation(c, _field, t)
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


# ---------------------------------------------------------------------------
# v1: wrapped variants (handoff §A.2, §C.5, §C.6)
# ---------------------------------------------------------------------------
#
# Each `*_wrapped` calls the matching v0 operation, then stamps a
# ValidationReport + Provenance onto an OperationOutput[T]. Sidecar
# dispatch (handoff §C.4) is opt-in via the `sidecar` kwarg and is
# meaningful for `apply_translation_wrapped`, `forward_sweep_invert_wrapped`,
# and `tau_obs_sweep_wrapped`. The remaining four operations have no
# sidecar dispatch — their wrapped variants only attach validation +
# provenance.


def apply_translation_wrapped(
    canonical: CanonicalState,
    field: AnyTranslationField,
    tau_obs: float,
    *,
    domain_distance_threshold: float = DEFAULT_DOMAIN_DISTANCE_THRESHOLD,
    tau_obs_weight: float = 1.0,
    sidecar: Optional[InverseLookupSidecar] = None,
) -> OperationOutput[SubstrateState]:
    """Wrapped variant of `apply_translation` (handoff §A.2 / §C.5)."""
    dispatch = DispatchPath.DIRECT_COMPUTE
    table_version: Optional[str] = None
    if sidecar is not None:
        hit = lookup_forward(sidecar, canonical, tau_obs)
        if hit is not None:
            substrate = hit
            dispatch = DispatchPath.TABLE_HIT
        else:
            substrate = apply_translation(
                canonical, field, tau_obs,
                domain_distance_threshold=domain_distance_threshold,
                tau_obs_weight=tau_obs_weight,
            )
            dispatch = DispatchPath.COMPUTE_FALLBACK
        table_version = sidecar.version
    else:
        substrate = apply_translation(
            canonical, field, tau_obs,
            domain_distance_threshold=domain_distance_threshold,
            tau_obs_weight=tau_obs_weight,
        )
    report = _validation.report_for_apply_translation(canonical, substrate)
    prov = make_provenance(
        "apply_translation",
        dispatch_path=dispatch,
        table_version=table_version,
    )
    return OperationOutput(value=substrate, validation=report, provenance=prov)


def forward_sweep_invert_wrapped(
    target_substrate: SubstrateState,
    field: AnyTranslationField,
    tau_obs: float,
    canonical_grid: np.ndarray,
    *,
    score_fn: Optional[Callable[[SubstrateState, SubstrateState], float]] = None,
    forward_map: Optional[Callable[[CanonicalState, float], SubstrateState]] = None,
    return_residual_field: bool = False,
    sidecar: Optional[InverseLookupSidecar] = None,
    compute_round_trip: bool = True,
) -> OperationOutput[CanonicalState]:
    """Wrapped variant of `forward_sweep_invert`.

    Sidecar dispatch is table-first: an inverse-table hit returns the
    recorded canonical with `dispatch_path = TABLE_HIT`; on miss the
    brute-force grid search runs with `dispatch_path = COMPUTE_FALLBACK`.

    `compute_round_trip` controls whether the wrapped variant runs a
    forward-then-back recovery for the validation report's
    `round_trip_residual`. Default True; turn off in tight inner loops.
    """
    dispatch = DispatchPath.DIRECT_COMPUTE
    table_version: Optional[str] = None
    residual_field: Optional[np.ndarray] = None
    if sidecar is not None:
        table_version = sidecar.version
        hit = lookup_inverse(sidecar, target_substrate, tau_obs)
        if hit is not None:
            recovered = hit
            dispatch = DispatchPath.TABLE_HIT
        else:
            result = forward_sweep_invert(
                target_substrate, field, tau_obs, canonical_grid,
                score_fn=score_fn, forward_map=forward_map,
                return_residual_field=return_residual_field,
            )
            recovered = result[0]
            if return_residual_field:
                residual_field = result[2]
            dispatch = DispatchPath.COMPUTE_FALLBACK
    else:
        result = forward_sweep_invert(
            target_substrate, field, tau_obs, canonical_grid,
            score_fn=score_fn, forward_map=forward_map,
            return_residual_field=return_residual_field,
        )
        recovered = result[0]
        if return_residual_field:
            residual_field = result[2]

    rt_residual: Optional[float] = None
    if compute_round_trip:
        # Forward-then-back via the same translation field. Skipped for
        # tangent-flow fields when delta=0 (rt would be trivially 0 by
        # construction) but the call is harmless.
        try:
            forward_back = apply_translation(recovered, field, tau_obs)
            rt_residual = math.sqrt(_default_substrate_score(forward_back, target_substrate))
        except ValueError:
            rt_residual = float("inf")

    report = _validation.report_for_forward_sweep_invert(
        target_substrate, recovered, round_trip_residual=rt_residual,
    )
    if residual_field is not None:
        report = report  # residual_field is a v0 return-shape concern, not part of validation
    prov = make_provenance(
        "forward_sweep_invert",
        dispatch_path=dispatch,
        table_version=table_version,
    )
    return OperationOutput(value=recovered, validation=report, provenance=prov)


def tau_obs_sweep_wrapped(
    target_substrates: Union[SubstrateState, list[SubstrateState]],
    field: AnyTranslationField,
    tau_obs_grid: np.ndarray,
    canonical_search_grid: np.ndarray,
    *,
    score_fn: Optional[Callable[[SubstrateState, SubstrateState], float]] = None,
    forward_map: Optional[Callable[[CanonicalState, float], SubstrateState]] = None,
    sidecar: Optional[InverseLookupSidecar] = None,
) -> OperationOutput[list[CanonicalState]]:
    """Wrapped variant of `tau_obs_sweep`.

    Per-frame dispatch via `forward_sweep_invert_wrapped`. The aggregate
    `provenance.dispatch_path` is `TABLE_HIT` only when every frame hit
    the table; otherwise `DIRECT_COMPUTE` and the per-frame mix is
    summarized in `notes`.
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

    trajectory: list[CanonicalState] = []
    n_table = 0
    n_fallback = 0
    n_direct = 0
    for i, tau in enumerate(tau_obs_grid):
        out = forward_sweep_invert_wrapped(
            targets[i], field, float(tau), canonical_search_grid,
            score_fn=score_fn, forward_map=forward_map,
            sidecar=sidecar, compute_round_trip=False,
        )
        trajectory.append(out.value)
        if out.provenance.dispatch_path == DispatchPath.TABLE_HIT:
            n_table += 1
        elif out.provenance.dispatch_path == DispatchPath.COMPUTE_FALLBACK:
            n_fallback += 1
        else:
            n_direct += 1

    aggregate = (
        DispatchPath.TABLE_HIT if n_table == len(tau_obs_grid)
        else DispatchPath.DIRECT_COMPUTE
    )
    notes = (
        f"frames: table_hit={n_table}, compute_fallback={n_fallback}, "
        f"direct_compute={n_direct}",
    )
    report = _validation.report_for_tau_obs_sweep(trajectory)
    prov = make_provenance(
        "tau_obs_sweep",
        dispatch_path=aggregate,
        table_version=(sidecar.version if sidecar is not None else None),
        notes=notes,
    )
    return OperationOutput(value=trajectory, validation=report, provenance=prov)


def regime_at_wrapped(
    canonical: CanonicalState,
    tau_obs: float,
) -> OperationOutput[RegimeReading]:
    reading = regime_at(canonical, tau_obs)
    report = _validation.report_for_regime_at(canonical)
    prov = make_provenance("regime_at")
    return OperationOutput(value=reading, validation=report, provenance=prov)


def gamut_classify_wrapped(
    canonical: CanonicalState,
    tau_obs: float,
    gamut: GamutSpec,
) -> OperationOutput[dict[str, Any]]:
    result = gamut_classify(canonical, tau_obs, gamut)
    report = _validation.report_for_gamut_classify(canonical)
    prov = make_provenance("gamut_classify")
    return OperationOutput(value=result, validation=report, provenance=prov)


def intent_map_wrapped(
    out_of_gamut: CanonicalState,
    tau_obs: float,
    gamut: GamutSpec,
    intent_id: str,
) -> OperationOutput[tuple[CanonicalState, dict[str, Any]]]:
    mapped, sacrifice = intent_map(out_of_gamut, tau_obs, gamut, intent_id)
    report = _validation.report_for_intent_map(out_of_gamut, mapped, sacrifice)
    prov = make_provenance("intent_map")
    return OperationOutput(value=(mapped, sacrifice), validation=report, provenance=prov)


def validate_driver_profile_wrapped(
    field: AnyTranslationField,
    reference_dataset: list[dict[str, Any]],
    canonical_search_grid: np.ndarray,
    *,
    intent_id: str = "I5",
) -> OperationOutput[dict[str, Any]]:
    summary = validate_driver_profile(
        field, reference_dataset, canonical_search_grid, intent_id=intent_id,
    )
    report = _validation.report_for_validate_driver_profile(summary)
    prov = make_provenance("validate_driver_profile")
    return OperationOutput(value=summary, validation=report, provenance=prov)
