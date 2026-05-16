"""v3 — Cross-substrate operations + per-intent RFC-S §5 metrics (BLOCK_IN §v3).

Three cross-substrate compositions:
  - gamut_overlap
  - canonical_distance
  - universality_agreement

Plus per-intent metric tightening in validate_driver_profile (the RFC-S §5
table; per BLOCK_IN §v3 these land alongside the cross-substrate ops
since both depend on the same metric structure).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mpa_scale_solver import (
    CanonicalState,
    GamutSpec,
    OperatingPoint,
    ScalingRule,
    TangentFlowField,
    TranslationField,
    TranslationRule,
    canonical_distance,
    canonical_distance_wrapped,
    gamut_overlap,
    gamut_overlap_wrapped,
    universality_agreement,
    universality_agreement_wrapped,
    validate_driver_profile,
)
from mpa_scale_solver.types import CanonicalPoint
from mpa_scale_solver.validation import (
    aggregate_per_intent_metrics,
    per_intent_cell_metric,
)


def _gamut(chit=(-1.0, 1.0), gamma=(-1.0, 1.0), tau=None) -> GamutSpec:
    return GamutSpec(
        chit_range=chit,
        gamma_AB_range=gamma,
        tau_obs_range=tau,
    )


def _tangent_identity_field() -> TangentFlowField:
    origin = TranslationRule(
        operating_point=OperatingPoint(label="origin", gt="c", axes={"tau_obs": 1.0}),
        xdot_choice="default",
        canonical=CanonicalPoint(chit=0.0, gamma_AB=0.0, k_frust=False, method="tangent_flow"),
    )
    return TangentFlowField(
        direction="forward",
        shape="tangent_flow",
        rule_at_origin=origin,
        scaling=ScalingRule(tau_obs_ref=1.0, delta_chit=0.0, delta_gamma=0.0),
    )


def _lookup_field(rule_canonicals: list[tuple[float, float, str]]) -> TranslationField:
    """Build a lookup_table field from (chit, gamma, label) triples."""
    rules = []
    for c, g, label in rule_canonicals:
        rules.append(TranslationRule(
            operating_point=OperatingPoint(label=label, gt="c", axes={"tau_obs": 1.0}),
            xdot_choice="default",
            canonical=CanonicalPoint(chit=c, gamma_AB=g, k_frust=False, method="lookup"),
        ))
    return TranslationField(
        direction="forward",
        shape="lookup_table",
        rule=rules,
    )


# ---------------------------------------------------------------------------
# gamut_overlap
# ---------------------------------------------------------------------------

class TestGamutOverlap:
    def test_identical_gamuts_jaccard_one(self):
        g = _gamut(chit=(-0.5, 0.5), gamma=(-0.5, 0.5))
        result = gamut_overlap(g, g)
        assert result["compatible"] is True
        assert result["jaccard"] == pytest.approx(1.0)
        assert result["chit_intersection"] == (-0.5, 0.5)
        assert result["gamma_AB_intersection"] == (-0.5, 0.5)

    def test_disjoint_chit_gamuts(self):
        a = _gamut(chit=(-0.5, -0.1), gamma=(-1.0, 1.0))
        b = _gamut(chit=(0.1, 0.5), gamma=(-1.0, 1.0))
        result = gamut_overlap(a, b)
        assert result["compatible"] is False
        assert result["chit_intersection"] is None
        assert result["intersection_area"] == 0.0
        assert result["jaccard"] == 0.0

    def test_partial_overlap_jaccard_math(self):
        # 1x1 squares overlapping in a 0.5x0.5 corner
        a = _gamut(chit=(0.0, 1.0), gamma=(0.0, 1.0))
        b = _gamut(chit=(0.5, 1.5), gamma=(0.5, 1.5))
        result = gamut_overlap(a, b)
        assert result["intersection_area"] == pytest.approx(0.25)
        # union = 1 + 1 - 0.25 = 1.75
        assert result["jaccard"] == pytest.approx(0.25 / 1.75)

    def test_tau_obs_intersection_when_both_present(self):
        a = _gamut(tau=(0.1, 1.0))
        b = _gamut(tau=(0.5, 2.0))
        result = gamut_overlap(a, b)
        assert result["tau_obs_intersection"] == (0.5, 1.0)

    def test_tau_obs_intersection_none_when_either_absent(self):
        a = _gamut(tau=(0.1, 1.0))
        b = _gamut()  # tau_obs_range=None
        result = gamut_overlap(a, b)
        assert result["tau_obs_intersection"] is None

    def test_wrapped_carries_validation_and_provenance(self):
        result = gamut_overlap_wrapped(
            _gamut(chit=(-0.5, 0.5)),
            _gamut(chit=(-0.5, 0.5)),
        )
        assert result.value["jaccard"] == pytest.approx(1.0)
        assert result.provenance.operation == "gamut_overlap"


# ---------------------------------------------------------------------------
# canonical_distance
# ---------------------------------------------------------------------------

class TestCanonicalDistance:
    def test_l2_metric(self):
        a = CanonicalState(chit=0.0, gamma_AB=0.0)
        b = CanonicalState(chit=3.0, gamma_AB=4.0)
        assert canonical_distance(a, b, "l2") == pytest.approx(5.0)

    def test_l1_metric(self):
        a = CanonicalState(chit=0.1, gamma_AB=0.2)
        b = CanonicalState(chit=0.3, gamma_AB=0.5)
        assert canonical_distance(a, b, "l1") == pytest.approx(0.2 + 0.3)

    def test_regime_metric_same(self):
        # Both deep_c
        a = CanonicalState(chit=0.8, gamma_AB=0.0)
        b = CanonicalState(chit=0.9, gamma_AB=0.0)
        assert canonical_distance(a, b, "regime") == 0.0

    def test_regime_metric_different(self):
        # deep_c vs c_near_s
        a = CanonicalState(chit=0.8, gamma_AB=0.0)
        b = CanonicalState(chit=0.5, gamma_AB=0.0)
        assert canonical_distance(a, b, "regime") == 1.0

    def test_universality_metric_matches_all_three(self):
        a = CanonicalState(chit=0.8, gamma_AB=0.3, k_frust=True)
        b = CanonicalState(chit=0.9, gamma_AB=0.2, k_frust=True)
        assert canonical_distance(a, b, "universality") == 0.0

    def test_universality_metric_breaks_on_k_frust(self):
        a = CanonicalState(chit=0.8, gamma_AB=0.3, k_frust=True)
        b = CanonicalState(chit=0.8, gamma_AB=0.3, k_frust=False)
        assert canonical_distance(a, b, "universality") == 1.0

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError, match="unknown metric"):
            canonical_distance(
                CanonicalState(chit=0.0, gamma_AB=0.0),
                CanonicalState(chit=1.0, gamma_AB=1.0),
                "cosine",
            )

    def test_wrapped_stamps_metric_in_provenance(self):
        out = canonical_distance_wrapped(
            CanonicalState(chit=0.5, gamma_AB=0.5),
            CanonicalState(chit=0.5, gamma_AB=0.5),
            "l1",
        )
        assert out.value == 0.0
        assert "metric=l1" in out.provenance.notes[0]


# ---------------------------------------------------------------------------
# universality_agreement
# ---------------------------------------------------------------------------

class TestUniversalityAgreement:
    def test_identical_profiles_full_agreement(self):
        field = _tangent_identity_field()
        gamut = _gamut(chit=(-0.5, 0.5), gamma=(-0.5, 0.5))
        grid = np.array(
            [[c, g] for c in [0.3, 0.1, -0.1, -0.3]
             for g in [0.2, -0.2]],
            dtype=np.float64,
        )
        result = universality_agreement(
            field, gamut, field, gamut,
            grid, tau_obs=1.0,
        )
        assert result["n_compared"] == 8
        assert result["n_agreed"] == 8
        assert result["agreement_rate"] == pytest.approx(1.0)

    def test_disjoint_gamuts_no_comparison(self):
        field_a = _tangent_identity_field()
        field_b = _tangent_identity_field()
        gamut_a = _gamut(chit=(-1.0, -0.5))
        gamut_b = _gamut(chit=(0.5, 1.0))
        grid = np.array([[0.0, 0.0], [0.2, 0.1]], dtype=np.float64)
        result = universality_agreement(
            field_a, gamut_a, field_b, gamut_b,
            grid, tau_obs=1.0,
        )
        assert result["n_compared"] == 0
        assert result["agreement_rate"] == 0.0

    def test_per_class_counts_populated(self):
        field = _tangent_identity_field()
        gamut = _gamut(chit=(-1.0, 1.0))
        # mix of deep_c, c_near_s, s_critical points
        grid = np.array([[0.8, 0.0], [0.5, 0.0], [0.0, 0.0]], dtype=np.float64)
        result = universality_agreement(field, gamut, field, gamut, grid, tau_obs=1.0)
        assert result["agreement_rate"] == pytest.approx(1.0)
        assert set(result["per_class_counts"].keys()) >= {"deep_c", "c_near_s", "s_critical"}
        for bucket in result["per_class_counts"].values():
            assert bucket["disagree"] == 0

    def test_wrapped_agreement(self):
        field = _tangent_identity_field()
        gamut = _gamut(chit=(-0.5, 0.5))
        grid = np.array([[0.0, 0.0], [0.3, 0.0]], dtype=np.float64)
        out = universality_agreement_wrapped(
            field, gamut, field, gamut, grid, tau_obs=1.0,
        )
        assert out.value["agreement_rate"] == pytest.approx(1.0)
        assert out.provenance.operation == "universality_agreement"


# ---------------------------------------------------------------------------
# Per-intent RFC-S §5 metric helpers
# ---------------------------------------------------------------------------

class TestPerIntentCellMetric:
    def test_i1_hamming_perfect_match(self):
        a = CanonicalState(chit=0.8, gamma_AB=0.3, k_frust=True)
        b = CanonicalState(chit=0.8, gamma_AB=0.3, k_frust=True)
        m = per_intent_cell_metric("I1", a, b)
        assert m["hamming"] == 0
        assert m["regime_match"] is True
        assert m["edge_type_match"] is True
        assert m["k_frust_match"] is True

    def test_i1_hamming_regime_drift(self):
        a = CanonicalState(chit=0.8, gamma_AB=0.3)
        b = CanonicalState(chit=0.5, gamma_AB=0.3)
        m = per_intent_cell_metric("I1", a, b)
        assert m["regime_match"] is False
        assert m["hamming"] == 1

    def test_i2_l2_drive(self):
        a = CanonicalState(chit=0.0, gamma_AB=0.0)
        b = CanonicalState(chit=3.0, gamma_AB=4.0)
        m = per_intent_cell_metric("I2", a, b)
        assert m["l2_drive"] == pytest.approx(5.0)
        assert m["gamma_deviation"] == pytest.approx(4.0)

    def test_i3_gamma_star_deviation(self):
        # |chit_a|=0.7 (on boundary), |chit_b|=0.9 -> gamma_star_dev = 0.2
        a = CanonicalState(chit=0.7, gamma_AB=0.0)
        b = CanonicalState(chit=0.9, gamma_AB=0.0)
        m = per_intent_cell_metric("I3", a, b)
        assert m["gamma_star_deviation"] == pytest.approx(0.2)
        assert m["capacity_class_match"] is True

    def test_i4_epsilon_sequence_distance_sign_match(self):
        a = CanonicalState(chit=0.0, gamma_AB=0.3)
        b = CanonicalState(chit=0.0, gamma_AB=0.5)
        m = per_intent_cell_metric("I4", a, b, in_gamut=True)
        assert m["epsilon_sequence_distance"] == 0
        assert m["survival"] is True

    def test_i4_epsilon_sequence_distance_sign_flip(self):
        a = CanonicalState(chit=0.0, gamma_AB=0.3)
        b = CanonicalState(chit=0.0, gamma_AB=-0.3)
        m = per_intent_cell_metric("I4", a, b)
        assert m["epsilon_sequence_distance"] == 1

    def test_i5_universality_class_match(self):
        a = CanonicalState(chit=0.8, gamma_AB=0.3)
        b = CanonicalState(chit=0.9, gamma_AB=0.4)  # both deep_c
        m = per_intent_cell_metric("I5", a, b)
        assert m["universality_class_match"] is True
        assert m["intra_class_l2"] is not None
        assert m["intra_class_l2"] == pytest.approx(math.sqrt(0.01 + 0.01))

    def test_unknown_intent_raises(self):
        with pytest.raises(ValueError, match="unknown intent"):
            per_intent_cell_metric(
                "I9",
                CanonicalState(chit=0.0, gamma_AB=0.0),
                CanonicalState(chit=0.0, gamma_AB=0.0),
            )


class TestAggregatePerIntentMetrics:
    def test_empty_cells(self):
        agg = aggregate_per_intent_metrics("I1", [])
        assert agg["n_cells"] == 0

    def test_i1_aggregate(self):
        cells = [
            {"hamming": 0, "regime_match": True, "edge_type_match": True, "k_frust_match": True},
            {"hamming": 1, "regime_match": False, "edge_type_match": True, "k_frust_match": True},
        ]
        agg = aggregate_per_intent_metrics("I1", cells)
        assert agg["hamming_rate"] == pytest.approx(0.5)
        assert agg["regime_match_rate"] == pytest.approx(0.5)

    def test_i2_aggregate(self):
        cells = [
            {"l2_drive": 1.0, "gamma_deviation": 0.5},
            {"l2_drive": 3.0, "gamma_deviation": 2.0},
        ]
        agg = aggregate_per_intent_metrics("I2", cells)
        assert agg["l2_drive_mean"] == pytest.approx(2.0)
        assert agg["l2_drive_max"] == pytest.approx(3.0)
        assert agg["gamma_deviation_max"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# validate_driver_profile per-intent integration
# ---------------------------------------------------------------------------

class TestValidateDriverProfilePerIntent:
    def _trivial_dataset_and_field(self):
        """Three canonicals at three distinct regimes; lookup-table field with rules covering them."""
        rules = []
        for chit, gamma in [(0.8, 0.3), (0.5, 0.1), (-0.5, -0.2)]:
            rules.append(TranslationRule(
                operating_point=OperatingPoint(
                    label=f"c={chit}", gt="c", axes={"tau_obs": 1.0},
                ),
                xdot_choice="default",
                canonical=CanonicalPoint(
                    chit=chit, gamma_AB=gamma, k_frust=False, method="lookup",
                ),
            ))
        field = TranslationField(direction="forward", shape="lookup_table", rule=rules)
        dataset = [
            {"canonical_state": CanonicalState(chit=0.8, gamma_AB=0.3), "tau_obs": 1.0},
            {"canonical_state": CanonicalState(chit=0.5, gamma_AB=0.1), "tau_obs": 1.0},
            {"canonical_state": CanonicalState(chit=-0.5, gamma_AB=-0.2), "tau_obs": 1.0},
        ]
        grid = np.array([[c, g] for c, g in [(0.8, 0.3), (0.5, 0.1), (-0.5, -0.2)]], dtype=np.float64)
        return field, dataset, grid

    def test_i5_per_intent_summary_present(self):
        field, dataset, grid = self._trivial_dataset_and_field()
        summary = validate_driver_profile(field, dataset, grid, intent_id="I5")
        assert "per_intent" in summary
        per = summary["per_intent"]
        assert per["intent"] == "I5"
        assert per["universality_class_agreement_rate"] == pytest.approx(1.0)

    def test_i1_per_intent_hamming(self):
        field, dataset, grid = self._trivial_dataset_and_field()
        summary = validate_driver_profile(field, dataset, grid, intent_id="I1")
        per = summary["per_intent"]
        assert per["intent"] == "I1"
        assert per["hamming_rate"] == pytest.approx(0.0)

    def test_i2_per_intent_l2_drive(self):
        field, dataset, grid = self._trivial_dataset_and_field()
        summary = validate_driver_profile(field, dataset, grid, intent_id="I2")
        per = summary["per_intent"]
        assert per["intent"] == "I2"
        assert per["l2_drive_mean"] == pytest.approx(0.0, abs=1e-9)

    def test_back_compat_keys_preserved(self):
        field, dataset, grid = self._trivial_dataset_and_field()
        summary = validate_driver_profile(field, dataset, grid, intent_id="I5")
        # v2.3 keys still present
        assert "regime_agreements" in summary
        assert "regime_agreement_rate" in summary
        assert "forward_residuals" in summary
        assert "round_trip_residuals" in summary

    def test_i4_with_gamut_records_survival(self):
        field, dataset, grid = self._trivial_dataset_and_field()
        gamut = _gamut(chit=(-1.0, 1.0), gamma=(-1.0, 1.0))
        summary = validate_driver_profile(field, dataset, grid, intent_id="I4", gamut=gamut)
        per = summary["per_intent"]
        assert per["intent"] == "I4"
        assert per["survival_rate"] == pytest.approx(1.0)
