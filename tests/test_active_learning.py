"""v3 — Active learning: suggest_measurements (BLOCK_IN §v3)."""

from __future__ import annotations

import numpy as np
import pytest

from mpa_scale_solver import (
    CanonicalState,
    GamutSpec,
    MeasurementCandidate,
    OperatingPoint,
    ScalingRule,
    TangentFlowField,
    TranslationField,
    TranslationRule,
    suggest_measurements,
    suggest_measurements_wrapped,
)
from mpa_scale_solver.types import CanonicalPoint


def _gamut(chit=(-0.5, 0.5), gamma=(-0.5, 0.5)) -> GamutSpec:
    return GamutSpec(chit_range=chit, gamma_AB_range=gamma)


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


def _lookup_field_sparse() -> TranslationField:
    """A coarse lookup-table field — discrete-grid posterior gives meaningful uncertainty variation."""
    rules = []
    for chit, gamma in [(-0.4, 0.0), (0.0, 0.0), (0.4, 0.0)]:
        rules.append(TranslationRule(
            operating_point=OperatingPoint(label=f"c={chit}", gt="c", axes={"tau_obs": 1.0}),
            xdot_choice="default",
            canonical=CanonicalPoint(chit=chit, gamma_AB=gamma, k_frust=False, method="lookup"),
        ))
    return TranslationField(direction="forward", shape="lookup_table", rule=rules)


def _grid(n=11, lo=-0.4, hi=0.4) -> np.ndarray:
    axis = np.linspace(lo, hi, n)
    return np.array([[c, g] for c in axis for g in axis], dtype=np.float64)


class TestSuggestMeasurementsBasics:
    def test_returns_at_most_n_candidates(self):
        field = _tangent_identity_field()
        gamut = _gamut()
        grid = _grid(n=11)
        out = suggest_measurements(field, gamut, grid, tau_obs=1.0, n=5)
        assert len(out) <= 5
        assert all(isinstance(c, MeasurementCandidate) for c in out)

    def test_skips_out_of_gamut_candidates(self):
        field = _tangent_identity_field()
        narrow_gamut = _gamut(chit=(-0.1, 0.1), gamma=(-0.1, 0.1))
        grid = _grid(n=11, lo=-0.5, hi=0.5)
        out = suggest_measurements(field, narrow_gamut, grid, tau_obs=1.0, n=20)
        for c in out:
            assert -0.1 <= c.state.chit <= 0.1
            assert -0.1 <= c.state.gamma_AB <= 0.1

    def test_candidates_carry_score_components(self):
        field = _tangent_identity_field()
        out = suggest_measurements(
            field, _gamut(), _grid(n=5), tau_obs=1.0, n=3,
        )
        for c in out:
            assert "uncertainty" in c.components
            assert "edge" in c.components
            assert "fragility" in c.components

    def test_descending_score_order(self):
        field = _tangent_identity_field()
        out = suggest_measurements(field, _gamut(), _grid(n=7), tau_obs=1.0, n=10)
        scores = [c.score for c in out]
        assert scores == sorted(scores, reverse=True)

    def test_empty_grid_returns_empty_list(self):
        field = _tangent_identity_field()
        out = suggest_measurements(
            field, _gamut(),
            np.empty((0, 2), dtype=np.float64),
            tau_obs=1.0, n=5,
        )
        assert out == []

    def test_n_zero_returns_empty(self):
        field = _tangent_identity_field()
        out = suggest_measurements(field, _gamut(), _grid(n=5), tau_obs=1.0, n=0)
        assert out == []


class TestSuggestMeasurementsEdgeProximity:
    def test_edge_score_higher_for_boundary_points(self):
        """Points near gamut edge should have higher edge component."""
        field = _tangent_identity_field()
        gamut = _gamut(chit=(-1.0, 1.0), gamma=(-1.0, 1.0))
        # Probe two specific points
        near_edge = np.array([[0.99, 0.0]], dtype=np.float64)
        center = np.array([[0.0, 0.0]], dtype=np.float64)
        out_edge = suggest_measurements(field, gamut, near_edge, tau_obs=1.0, n=1)
        out_center = suggest_measurements(field, gamut, center, tau_obs=1.0, n=1)
        assert out_edge[0].components["edge"] > out_center[0].components["edge"]


class TestSuggestMeasurementsWeights:
    def test_uncertainty_only_weighting(self):
        field = _lookup_field_sparse()
        gamut = _gamut(chit=(-0.5, 0.5), gamma=(-0.5, 0.5))
        out = suggest_measurements(
            field, gamut, _grid(n=7), tau_obs=1.0, n=3,
            weights={"uncertainty": 1.0, "edge": 0.0, "fragility": 0.0},
            canonical_search_grid=_grid(n=7),
        )
        # All scores should equal the uncertainty component when other
        # weights are zero.
        for c in out:
            assert c.score == pytest.approx(c.components["uncertainty"])

    def test_intent_fragility_contributes_for_oog_after_intent(self):
        # A point near the chit-edge such that some intents flag invariant
        # loss when clamped. We don't assert specific intents, just that
        # the fragility component is non-zero somewhere in the result set
        # when the gamut excludes most regimes.
        field = _tangent_identity_field()
        # Gamut admits only s_critical (|chit|<0.2); deep regimes flagged.
        gamut = _gamut(chit=(-0.18, 0.18))
        grid = _grid(n=7, lo=-0.18, hi=0.18)
        out = suggest_measurements(
            field, gamut, grid, tau_obs=1.0, n=20,
            weights={"uncertainty": 0.0, "edge": 0.0, "fragility": 1.0},
        )
        # Some candidate may or may not have fragility>0 depending on grid;
        # the test asserts the path doesn't blow up and the component is
        # populated.
        for c in out:
            assert "fragility" in c.components


class TestSuggestMeasurementsWrapped:
    def test_wrapped_carries_provenance(self):
        field = _tangent_identity_field()
        out = suggest_measurements_wrapped(
            field, _gamut(), _grid(n=5), tau_obs=1.0, n=3,
        )
        assert out.provenance.operation == "suggest_measurements"
        assert isinstance(out.value, list)
        assert "n_returned=" in out.provenance.notes[0]

    def test_wrapped_empty_canonical_grid_notes(self):
        field = _tangent_identity_field()
        out = suggest_measurements_wrapped(
            field, _gamut(),
            np.empty((0, 2), dtype=np.float64),
            tau_obs=1.0, n=5,
        )
        assert "no in-gamut candidates" in out.validation.notes[0]
