"""Unit tests, one section per operation (handoff §D.2)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mpa_scale_solver import (
    CanonicalPoint,
    CanonicalState,
    GamutSpec,
    OperatingPoint,
    RegimeReading,
    ScalingRule,
    SubstrateState,
    TangentFlowField,
    TranslationField,
    TranslationRule,
    apply_translation,
    forward_sweep_invert,
    gamut_classify,
    generate_locus,
    intent_map,
    interp_locus,
    locus_residual,
    regime_at,
    regime_display_band,
    tau_obs_sweep,
    validate_driver_profile,
    vertex_regime,
)
from mpa_scale_solver.operations import TranslationFieldIndex


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trivial_field() -> TranslationField:
    """Minimal lookup_table field: one rule, exact match for chit=0 gamma=0."""
    return TranslationField(
        direction="forward",
        shape="lookup_table",
        rule=[
            TranslationRule(
                operating_point=OperatingPoint(label="origin", gt="s", axes={}),
                xdot_choice="x",
                canonical=CanonicalPoint(chit=0.0, gamma_AB=0.0, k_frust=False, method="test"),
            ),
        ],
    )


def _three_cell_field() -> TranslationField:
    """Three-cell field at chit in {-1, 0, +1}, gamma in {0}, no tau_obs axis."""
    rules = []
    for chit in (-1.0, 0.0, 1.0):
        rules.append(TranslationRule(
            operating_point=OperatingPoint(
                label=f"chit={int(chit):+d}", gt="r" if chit < 0 else ("c" if chit > 0 else "s"),
                axes={"chit_label": chit},
            ),
            xdot_choice="x",
            canonical=CanonicalPoint(chit=chit, gamma_AB=0.0, k_frust=False, method="test"),
        ))
    return TranslationField(direction="forward", shape="lookup_table", rule=rules)


# ---------------------------------------------------------------------------
# vertex_regime / regime_at — 5-bucket per §C.4
# ---------------------------------------------------------------------------

class TestRegime:
    @pytest.mark.parametrize("chit, expected", [
        (1.0, "deep_c"),
        (0.7, "deep_c"),
        (0.5, "c_near_s"),
        (0.2, "c_near_s"),
        (0.0, "s_critical"),
        (-0.19, "s_critical"),
        (-0.2, "r_near_s"),
        (-0.5, "r_near_s"),
        (-0.7, "deep_r"),
        (-2.0, "deep_r"),
    ])
    def test_five_bucket_thresholds(self, chit, expected):
        assert vertex_regime(chit) == expected

    def test_regime_at_carries_k_frust(self):
        c = CanonicalState(chit=0.5, gamma_AB=0.0, k_frust=True)
        r = regime_at(c, tau_obs=1.0)
        assert isinstance(r, RegimeReading)
        assert r.regime == "c_near_s"
        assert r.k_frust is True

    @pytest.mark.parametrize("label, band", [
        ("deep_c", "c"),
        ("c_near_s", "c"),
        ("s_critical", "s"),
        ("r_near_s", "r"),
        ("deep_r", "r"),
    ])
    def test_display_band_collapse(self, label, band):
        assert regime_display_band(label) == band


# ---------------------------------------------------------------------------
# apply_translation
# ---------------------------------------------------------------------------

class TestApplyTranslation:
    def test_trivial_exact_match(self):
        field = _trivial_field()
        c = CanonicalState(chit=0.0, gamma_AB=0.0)
        s = apply_translation(c, field, tau_obs=1.0)
        assert s.label == "origin"
        assert s.tau_obs == 1.0

    def test_returns_substrate_state(self):
        s = apply_translation(
            CanonicalState(chit=0.7, gamma_AB=0.0),
            _three_cell_field(),
            tau_obs=1.0,
        )
        assert isinstance(s, SubstrateState)
        # Nearest rule should be chit=+1 (distance 0.3 vs chit=0 distance 0.7).
        assert s.label == "chit=+1"

    def test_nearest_neighbor_breaks_symmetrically(self):
        # midpoint exactly between two rules; numpy argmin returns first
        s = apply_translation(
            CanonicalState(chit=0.5, gamma_AB=0.0),
            _three_cell_field(),
            tau_obs=1.0,
        )
        assert s.label in ("chit=+0", "chit=+1")

    def test_empty_field_raises(self):
        with pytest.raises(ValueError, match="no rules"):
            apply_translation(
                CanonicalState(chit=0.0, gamma_AB=0.0),
                TranslationField(direction="forward", shape="lookup_table", rule=[]),
                tau_obs=1.0,
            )

    def test_domain_threshold_raises(self):
        with pytest.raises(ValueError, match="outside translation field domain"):
            apply_translation(
                CanonicalState(chit=100.0, gamma_AB=100.0),
                _trivial_field(),
                tau_obs=1.0,
                domain_distance_threshold=1.0,
            )

    def test_index_reuse(self):
        """Pre-built index gives identical result to per-call indexing."""
        field = _three_cell_field()
        index = TranslationFieldIndex(field)
        c = CanonicalState(chit=0.3, gamma_AB=0.0)
        s_via_field = apply_translation(c, field, tau_obs=1.0)
        s_via_index = apply_translation(c, index, tau_obs=1.0)
        assert s_via_field.label == s_via_index.label


# ---------------------------------------------------------------------------
# apply_translation on tangent-flow fields (v1 dispatch on shape)
# ---------------------------------------------------------------------------

def _tangent_flow_field(*, delta_chit=0.0, delta_gamma=0.0, refinement=None) -> TangentFlowField:
    origin = TranslationRule(
        operating_point=OperatingPoint(label="origin", gt="s", axes={"tau_obs": 1.0}),
        xdot_choice="identity",
        canonical=CanonicalPoint(chit=0.0, gamma_AB=0.0, k_frust=False, method="test"),
    )
    return TangentFlowField(
        direction="forward", shape="tangent_flow",
        rule_at_origin=origin,
        scaling=ScalingRule(
            tau_obs_ref=1.0,
            delta_chit=delta_chit,
            delta_gamma=delta_gamma,
            refinement=refinement,
        ),
    )


class TestApplyTranslationTangentFlow:
    def test_identity_scaling_returns_canonical_values(self):
        """delta = 0 in both axes = identity translation."""
        field = _tangent_flow_field()
        c = CanonicalState(chit=0.7, gamma_AB=-0.3)
        s = apply_translation(c, field, tau_obs=5.0)
        assert s.observables["substrate_chit"] == pytest.approx(0.7)
        assert s.observables["substrate_gamma_AB"] == pytest.approx(-0.3)
        assert s.label == "origin"

    def test_log_chit_drift_applied(self):
        """delta_chit log-drift adds delta_chit * log(tau_obs/tau_ref)."""
        field = _tangent_flow_field(delta_chit=0.5)
        c = CanonicalState(chit=1.0, gamma_AB=-0.2)
        s = apply_translation(c, field, tau_obs=math.e)  # log(e/1) = 1
        # substrate_chit = 1.0 + 0.5*1 = 1.5
        assert s.observables["substrate_chit"] == pytest.approx(1.5)

    def test_power_gamma_scaling_applied(self):
        """delta_gamma power-scales gamma by (tau/tau_ref)^delta_gamma."""
        field = _tangent_flow_field(delta_gamma=-1.0)
        c = CanonicalState(chit=0.5, gamma_AB=-0.4)
        s = apply_translation(c, field, tau_obs=2.0)
        # substrate_gamma_AB = -0.4 * (2/1)^-1 = -0.2
        assert s.observables["substrate_gamma_AB"] == pytest.approx(-0.2)

    def test_zero_tau_obs_returns_canonical_unmodified(self):
        """At tau_obs <= 0 we cannot evaluate log/ratio; treat as identity."""
        field = _tangent_flow_field(delta_chit=0.5)
        c = CanonicalState(chit=1.0, gamma_AB=-0.2)
        s = apply_translation(c, field, tau_obs=0.0)
        assert s.observables["substrate_chit"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# forward_sweep_invert
# ---------------------------------------------------------------------------

class TestForwardSweepInvert:
    def test_recovers_table_canonical(self):
        """Inversion against a 3-cell field recovers a candidate whose
        forward projection lands in the target rule's Voronoi cell."""
        field = _three_cell_field()
        target = SubstrateState(
            tau_obs=1.0,
            label="chit=+1",
            axes={"chit_label": 1.0},
            observables={},
        )
        # Candidate grid aligned to rule canonicals — recovery is exact.
        grid = np.array([[c, 0.0] for c in [-1.0, 0.0, 1.0]])
        recovered, residual = forward_sweep_invert(target, field, 1.0, grid)
        assert recovered.chit == pytest.approx(1.0)
        assert residual == pytest.approx(0.0)

    def test_returns_residual_field(self):
        field = _three_cell_field()
        target = SubstrateState(tau_obs=1.0, axes={"chit_label": 0.0}, observables={})
        grid = np.array([[c, 0.0] for c in np.linspace(-1.0, 1.0, 21)])
        out = forward_sweep_invert(target, field, 1.0, grid, return_residual_field=True)
        assert len(out) == 3
        assert out[2].shape == (21,)

    def test_callable_forward_map(self):
        """forward_map kwarg bypasses the field for analytical fixtures."""
        from mpa_scale_solver._test_fixtures import aging_log_forward, AgingLogParams
        params = AgingLogParams(chit_aging_coeff=1.0, tau_aging=1.0)
        # substrate observation at canonical (chit=0.5, gamma=0) and tau_obs=1.0
        truth = aging_log_forward(CanonicalState(chit=0.5, gamma_AB=0.0), 1.0, params)
        grid = np.array([[c, 0.0] for c in np.linspace(-1.0, 1.0, 51)])
        recovered, _ = forward_sweep_invert(
            truth, _trivial_field(), 1.0, grid,
            forward_map=lambda c, t: aging_log_forward(c, t, params),
        )
        assert recovered.chit == pytest.approx(0.5, abs=0.05)

    def test_bad_grid_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            forward_sweep_invert(
                SubstrateState(tau_obs=1.0),
                _trivial_field(),
                1.0,
                np.array([1.0, 2.0, 3.0]),  # not (N, 2)
            )


# ---------------------------------------------------------------------------
# tau_obs_sweep — per-frame fan-out (handoff §C.1)
# ---------------------------------------------------------------------------

class TestTauObsSweep:
    def test_single_substrate_replicates(self):
        field = _three_cell_field()
        target = SubstrateState(tau_obs=1.0, axes={"chit_label": 0.0})
        grid_tau = np.array([0.5, 1.0, 2.0])
        grid_canon = np.array([[c, 0.0] for c in [-1.0, 0.0, 1.0]])
        trajectory = tau_obs_sweep(target, field, grid_tau, grid_canon)
        assert len(trajectory) == 3
        for s in trajectory:
            assert s.chit == pytest.approx(0.0)

    def test_per_frame_targets(self):
        field = _three_cell_field()
        targets = [
            SubstrateState(tau_obs=0.5, axes={"chit_label": -1.0}),
            SubstrateState(tau_obs=1.0, axes={"chit_label": 0.0}),
            SubstrateState(tau_obs=2.0, axes={"chit_label": 1.0}),
        ]
        grid_tau = np.array([0.5, 1.0, 2.0])
        grid_canon = np.array([[c, 0.0] for c in [-1.0, 0.0, 1.0]])
        trajectory = tau_obs_sweep(targets, field, grid_tau, grid_canon)
        assert trajectory[0].chit == pytest.approx(-1.0)
        assert trajectory[1].chit == pytest.approx(0.0)
        assert trajectory[2].chit == pytest.approx(1.0)

    def test_target_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length"):
            tau_obs_sweep(
                [SubstrateState(tau_obs=1.0)],
                _three_cell_field(),
                np.array([0.5, 1.0, 2.0]),
                np.array([[0.0, 0.0]]),
            )


# ---------------------------------------------------------------------------
# gamut_classify
# ---------------------------------------------------------------------------

class TestGamutClassify:
    def test_in_gamut(self):
        gamut = GamutSpec(chit_range=(-1.0, 1.0), gamma_AB_range=(-1.0, 1.0))
        result = gamut_classify(CanonicalState(chit=0.5, gamma_AB=-0.5), 1.0, gamut)
        assert result["in_gamut"] is True
        assert result["diagnoses"] == []

    def test_out_of_gamut_chit(self):
        gamut = GamutSpec(chit_range=(-1.0, 1.0), gamma_AB_range=(-1.0, 1.0))
        result = gamut_classify(CanonicalState(chit=2.0, gamma_AB=0.0), 1.0, gamut)
        assert result["in_gamut"] is False
        assert len(result["diagnoses"]) == 1
        assert result["diagnoses"][0]["axis"] == "chit"
        assert result["diagnoses"][0]["distance"] == pytest.approx(1.0)

    def test_tau_obs_range_checked_when_present(self):
        gamut = GamutSpec(
            chit_range=(-1.0, 1.0), gamma_AB_range=(-1.0, 1.0),
            tau_obs_range=(0.1, 10.0),
        )
        ok = gamut_classify(CanonicalState(chit=0.0, gamma_AB=0.0), 1.0, gamut)
        assert ok["in_gamut"] is True
        bad = gamut_classify(CanonicalState(chit=0.0, gamma_AB=0.0), 100.0, gamut)
        assert bad["in_gamut"] is False
        assert any(d["axis"] == "tau_obs" for d in bad["diagnoses"])


# ---------------------------------------------------------------------------
# intent_map (I5 v1 contract — positive coverage; I1–I4 in test_intents.py)
# ---------------------------------------------------------------------------

class TestIntentMap:
    def test_i5_clamps_within_gamut(self):
        gamut = GamutSpec(chit_range=(-1.0, 1.0), gamma_AB_range=(-1.0, 1.0))
        mapped, sacrifice = intent_map(
            CanonicalState(chit=2.0, gamma_AB=0.0), 1.0, gamut, "I5",
        )
        assert mapped.chit == 1.0
        assert sacrifice["delta_chit"] == -1.0
        # both states are c-class
        assert sacrifice["regime_preserved"] is True

    def test_i5_flags_regime_break_on_cross(self):
        gamut = GamutSpec(chit_range=(-0.5, 0.5), gamma_AB_range=(-1.0, 1.0))
        _, sacrifice = intent_map(
            CanonicalState(chit=2.0, gamma_AB=0.0), 1.0, gamut, "I5",
        )
        # original is deep_c (chit=2), mapped is c_near_s (chit=0.5)
        assert sacrifice["regime_preserved"] is False
        assert sacrifice["original_regime"] == "deep_c"
        assert sacrifice["mapped_regime"] == "c_near_s"

    def test_unknown_intent_raises(self):
        gamut = GamutSpec(chit_range=(-1.0, 1.0), gamma_AB_range=(-1.0, 1.0))
        with pytest.raises(ValueError, match="unknown intent"):
            intent_map(CanonicalState(chit=0.0, gamma_AB=0.0), 1.0, gamut, "I99")


# ---------------------------------------------------------------------------
# validate_driver_profile
# ---------------------------------------------------------------------------

class TestValidateDriverProfile:
    def test_round_trip_on_three_cell_field(self):
        field = _three_cell_field()
        # Candidate grid aligned to rule canonicals so round-trip is exact.
        grid = np.array([[c, 0.0] for c in [-1.0, 0.0, 1.0]])
        ref = [
            {
                "canonical_state": CanonicalState(chit=chit, gamma_AB=0.0),
                "tau_obs": 1.0,
            }
            for chit in (-1.0, 0.0, 1.0)
        ]
        result = validate_driver_profile(field, ref, grid)
        assert result["round_trip_mean"] == pytest.approx(0.0)
        assert result["regime_agreement_rate"] == 1.0


# ---------------------------------------------------------------------------
# gfdr_model port sanity (handoff §B.5)
# ---------------------------------------------------------------------------

class TestGfdrModel:
    @pytest.mark.parametrize("regime, chit", [
        ("deep_c", 1.0), ("c_near_s", 0.3), ("s_critical", 0.0),
        ("r_near_s", -0.3), ("deep_r", -1.0),
    ])
    def test_generate_locus_branches(self, regime, chit):
        pts = generate_locus(chit, regime)
        assert len(pts) == 80
        # tau strictly increasing
        taus = [p["tau"] for p in pts]
        assert taus == sorted(taus)
        for p in pts:
            assert 0.0 <= p["C"] <= 1.0 + 1e-9
            assert math.isfinite(p["chi"])

    def test_locus_residual_minimum_at_true_chit(self):
        true_chit = 0.5
        empirical = generate_locus(true_chit, vertex_regime(true_chit))
        residuals = {c: locus_residual(empirical, c) for c in np.linspace(-1, 1, 41)}
        best = min(residuals, key=residuals.get)
        assert abs(best - true_chit) < 0.1

    def test_interp_locus_endpoints(self):
        pts = generate_locus(0.5, "c_near_s")
        # Far below tau_min: pin to first
        first = interp_locus(pts, 1e-9)
        assert first["C"] == pts[0]["C"]
        # Far above tau_max: pin to last
        last = interp_locus(pts, 1e9)
        assert last["C"] == pts[-1]["C"]
