"""v4 symbolic query DSL (BLOCK_IN §v4).

Coverage for the five DSL patterns:
  1. regime at chit=A gamma=B [tau=T]
  2. gamut at chit=A gamma=B [tau=T]                (gamut= kwarg)
  3. translate chit=A gamma=B at tau=T              (field= kwarg)
  4. invert chit_obs=X gamma_obs=Y at tau=T         (tangent_flow only)
  5. tau where regime crosses B for chit=A gamma=B  (closed form for
     tangent_flow; bisection for other shapes)
"""

from __future__ import annotations

import math

import pytest

from mpa_scale_solver import (
    CanonicalPoint,
    CanonicalState,
    GamutSpec,
    OperatingPoint,
    QueryParseError,
    QueryResult,
    RegimeReading,
    ScalingRule,
    SubstrateState,
    TangentFlowField,
    TranslationField,
    TranslationRule,
    query,
    supported_patterns,
)


def _tangent_flow(*, delta_chit=0.0, delta_gamma=0.0, tau_obs_ref=1.0) -> TangentFlowField:
    origin = TranslationRule(
        operating_point=OperatingPoint(label="origin", gt="s", axes={"tau_obs": 1.0}),
        xdot_choice="identity",
        canonical=CanonicalPoint(chit=0.0, gamma_AB=0.0, k_frust=False, method="test"),
    )
    return TangentFlowField(
        direction="forward", shape="tangent_flow",
        rule_at_origin=origin,
        scaling=ScalingRule(
            tau_obs_ref=tau_obs_ref,
            delta_chit=delta_chit,
            delta_gamma=delta_gamma,
        ),
    )


# ---------------------------------------------------------------------------
# regime at
# ---------------------------------------------------------------------------


class TestRegimeAt:
    def test_basic_match(self):
        r = query("regime at chit=0.5 gamma=0.0")
        assert isinstance(r, QueryResult)
        assert r.pattern == "regime at"
        assert isinstance(r.numerical, RegimeReading)
        assert r.numerical.regime == "c_near_s"
        assert "c_near_s" in r.closed_form

    def test_with_explicit_tau(self):
        r = query("regime at chit=-0.8 gamma=0.5 tau=2.0")
        assert r.numerical.regime == "deep_r"

    def test_case_insensitive(self):
        r = query("REGIME AT chit=0.0 gamma=0.0")
        assert r.numerical.regime == "s_critical"


# ---------------------------------------------------------------------------
# gamut at
# ---------------------------------------------------------------------------


class TestGamutAt:
    def test_in_gamut(self):
        g = GamutSpec(chit_range=(-1.0, 1.0), gamma_AB_range=(-1.0, 1.0))
        r = query("gamut at chit=0.5 gamma=-0.3", gamut=g)
        assert r.pattern == "gamut at"
        assert r.numerical["in_gamut"] is True
        assert "True" in r.closed_form

    def test_out_of_gamut(self):
        g = GamutSpec(chit_range=(-1.0, 1.0), gamma_AB_range=(-1.0, 1.0))
        r = query("gamut at chit=2.0 gamma=0.0", gamut=g)
        assert r.numerical["in_gamut"] is False

    def test_missing_gamut_raises(self):
        with pytest.raises(QueryParseError, match="gamut"):
            query("gamut at chit=0.0 gamma=0.0")


# ---------------------------------------------------------------------------
# translate
# ---------------------------------------------------------------------------


class TestTranslate:
    def test_tangent_flow_identity(self):
        f = _tangent_flow()
        r = query("translate chit=0.7 gamma=-0.3 at tau=5.0", field=f)
        assert r.pattern == "translate"
        s: SubstrateState = r.numerical
        assert s.observables["substrate_chit"] == pytest.approx(0.7)
        assert s.observables["substrate_gamma_AB"] == pytest.approx(-0.3)
        # closed-form populated for tangent_flow
        assert r.closed_form is not None
        assert "ln" in r.closed_form

    def test_tangent_flow_with_drift(self):
        f = _tangent_flow(delta_chit=0.5)
        r = query("translate chit=1.0 gamma=-0.2 at tau=2.718281828", field=f)
        # substrate_chit = 1.0 + 0.5*ln(e) = 1.5
        assert r.numerical.observables["substrate_chit"] == pytest.approx(1.5, abs=1e-6)

    def test_lookup_table_no_closed_form(self):
        rules = [TranslationRule(
            operating_point=OperatingPoint(label="origin", gt="s", axes={}),
            xdot_choice="x",
            canonical=CanonicalPoint(chit=0.0, gamma_AB=0.0, k_frust=False, method="t"),
        )]
        f = TranslationField(direction="forward", shape="lookup_table", rule=rules)
        r = query("translate chit=0.0 gamma=0.0 at tau=1.0", field=f)
        # numerical result present, closed_form is None for lookup-table.
        assert r.numerical is not None
        assert r.closed_form is None

    def test_missing_field_raises(self):
        with pytest.raises(QueryParseError, match="field"):
            query("translate chit=0.0 gamma=0.0 at tau=1.0")


# ---------------------------------------------------------------------------
# invert
# ---------------------------------------------------------------------------


class TestInvert:
    def test_tangent_flow_round_trip(self):
        f = _tangent_flow(delta_chit=0.3, delta_gamma=-0.5)
        # Forward project a known canonical at tau=2.0
        chit, gamma = 0.4, -0.2
        tau = 2.0
        s_chit = chit + 0.3 * math.log(tau / 1.0)
        s_gamma = gamma * (tau / 1.0) ** -0.5
        # Now invert via DSL
        r = query(
            f"invert chit_obs={s_chit} gamma_obs={s_gamma} at tau={tau}",
            field=f,
        )
        assert r.pattern == "invert"
        rec: CanonicalState = r.numerical
        assert rec.chit == pytest.approx(chit, abs=1e-9)
        assert rec.gamma_AB == pytest.approx(gamma, abs=1e-9)
        assert r.closed_form is not None

    def test_lookup_table_rejected(self):
        rules = [TranslationRule(
            operating_point=OperatingPoint(label="o", gt="s", axes={}),
            xdot_choice="x",
            canonical=CanonicalPoint(chit=0.0, gamma_AB=0.0, k_frust=False, method="t"),
        )]
        f = TranslationField(direction="forward", shape="lookup_table", rule=rules)
        with pytest.raises(QueryParseError, match="tangent_flow"):
            query("invert chit_obs=0.0 gamma_obs=0.0 at tau=1.0", field=f)


# ---------------------------------------------------------------------------
# tau where regime crosses
# ---------------------------------------------------------------------------


class TestTauCrossing:
    def test_tangent_flow_closed_form(self):
        # chit(tau) = 0.5 + 0.3 * ln(tau / 1.0) = 0.7
        # -> tau = exp((0.7 - 0.5) / 0.3) ≈ 1.9477
        f = _tangent_flow(delta_chit=0.3)
        r = query("tau where regime crosses 0.7 for chit=0.5 gamma=0.0", field=f)
        assert r.pattern == "tau where regime crosses"
        assert r.numerical == pytest.approx(math.exp(0.2 / 0.3), rel=1e-10)
        assert "tau_ref" in r.closed_form

    def test_invalid_boundary_rejected(self):
        f = _tangent_flow(delta_chit=0.3)
        with pytest.raises(QueryParseError, match="boundary"):
            query("tau where regime crosses 0.5 for chit=0.0 gamma=0.0", field=f)

    def test_delta_chit_zero_returns_note(self):
        f = _tangent_flow(delta_chit=0.0)
        r = query("tau where regime crosses 0.7 for chit=0.5 gamma=0.0", field=f)
        assert r.numerical is None
        assert any("constant" in n.lower() for n in r.notes)

    def test_missing_field_raises(self):
        with pytest.raises(QueryParseError, match="field"):
            query("tau where regime crosses 0.7 for chit=0.0 gamma=0.0")


# ---------------------------------------------------------------------------
# Parser surface
# ---------------------------------------------------------------------------


class TestParser:
    def test_unrecognized_query_raises_with_help(self):
        with pytest.raises(QueryParseError, match="Supported patterns"):
            query("not a real query")

    def test_supported_patterns_returns_tuple(self):
        patterns = supported_patterns()
        assert isinstance(patterns, tuple)
        assert len(patterns) == 5

    def test_scientific_notation_accepted(self):
        r = query("regime at chit=5e-1 gamma=-1.5e-1")
        assert r.numerical.regime == "c_near_s"
