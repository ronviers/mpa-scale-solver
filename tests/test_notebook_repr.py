"""v4 notebook ergonomics (BLOCK_IN §v4).

Coverage:
  - `_repr_html_` returns an HTML string on user-facing dataclasses
  - Custom `__repr__` on Posterior / LearnedField / OperationOutput is
    compact and informative (no raw nested tuples or weight dumps)
  - Plot helpers in `mpa_scale_solver.plotting` return figure objects
    when matplotlib is available; raise informatively when not

Plot tests use matplotlib if installed (it's in `test` extras already);
skip cleanly otherwise.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mpa_scale_solver import (
    CanonicalPoint,
    CanonicalState,
    DispatchPath,
    GamutSpec,
    InversionResult,
    MeasurementCandidate,
    OperatingPoint,
    OperationOutput,
    Posterior,
    Provenance,
    RegimeReading,
    ScalingRule,
    SubstrateState,
    TangentFlowField,
    TranslationField,
    TranslationRule,
    ValidationReport,
    __version__,
)


def _provenance() -> Provenance:
    return Provenance(
        solver_version=__version__,
        operation="test",
        timestamp_ns=0,
        dispatch_path=DispatchPath.DIRECT_COMPUTE,
    )


# ---------------------------------------------------------------------------
# _repr_html_ presence and shape
# ---------------------------------------------------------------------------


class TestReprHtml:
    def test_canonical_state_html_includes_regime(self):
        html = CanonicalState(chit=0.5, gamma_AB=0.0)._repr_html_()
        assert "<table" in html
        assert "CanonicalState" in html
        assert "c_near_s" in html

    def test_substrate_state_html(self):
        html = SubstrateState(
            tau_obs=1.0, label="cell-a", observables={"x": 0.5},
        )._repr_html_()
        assert "SubstrateState" in html
        assert "cell-a" in html

    def test_gamut_spec_html(self):
        html = GamutSpec(
            chit_range=(-1.0, 1.0), gamma_AB_range=(-0.5, 0.5),
        )._repr_html_()
        assert "GamutSpec" in html
        assert "-1" in html

    def test_regime_reading_html(self):
        html = RegimeReading(regime="deep_c", k_frust=True)._repr_html_()
        assert "deep_c" in html
        assert "True" in html

    def test_translation_field_html_counts_rules(self):
        rules = [TranslationRule(
            operating_point=OperatingPoint(label=f"r{i}", gt="s", axes={}),
            xdot_choice="x",
            canonical=CanonicalPoint(chit=0.0, gamma_AB=0.0, k_frust=False, method="t"),
        ) for i in range(3)]
        field = TranslationField(direction="forward", shape="lookup_table", rule=rules)
        html = field._repr_html_()
        assert "lookup_table" in html
        assert "3" in html

    def test_tangent_flow_field_html(self):
        origin = TranslationRule(
            operating_point=OperatingPoint(label="origin", gt="s", axes={}),
            xdot_choice="x",
            canonical=CanonicalPoint(chit=0.0, gamma_AB=0.0, k_frust=False, method="t"),
        )
        field = TangentFlowField(
            direction="forward", shape="tangent_flow",
            rule_at_origin=origin,
            scaling=ScalingRule(tau_obs_ref=1.0, delta_chit=0.3),
        )
        html = field._repr_html_()
        assert "TangentFlowField" in html
        assert "0.3" in html

    def test_posterior_html(self):
        p = Posterior(
            mean=CanonicalState(chit=0.5, gamma_AB=-0.1),
            covariance=((0.01, 0.0), (0.0, 0.02)),
            log_evidence=-12.5,
        )
        html = p._repr_html_()
        assert "Posterior" in html
        assert "0.03" in html  # cov trace = 0.01 + 0.02

    def test_operation_output_html(self):
        out = OperationOutput(
            value=CanonicalState(chit=0.0, gamma_AB=0.0),
            validation=ValidationReport(),
            provenance=_provenance(),
        )
        html = out._repr_html_()
        assert "OperationOutput" in html
        assert "direct_compute" in html

    def test_measurement_candidate_html(self):
        c = MeasurementCandidate(
            state=CanonicalState(chit=0.4, gamma_AB=0.0),
            tau_obs=1.0, score=0.5,
            components={"uncertainty": 0.3, "edge": 0.2, "fragility": 0.0},
        )
        html = c._repr_html_()
        assert "MeasurementCandidate" in html
        assert "uncertainty" in html

    def test_inversion_result_html(self):
        ir = InversionResult(
            state=CanonicalState(chit=0.0, gamma_AB=0.0),
            residual=0.01, tau_obs=1.0, frame_index=3,
        )
        html = ir._repr_html_()
        assert "InversionResult" in html
        assert "3" in html


# ---------------------------------------------------------------------------
# Custom __repr__ on the previously-ugly dataclasses
# ---------------------------------------------------------------------------


class TestReprOverrides:
    def test_posterior_repr_is_compact(self):
        p = Posterior(
            mean=CanonicalState(chit=0.5, gamma_AB=-0.1),
            covariance=((0.01, 0.0), (0.0, 0.02)),
            log_evidence=-12.5,
        )
        r = repr(p)
        # Compact: no nested tuple dump, summary statistics instead.
        assert r.startswith("Posterior(")
        assert "cov_trace=" in r
        assert "((0.01" not in r  # raw nested tuple is suppressed

    def test_learned_field_repr_does_not_dump_weights(self):
        # Build a small LearnedField with non-trivial weights.
        from mpa_scale_solver import LearnedField
        rule = TranslationRule(
            operating_point=OperatingPoint(label="o", gt="s", axes={}),
            xdot_choice="x",
            canonical=CanonicalPoint(chit=0.0, gamma_AB=0.0, k_frust=False, method="t"),
        )
        W = tuple(tuple(0.1 for _ in range(3)) for _ in range(4))
        b = tuple(0.0 for _ in range(4))
        lf = LearnedField(
            direction="forward", shape="learned",
            rule_at_origin=rule,
            weights=((W, b),),
            architecture=(3, 4),
        )
        r = repr(lf)
        assert r.startswith("LearnedField(")
        assert "architecture=[3, 4]" in r
        assert "0.1" not in r  # weights not dumped

    def test_operation_output_repr_is_one_line(self):
        out = OperationOutput(
            value=CanonicalState(chit=0.0, gamma_AB=0.0),
            validation=ValidationReport(),
            provenance=_provenance(),
        )
        r = repr(out)
        assert r.startswith("OperationOutput(")
        assert "value=CanonicalState" in r
        assert "validation.ok=True" in r
        assert "\n" not in r  # one line


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


_HAS_MPL = True
try:
    import matplotlib  # noqa: F401
    import matplotlib.pyplot as plt
    matplotlib.use("Agg")  # headless backend for CI
except ImportError:
    _HAS_MPL = False


@pytest.mark.skipif(not _HAS_MPL, reason="matplotlib not installed")
class TestPlottingMatplotlib:
    def test_plot_trajectory_returns_figure(self):
        from mpa_scale_solver.plotting import plot_trajectory
        traj = [
            CanonicalState(chit=c, gamma_AB=0.0)
            for c in np.linspace(-1.0, 1.0, 9)
        ]
        fig = plot_trajectory(traj)
        assert fig is not None
        plt.close(fig)

    def test_plot_gamut_with_points(self):
        from mpa_scale_solver.plotting import plot_gamut
        g = GamutSpec(chit_range=(-1.0, 1.0), gamma_AB_range=(-1.0, 1.0))
        pts = [
            CanonicalState(chit=0.0, gamma_AB=0.0),
            CanonicalState(chit=2.0, gamma_AB=0.0),  # out
        ]
        fig = plot_gamut(g, points=pts, title="test")
        assert fig is not None
        plt.close(fig)

    def test_plot_residual_field(self):
        from mpa_scale_solver.plotting import plot_residual_field
        grid = np.array([[c, 0.0] for c in np.linspace(-1.0, 1.0, 11)])
        residuals = np.linspace(1.0, 0.0, 11) ** 2
        recovered = CanonicalState(chit=1.0, gamma_AB=0.0)
        fig = plot_residual_field(residuals, grid, recovered=recovered)
        assert fig is not None
        plt.close(fig)

    def test_plot_posterior(self):
        from mpa_scale_solver.plotting import plot_posterior
        p = Posterior(
            mean=CanonicalState(chit=0.3, gamma_AB=-0.1),
            covariance=((0.05, 0.01), (0.01, 0.03)),
        )
        fig = plot_posterior(p)
        assert fig is not None
        plt.close(fig)

    def test_unknown_backend_raises(self):
        from mpa_scale_solver.plotting import plot_trajectory
        with pytest.raises(ValueError, match="unsupported backend"):
            plot_trajectory([CanonicalState(chit=0.0, gamma_AB=0.0)], backend="ascii")
