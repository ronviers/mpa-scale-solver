"""v5 — continuous Banach self-test cadence (BLOCK_IN §v5).

Coverage:
  - BanachDriftReport shape + drift_within_tolerance property
  - run_banach_self_test agreement vs analytical (drift <= DRIFT_TOLERANCE)
  - SelfTestCadence tick / counter / callback semantics
  - streaming.forward_sweep_invert_stream cadence hook fires per emitted frame
  - state-locality: the self-test does not perturb the primary inversion
"""

from __future__ import annotations

import numpy as np
import pytest

from mpa_scale_solver import (
    BanachDriftReport,
    BanachSubstrate,
    CanonicalPoint,
    DRIFT_TOLERANCE,
    InversionResult,
    OperatingPoint,
    SelfTestCadence,
    SubstrateState,
    TranslationField,
    TranslationRule,
    __version__,
    forward_sweep_invert_stream,
    run_banach_self_test,
)
from mpa_scale_solver.self_test import _DEFAULT_NU_SAMPLES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _three_cell_field() -> TranslationField:
    """Mirror of the streaming test helper (lookup_table, no tau axis)."""
    rules = []
    for chit in (-1.0, 0.0, 1.0):
        rules.append(TranslationRule(
            operating_point=OperatingPoint(
                label=f"chit={int(chit):+d}",
                gt="r" if chit < 0 else ("c" if chit > 0 else "s"),
                axes={"chit_label": chit},
            ),
            xdot_choice="x",
            canonical=CanonicalPoint(
                chit=chit, gamma_AB=0.0, k_frust=False, method="test",
            ),
        ))
    return TranslationField(direction="forward", shape="lookup_table", rule=rules)


def _grid() -> np.ndarray:
    return np.array([[c, 0.0] for c in [-1.0, 0.0, 1.0]])


# ---------------------------------------------------------------------------
# run_banach_self_test
# ---------------------------------------------------------------------------

class TestRunBanachSelfTest:
    def test_default_run_has_no_drift(self):
        """JAX surface matches the analytical closed form within tolerance."""
        report = run_banach_self_test()
        assert isinstance(report, BanachDriftReport)
        assert report.drift_within_tolerance
        assert report.max_chit_drift <= DRIFT_TOLERANCE
        assert report.max_gamma_drift <= DRIFT_TOLERANCE
        assert report.asymptotic_closure_compliant
        assert report.solver_version == __version__

    def test_sample_count_matches_default_nu_samples(self):
        report = run_banach_self_test()
        assert report.sample_count == len(_DEFAULT_NU_SAMPLES)

    def test_custom_nu_samples_honored(self):
        custom = (0.1, 0.3, 0.9)
        report = run_banach_self_test(nu_samples=custom)
        assert report.sample_count == 3

    def test_call_index_recorded(self):
        report = run_banach_self_test(call_index=42)
        assert report.call_index == 42

    def test_custom_substrate(self):
        """Drift check holds for any Banach substrate parameterization."""
        substrate = BanachSubstrate(chit_0=2.5, gamma_AB_0=-0.8, lambda_chit=0.5)
        report = run_banach_self_test(substrate=substrate)
        assert report.drift_within_tolerance

    def test_timestamp_is_monotonic(self):
        """timestamp_ns advances across calls (sanity — not a race check)."""
        r1 = run_banach_self_test()
        r2 = run_banach_self_test()
        assert r2.timestamp_ns >= r1.timestamp_ns


# ---------------------------------------------------------------------------
# SelfTestCadence
# ---------------------------------------------------------------------------

class TestSelfTestCadence:
    def test_default_k_is_100(self):
        cadence = SelfTestCadence()
        assert cadence.k == 100

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError, match="positive"):
            SelfTestCadence(k=0)
        with pytest.raises(ValueError, match="positive"):
            SelfTestCadence(k=-5)

    def test_tick_returns_none_until_k(self):
        cadence = SelfTestCadence(k=3)
        assert cadence.tick() is None
        assert cadence.tick() is None
        report = cadence.tick()
        assert report is not None
        assert report.call_index == 3

    def test_tick_periodicity(self):
        cadence = SelfTestCadence(k=2)
        reports = [cadence.tick() for _ in range(6)]
        # tick 1: None; tick 2: report (call_index=2); 3: None; 4: report (4); ...
        fired = [r for r in reports if r is not None]
        assert len(fired) == 3
        assert [r.call_index for r in fired] == [2, 4, 6]

    def test_callback_invoked_on_test_tick(self):
        cadence = SelfTestCadence(k=2)
        seen = []
        cadence.tick(callback=seen.append)
        assert seen == []  # first tick — no test
        cadence.tick(callback=seen.append)
        assert len(seen) == 1
        assert isinstance(seen[0], BanachDriftReport)

    def test_callback_not_invoked_on_skip_tick(self):
        cadence = SelfTestCadence(k=5)
        seen = []
        for _ in range(4):
            cadence.tick(callback=seen.append)
        assert seen == []

    def test_last_report_persists(self):
        cadence = SelfTestCadence(k=2)
        assert cadence.last_report is None
        cadence.tick()
        assert cadence.last_report is None  # not yet test tick
        cadence.tick()
        assert cadence.last_report is not None
        # After another non-test tick, last_report is still the prior one.
        cadence.tick()
        assert cadence.last_report.call_index == 2

    def test_reset_clears_state(self):
        cadence = SelfTestCadence(k=2)
        cadence.tick()
        cadence.tick()  # fires
        cadence.reset()
        assert cadence.call_count == 0
        assert cadence.last_report is None

    def test_callback_exception_propagates(self):
        """A raising callback is the consumer's bug; we don't swallow."""
        cadence = SelfTestCadence(k=1)
        def bad(_): raise RuntimeError("user code")
        with pytest.raises(RuntimeError, match="user code"):
            cadence.tick(callback=bad)


# ---------------------------------------------------------------------------
# Streaming hookup — cadence fires per emitted frame
# ---------------------------------------------------------------------------

class TestStreamingSelfTestHook:
    def test_cadence_advances_per_frame(self):
        field = _three_cell_field()
        obs = [
            SubstrateState(tau_obs=1.0, axes={"chit_label": -1.0}),
            SubstrateState(tau_obs=1.0, axes={"chit_label": 0.0}),
            SubstrateState(tau_obs=1.0, axes={"chit_label": 1.0}),
        ]
        cadence = SelfTestCadence(k=2)
        captured = []
        results = list(forward_sweep_invert_stream(
            obs, field, _grid(), tau_obs=1.0,
            self_test_cadence=cadence,
            self_test_callback=captured.append,
        ))
        assert len(results) == 3
        assert cadence.call_count == 3
        # k=2 with 3 frames: fires once (on frame 2).
        assert len(captured) == 1

    def test_state_local_self_test_does_not_perturb_recovery(self):
        """Cadence + callback recovery must match no-cadence recovery exactly."""
        field = _three_cell_field()
        obs = [
            SubstrateState(tau_obs=1.0, axes={"chit_label": c})
            for c in [-1.0, 0.0, 1.0]
        ]
        baseline = [
            r.state.chit
            for r in forward_sweep_invert_stream(obs, field, _grid(), tau_obs=1.0)
        ]
        cadence = SelfTestCadence(k=1)  # fires every frame — maximum perturbation chance
        with_cadence = [
            r.state.chit
            for r in forward_sweep_invert_stream(
                obs, field, _grid(), tau_obs=1.0,
                self_test_cadence=cadence,
            )
        ]
        assert baseline == with_cadence

    def test_no_cadence_emits_no_reports(self):
        field = _three_cell_field()
        obs = [SubstrateState(tau_obs=1.0, axes={"chit_label": 0.0})] * 5
        captured = []
        results = list(forward_sweep_invert_stream(
            obs, field, _grid(), tau_obs=1.0,
            self_test_callback=captured.append,  # no cadence -> never called
        ))
        assert len(results) == 5
        assert captured == []

    def test_cadence_without_callback_still_runs_test(self):
        """The callback is optional; reports still accumulate on .last_report."""
        field = _three_cell_field()
        obs = [SubstrateState(tau_obs=1.0, axes={"chit_label": 0.0})] * 4
        cadence = SelfTestCadence(k=2)
        list(forward_sweep_invert_stream(
            obs, field, _grid(), tau_obs=1.0,
            self_test_cadence=cadence,
        ))
        assert cadence.last_report is not None
        # k=2 with 4 frames: fires twice (frames 2 and 4), last_report = frame 4.
        assert cadence.last_report.call_index == 4
        assert cadence.last_report.drift_within_tolerance


# ---------------------------------------------------------------------------
# Drift detection — synthetic drift via bad substrate parameters
# ---------------------------------------------------------------------------

class TestDriftDetection:
    def test_zero_substrate_yields_zero_drift(self):
        """A substrate whose chit_0 = 0 has zero analytical state and zero
        JAX state; drift is exactly zero (asymptotic case)."""
        substrate = BanachSubstrate(chit_0=0.0, gamma_AB_0=0.0)
        report = run_banach_self_test(substrate=substrate)
        assert report.max_chit_drift == 0.0
        assert report.max_gamma_drift == 0.0
