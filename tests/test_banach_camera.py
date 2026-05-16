"""Banach substrate camera test (handoff §C.3 / §E acceptance criterion 2).

Replaces the v0 hand-crafted aging_log synthetic with the framework's own
self-reference. The Banach substrate's `state_at(nu)` is the analytical
truth (closed-form exponential decay per Q1 of the v1 build session);
the solver's `forward_sweep_invert` recovers canonical state per frame
and we score against the analytical truth.

Pass criterion: max |residual| <= 0.001 per axis (chit, gamma_AB) across
all frames. The tolerance tightens from v0's 0.05 (against synthetic)
because the Banach truth is exact closed-form; only the brute-force grid
resolution limits accuracy.

The legacy `test_camera_migration.py` is kept passing as back-compat
coverage of the lookup-table dispatch path.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from mpa_scale_solver import (
    BanachSubstrate,
    CanonicalState,
    DispatchPath,
    apply_translation,
    apply_translation_wrapped,
    flow,
    forward_sweep_invert,
    forward_sweep_invert_wrapped,
    regime_at,
    tau_obs_sweep,
    tau_obs_sweep_wrapped,
)


# Pass tolerance per handoff §E item 2 (tightened from 0.05 to 0.001).
TOLERANCE = 0.001


# Banach reference instance: c-band start, cooperative gamma. Traces the
# full c -> s migration interior; asymptotes toward (0, 0) without
# reaching it at any finite nu (Asymptotic-Closure-compliant).
CHIT_0 = 1.5
GAMMA_AB_0 = -0.5

# tau_obs sweep: 80 log-spaced frames over [0.01, 10]. Covers the entire
# migration interior; deep enough to enter s_critical without driving
# `exp(-nu)` to numerical zero (which would flag Asymptotic-Closure on
# the otherwise-exempt Banach substrate).
TAU_OBS_GRID = np.logspace(-2, 1, 80)


def _adaptive_grid(target_substrate, *, window: float = 0.005, n: int = 11):
    """Tight 2D grid around the substrate observation.

    For Banach (identity translation) the substrate's `substrate_chit` /
    `substrate_gamma_AB` are the canonical values themselves, so a small
    window around them brackets the true canonical at sub-tolerance
    resolution. Step = 2*window/(n-1); with window=0.005 and n=11 the
    step is 0.001 — exactly the tolerance floor.

    Real-substrate consumers in mpa-conform supply a grid sized from
    their driver-profile prior; here we use the substrate observation
    directly because it IS the prior (identity translation).
    """
    chit_c = float(target_substrate.observables["substrate_chit"])
    gamma_c = float(target_substrate.observables["substrate_gamma_AB"])
    chit_grid = np.linspace(chit_c - window, chit_c + window, n)
    gamma_grid = np.linspace(gamma_c - window, gamma_c + window, n)
    cg, gg = np.meshgrid(chit_grid, gamma_grid, indexing="ij")
    return np.column_stack([cg.ravel(), gg.ravel()])


# ---------------------------------------------------------------------------
# flow() exact identity
# ---------------------------------------------------------------------------

class TestFlowExact:
    """`flow()` evaluates the same closed form as `state_at()`."""

    @pytest.mark.parametrize("nu", [0.0, 0.5, 1.0, 2.0, 3.0, 5.0])
    def test_flow_matches_state_at(self, nu):
        substrate = BanachSubstrate(chit_0=CHIT_0, gamma_AB_0=GAMMA_AB_0)
        field = substrate.translation_field()
        flowed = flow(substrate.canonical_initial(), nu, field)
        truth = substrate.state_at(nu)
        # Both compute chit_0 * exp(-lambda * nu) — agree to machine
        # precision (modulo float operation order).
        assert flowed.chit == pytest.approx(truth.chit, abs=1e-15)
        assert flowed.gamma_AB == pytest.approx(truth.gamma_AB, abs=1e-15)

    def test_flow_lookup_table_raises(self):
        """v1 only implements flow for tangent-flow fields."""
        from mpa_scale_solver import TranslationField

        substrate = BanachSubstrate()
        with pytest.raises(NotImplementedError, match="lookup_table"):
            flow(
                substrate.canonical_initial(),
                1.0,
                TranslationField(direction="forward", shape="lookup_table", rule=[]),
            )


# ---------------------------------------------------------------------------
# apply_translation identity
# ---------------------------------------------------------------------------

class TestApplyTranslationIdentity:
    """Banach's tangent-flow field with delta=0 is identity translation."""

    @pytest.mark.parametrize("nu", [0.01, 0.5, 1.0, 2.0, 10.0])
    def test_substrate_observables_equal_canonical(self, nu):
        substrate = BanachSubstrate(chit_0=CHIT_0, gamma_AB_0=GAMMA_AB_0)
        field = substrate.translation_field()
        canonical = CanonicalState(chit=0.7, gamma_AB=-0.3)
        s = apply_translation(canonical, field, tau_obs=nu)
        assert s.observables["substrate_chit"] == pytest.approx(0.7)
        assert s.observables["substrate_gamma_AB"] == pytest.approx(-0.3)
        assert s.label == "banach_origin"


# ---------------------------------------------------------------------------
# Camera test: brute-force inversion residual <= 0.001
# ---------------------------------------------------------------------------

def _run_camera_inversion():
    substrate = BanachSubstrate(chit_0=CHIT_0, gamma_AB_0=GAMMA_AB_0)
    field = substrate.translation_field()
    targets = [substrate.substrate_at(float(nu)) for nu in TAU_OBS_GRID]

    chit_residuals: list[float] = []
    gamma_residuals: list[float] = []
    for nu, target in zip(TAU_OBS_GRID, targets):
        grid = _adaptive_grid(target)
        recovered, _ = forward_sweep_invert(target, field, float(nu), grid)
        truth = substrate.state_at(float(nu))
        chit_residuals.append(abs(recovered.chit - truth.chit))
        gamma_residuals.append(abs(recovered.gamma_AB - truth.gamma_AB))
    return chit_residuals, gamma_residuals


def test_banach_camera_residual_within_tolerance():
    chit_resids, gamma_resids = _run_camera_inversion()
    max_chit = max(chit_resids)
    max_gamma = max(gamma_resids)
    assert max_chit < TOLERANCE, (
        f"max |chit residual| = {max_chit:.6f} exceeds tolerance {TOLERANCE}"
    )
    assert max_gamma < TOLERANCE, (
        f"max |gamma_AB residual| = {max_gamma:.6f} exceeds tolerance {TOLERANCE}"
    )


def test_banach_camera_via_tau_obs_sweep_with_sidecar():
    """Same recovery via the sweep + sidecar (table-first dispatch).

    The shared-grid path is too slow for a tight residual at this size;
    the production fast path is the sidecar (curator-precomputed inverse
    table). With the Banach sidecar built on the same tau_obs grid the
    sweep evaluates, every frame is a TABLE_HIT and recovery is exact.
    """
    substrate = BanachSubstrate(chit_0=CHIT_0, gamma_AB_0=GAMMA_AB_0)
    field = substrate.translation_field()
    targets = [substrate.substrate_at(float(nu)) for nu in TAU_OBS_GRID]
    sidecar = substrate.build_sidecar(TAU_OBS_GRID)

    # Dummy small search grid: the sidecar takes every frame.
    search_grid = np.array([[0.0, 0.0]])

    out = tau_obs_sweep_wrapped(
        targets, field, TAU_OBS_GRID, search_grid, sidecar=sidecar,
    )
    assert out.provenance.dispatch_path == DispatchPath.TABLE_HIT
    truths = [substrate.state_at(float(nu)) for nu in TAU_OBS_GRID]
    for recovered, truth in zip(out.value, truths):
        # Table-hit returns the exact recorded canonical (modulo float
        # round-trip through the table's float64 keys).
        assert recovered.chit == pytest.approx(truth.chit, abs=1e-12)
        assert recovered.gamma_AB == pytest.approx(truth.gamma_AB, abs=1e-12)


# ---------------------------------------------------------------------------
# Migration regime trace
# ---------------------------------------------------------------------------

class TestBanachMigration:
    """The c-band start traces deep_c -> c_near_s -> s_critical as nu sweeps."""

    def test_initial_deep_c(self):
        substrate = BanachSubstrate(chit_0=CHIT_0, gamma_AB_0=GAMMA_AB_0)
        assert regime_at(substrate.state_at(0.0), 0.0).regime == "deep_c"

    def test_mid_c_near_s(self):
        # chit(1.0) = 1.5 * exp(-1) ~= 0.552 -> c_near_s (0.2 <= chit < 0.7)
        substrate = BanachSubstrate(chit_0=CHIT_0, gamma_AB_0=GAMMA_AB_0)
        assert regime_at(substrate.state_at(1.0), 1.0).regime == "c_near_s"

    def test_late_s_critical(self):
        # chit(3.0) = 1.5 * exp(-3) ~= 0.0747 -> s_critical
        substrate = BanachSubstrate(chit_0=CHIT_0, gamma_AB_0=GAMMA_AB_0)
        assert regime_at(substrate.state_at(3.0), 3.0).regime == "s_critical"

    def test_asymptotic_closure_not_violated(self):
        """At any finite nu, chit > 0 (never exactly the asymptotic limit)."""
        substrate = BanachSubstrate(chit_0=CHIT_0, gamma_AB_0=GAMMA_AB_0)
        for nu in (0.01, 1.0, 10.0, 50.0, 100.0):
            s = substrate.state_at(nu)
            assert s.chit > 0.0
            assert s.gamma_AB < 0.0


# ---------------------------------------------------------------------------
# Wrapped variants on Banach
# ---------------------------------------------------------------------------

class TestBanachWrappedOps:
    def test_apply_translation_wrapped_carries_provenance(self):
        substrate = BanachSubstrate()
        field = substrate.translation_field()
        out = apply_translation_wrapped(
            substrate.canonical_initial(), field, tau_obs=1.0,
        )
        assert out.provenance.operation == "apply_translation"
        assert out.provenance.dispatch_path == DispatchPath.DIRECT_COMPUTE
        from mpa_scale_solver import __version__
        assert out.provenance.solver_version == __version__
        assert out.validation.asymptotic_closure_compliant is True

    def test_tau_obs_sweep_wrapped_validates_k_frust(self):
        substrate = BanachSubstrate()
        field = substrate.translation_field()
        targets = [substrate.substrate_at(float(nu)) for nu in TAU_OBS_GRID]

        chit_axis = np.linspace(-0.05, CHIT_0 + 0.05, 1561)
        gamma_axis = np.linspace(GAMMA_AB_0 - 0.05, 0.05, 561)
        cg, gg = np.meshgrid(chit_axis, gamma_axis, indexing="ij")
        search_grid = np.column_stack([cg.ravel(), gg.ravel()])

        out = tau_obs_sweep_wrapped(targets, field, TAU_OBS_GRID, search_grid)
        assert out.validation.k_frust_invariant is True
        assert len(out.value) == len(TAU_OBS_GRID)
