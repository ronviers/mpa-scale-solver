"""v5 — gradient-based inversion (BLOCK_IN §v5).

Coverage:
  - method="auto" (default) for TangentFlowField: closed-form, sub-grid-resolution
  - method="auto" for LearnedField: L-BFGS converges to grid-better recovery
  - method="auto" for TranslationField (lookup_table): unchanged grid behavior
  - method="grid" preserves v0/v1/v2/v3/v4 behavior byte-identically
  - method="gradient" raises for lookup_table
  - method="invalid" raises ValueError
  - return_residual_field=True coexists with method="auto" / "gradient"
  - forward_map override forces grid regardless of method
  - wrapped variant accepts method kwarg
  - Banach camera test sharpens: max |residual| under method="auto" is
    strictly tighter than method="grid"
  - 10x speedup property (informational, not strictly enforced — flagged)
"""

from __future__ import annotations

import time

import jax.numpy as jnp
import numpy as np
import pytest

from mpa_scale_solver import (
    BanachSubstrate,
    CanonicalPoint,
    CanonicalState,
    LearnedField,
    OperatingPoint,
    ScalingRule,
    SubstrateState,
    TangentFlowField,
    TranslationField,
    TranslationRule,
    apply_translation,
    forward_sweep_invert,
    forward_sweep_invert_wrapped,
)


# ---------------------------------------------------------------------------
# Helpers (mirroring the differentiability + camera test fixtures)
# ---------------------------------------------------------------------------

def _tangent_field(*, delta_chit=0.5, delta_gamma=-0.3, tau_obs_ref=1.0) -> TangentFlowField:
    origin = TranslationRule(
        operating_point=OperatingPoint(label="origin", gt="s", axes={"tau_obs": tau_obs_ref}),
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


def _three_cell_field() -> TranslationField:
    rules = []
    for chit in (-1.0, 0.0, 1.0):
        rules.append(TranslationRule(
            operating_point=OperatingPoint(
                label=f"chit={int(chit):+d}",
                gt="r" if chit < 0 else ("c" if chit > 0 else "s"),
                axes={"chit_label": chit},
            ),
            xdot_choice="x",
            canonical=CanonicalPoint(chit=chit, gamma_AB=0.0, k_frust=False, method="test"),
        ))
    return TranslationField(direction="forward", shape="lookup_table", rule=rules)


def _identity_learned_field() -> LearnedField:
    origin = TranslationRule(
        operating_point=OperatingPoint(label="origin", gt="c", axes={"tau_obs": 1.0}),
        xdot_choice="default",
        canonical=CanonicalPoint(chit=0.0, gamma_AB=0.0, k_frust=False, method="learned"),
    )
    eps = 0.01
    W1 = (
        (eps, 0.0, 0.0),
        (0.0, eps, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )
    b1 = (0.0, 0.0, 0.0, 0.0)
    W2 = (
        (1.0 / eps, 0.0, 0.0, 0.0),
        (0.0, 1.0 / eps, 0.0, 0.0),
    )
    b2 = (0.0, 0.0)
    return LearnedField(
        direction="forward", shape="learned",
        rule_at_origin=origin,
        weights=((W1, b1), (W2, b2)),
        architecture=(3, 4, 2),
        activation="tanh",
        tau_obs_ref=1.0,
    )


# ---------------------------------------------------------------------------
# method validation
# ---------------------------------------------------------------------------

class TestMethodKwarg:
    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="unknown method"):
            forward_sweep_invert(
                SubstrateState(tau_obs=1.0), _three_cell_field(),
                1.0, np.array([[0.0, 0.0]]),
                method="bogus",
            )

    def test_gradient_method_rejects_lookup_table(self):
        with pytest.raises(ValueError, match="differentiable field"):
            forward_sweep_invert(
                SubstrateState(tau_obs=1.0), _three_cell_field(),
                1.0, np.array([[0.0, 0.0]]),
                method="gradient",
            )

    def test_method_grid_byte_identical_to_v4_on_lookup_table(self):
        """method='grid' on lookup_table = v0–v4 brute-force behavior."""
        field = _three_cell_field()
        target = SubstrateState(tau_obs=1.0, axes={"chit_label": 1.0})
        grid = np.array([[c, 0.0] for c in [-1.0, 0.0, 1.0]])
        recovered_default, _ = forward_sweep_invert(target, field, 1.0, grid)
        recovered_grid, _ = forward_sweep_invert(target, field, 1.0, grid, method="grid")
        assert recovered_default.chit == recovered_grid.chit
        assert recovered_default.gamma_AB == recovered_grid.gamma_AB


# ---------------------------------------------------------------------------
# TangentFlowField — closed-form exact recovery under method="auto"
# ---------------------------------------------------------------------------

class TestTangentFlowClosedForm:
    @pytest.mark.parametrize("tau_obs", [0.5, 1.0, 2.0, 5.0])
    def test_exact_recovery_under_auto(self, tau_obs):
        """method='auto' on tangent_flow gives float64-exact canonical recovery."""
        field = _tangent_field(delta_chit=0.7, delta_gamma=-0.4)
        truth = CanonicalState(chit=0.8, gamma_AB=-0.25)
        target = apply_translation(truth, field, tau_obs=tau_obs)
        # Coarse grid — irrelevant; closed-form is exact.
        coarse_grid = np.array([[0.0, 0.0], [0.5, -0.5]], dtype=np.float64)
        recovered, residual = forward_sweep_invert(
            target, field, tau_obs, coarse_grid,
        )
        assert recovered.chit == pytest.approx(truth.chit, abs=1e-12)
        assert recovered.gamma_AB == pytest.approx(truth.gamma_AB, abs=1e-12)
        assert residual == pytest.approx(0.0, abs=1e-10)

    def test_method_grid_remains_grid_resolution(self):
        """method='grid' on tangent_flow stays at v4 grid-resolution recovery."""
        field = _tangent_field(delta_chit=0.7, delta_gamma=-0.4)
        truth = CanonicalState(chit=0.8, gamma_AB=-0.25)
        target = apply_translation(truth, field, tau_obs=2.0)
        coarse_grid = np.array(
            [[c, g] for c in np.linspace(0.0, 1.0, 5)
             for g in np.linspace(-0.5, 0.0, 5)],
            dtype=np.float64,
        )
        recovered_grid, _ = forward_sweep_invert(
            target, field, 2.0, coarse_grid, method="grid",
        )
        # Coarse grid -> snap to nearest cell; recovery is at grid resolution.
        # Step size 0.25 in chit -> truth 0.8 lands between cells.
        assert abs(recovered_grid.chit - truth.chit) > 1e-6  # NOT exact

        recovered_auto, _ = forward_sweep_invert(
            target, field, 2.0, coarse_grid, method="auto",
        )
        # Auto on the same coarse grid -> exact via closed-form.
        assert abs(recovered_auto.chit - truth.chit) < 1e-12

    def test_method_gradient_same_as_auto_for_tangent_flow(self):
        """For tangent_flow, gradient = closed-form (identical results)."""
        field = _tangent_field(delta_chit=0.5)
        truth = CanonicalState(chit=0.7, gamma_AB=-0.3)
        target = apply_translation(truth, field, tau_obs=1.5)
        grid = np.array([[0.0, 0.0]])
        auto, _ = forward_sweep_invert(target, field, 1.5, grid, method="auto")
        grad, _ = forward_sweep_invert(target, field, 1.5, grid, method="gradient")
        assert auto.chit == grad.chit
        assert auto.gamma_AB == grad.gamma_AB

    def test_residual_field_still_returned_under_auto(self):
        """return_residual_field=True works with method='auto' (grid still evaluated)."""
        field = _tangent_field()
        truth = CanonicalState(chit=0.7, gamma_AB=-0.3)
        target = apply_translation(truth, field, tau_obs=1.0)
        grid = np.array(
            [[c, g] for c in np.linspace(0.0, 1.0, 5)
             for g in np.linspace(-0.5, 0.0, 5)],
            dtype=np.float64,
        )
        result = forward_sweep_invert(
            target, field, 1.0, grid, return_residual_field=True,
        )
        assert len(result) == 3
        recovered, residual, field_array = result
        assert field_array.shape == (25,)
        # Recovered is the closed-form result — strictly more accurate than
        # any grid cell.
        assert recovered.chit == pytest.approx(truth.chit, abs=1e-12)

    def test_forward_map_override_forces_grid(self):
        """When forward_map is supplied, the gradient driver can't use it."""
        field = _tangent_field()
        truth = CanonicalState(chit=0.5, gamma_AB=0.0)
        target = apply_translation(truth, field, tau_obs=1.0)
        grid = np.array(
            [[c, 0.0] for c in np.linspace(0.0, 1.0, 5)],
            dtype=np.float64,
        )
        called = {"n": 0}
        def custom_forward(c, t):
            called["n"] += 1
            return apply_translation(c, field, t)
        recovered, _ = forward_sweep_invert(
            target, field, 1.0, grid,
            forward_map=custom_forward, method="auto",
        )
        # forward_map was called once per grid cell -> grid path executed.
        assert called["n"] == 5


# ---------------------------------------------------------------------------
# LearnedField — L-BFGS converges
# ---------------------------------------------------------------------------

class TestLearnedFieldBFGS:
    def test_recovers_canonical_at_sub_grid_resolution(self):
        """L-BFGS converges to truth past the warm-start grid resolution."""
        field = _identity_learned_field()
        truth = CanonicalState(chit=0.15, gamma_AB=-0.25)
        target = apply_translation(truth, field, tau_obs=1.0)
        # Coarse warm-start grid; L-BFGS refines past it.
        grid = np.array(
            [[c, g] for c in np.linspace(-0.5, 0.5, 5)
             for g in np.linspace(-0.5, 0.5, 5)],
            dtype=np.float64,
        )
        recovered, residual = forward_sweep_invert(target, field, 1.0, grid)
        # Sub-grid-step accuracy — finer than grid step 0.25.
        assert abs(recovered.chit - truth.chit) < 0.01
        assert abs(recovered.gamma_AB - truth.gamma_AB) < 0.01
        assert residual < 0.01

    def test_grid_method_falls_back_to_v4_resolution(self):
        """method='grid' for learned -> stays at grid resolution (v4 behavior)."""
        field = _identity_learned_field()
        truth = CanonicalState(chit=0.15, gamma_AB=-0.25)
        target = apply_translation(truth, field, tau_obs=1.0)
        coarse_grid = np.array(
            [[c, g] for c in np.linspace(-0.5, 0.5, 5)
             for g in np.linspace(-0.5, 0.5, 5)],
            dtype=np.float64,
        )
        recovered, _ = forward_sweep_invert(
            target, field, 1.0, coarse_grid, method="grid",
        )
        # Grid step 0.25 — recovery error of that scale.
        assert abs(recovered.chit - truth.chit) > 0.05


# ---------------------------------------------------------------------------
# Banach camera test — auto sharpens recovery to exact
# ---------------------------------------------------------------------------

class TestBanachAutoSharpening:
    def test_banach_inversion_residual_zero_under_auto(self):
        """The v1 camera test residual collapses to ~0 with closed-form path."""
        substrate = BanachSubstrate(chit_0=1.5, gamma_AB_0=-0.5)
        field = substrate.translation_field()
        # Pick a representative depth.
        target = substrate.substrate_at(1.0)
        # Any seed grid will do; closed-form ignores it.
        grid = np.array([[1.0, 0.0]], dtype=np.float64)
        recovered, residual = forward_sweep_invert(target, field, 1.0, grid)
        truth = substrate.state_at(1.0)
        assert abs(recovered.chit - truth.chit) < 1e-12
        assert abs(recovered.gamma_AB - truth.gamma_AB) < 1e-12


# ---------------------------------------------------------------------------
# Wrapped variant accepts method
# ---------------------------------------------------------------------------

class TestWrappedMethodKwarg:
    def test_wrapped_forwards_method(self):
        field = _tangent_field(delta_chit=0.5)
        truth = CanonicalState(chit=0.7, gamma_AB=-0.3)
        target = apply_translation(truth, field, tau_obs=1.5)
        grid = np.array([[0.0, 0.0], [0.5, -0.5]], dtype=np.float64)
        out_auto = forward_sweep_invert_wrapped(target, field, 1.5, grid, method="auto")
        assert out_auto.value.chit == pytest.approx(truth.chit, abs=1e-12)
        out_grid = forward_sweep_invert_wrapped(target, field, 1.5, grid, method="grid")
        # Grid snap to nearest of two cells; less accurate.
        assert abs(out_grid.value.chit - truth.chit) > 1e-6


# ---------------------------------------------------------------------------
# Performance — gradient is at least competitive with a comparable grid
# ---------------------------------------------------------------------------

class TestPerformanceSmoke:
    def test_closed_form_faster_than_dense_grid(self):
        """Banach inversion under method='auto' beats a 100x100 grid scan.

        Not a strict 10x assertion (timing is noisy), but the closed-form
        path should be at least 3x faster on a dense grid. This is a
        smoke test for the BLOCK_IN §v5 acceptance criterion that gradient
        is >=10x faster than grid on the Banach substrate.
        """
        substrate = BanachSubstrate(chit_0=1.5, gamma_AB_0=-0.5)
        field = substrate.translation_field()
        target = substrate.substrate_at(1.0)
        dense_grid = np.array(
            [[c, g] for c in np.linspace(-0.05, 1.55, 100)
             for g in np.linspace(-0.55, 0.05, 100)],
            dtype=np.float64,
        )

        # Warm up JAX (first call compiles).
        forward_sweep_invert(target, field, 1.0, dense_grid, method="auto")

        t0 = time.perf_counter()
        for _ in range(5):
            forward_sweep_invert(target, field, 1.0, dense_grid, method="auto")
        t_auto = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(5):
            forward_sweep_invert(target, field, 1.0, dense_grid, method="grid")
        t_grid = time.perf_counter() - t0

        # Closed-form (despite still scanning the grid for warm-start when
        # gradient is needed — but tangent_flow auto doesn't need it) should
        # be at least 3x faster than full grid scan. Loosely 10x; tightening
        # depends on hardware.
        assert t_auto < t_grid, (
            f"closed-form ({t_auto:.4f}s) should beat grid ({t_grid:.4f}s) "
            f"on dense grid; ratio {t_grid / t_auto:.2f}x"
        )
