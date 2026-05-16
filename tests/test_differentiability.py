"""Differentiability tests for the v2 JAX surface (BLOCK_IN §v2 cut (a)).

Validates `jax_core` / `jax_ops`:

  - Forward maps return JAX arrays compatible with `jax.grad`.
  - Autograd matches finite-difference within IEEE-754 tolerance.
  - The v2 forward-map values agree with the v0/v1 closed-form math
    at high precision (so consumers can compose either path without
    drifting).
  - Gradient-based inversion (BFGS) recovers canonical from substrate
    on tangent-flow fields at sub-tolerance precision.
  - Banach analytical state under `jax.grad` matches the closed-form
    partial derivatives.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mpa_scale_solver import (
    BanachSubstrate,
    CanonicalPoint,
    CanonicalState,
    OperatingPoint,
    ScalingRule,
    SubstrateState,
    TangentFlowField,
    TranslationField,
    TranslationRule,
    apply_translation,
    flow,
)
from mpa_scale_solver.jax_core import (
    banach_state,
    lookup_squared_distance,
    tangent_flow_canonical,
    tangent_flow_canonical_inverse,
    tangent_flow_inversion_residual,
    tangent_flow_substrate,
)
from mpa_scale_solver.jax_ops import (
    banach_state_diff,
    flow_diff,
    forward_sweep_invert_diff,
    tangent_flow_forward_jacobian,
    tangent_flow_substrate_diff,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tangent_field(*, delta_chit=0.5, delta_gamma=-0.3, tau_obs_ref=1.0):
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


def _banach_field(*, lambda_chit=1.0, lambda_gamma=1.0, chit_0=1.5, gamma_AB_0=-0.5):
    return BanachSubstrate(
        chit_0=chit_0, gamma_AB_0=gamma_AB_0,
        lambda_chit=lambda_chit, lambda_gamma=lambda_gamma,
    ).translation_field()


def _finite_diff(f, x, eps=1e-6):
    """Central-difference Jacobian of a vector-valued f at x (numpy)."""
    x = np.asarray(x, dtype=np.float64)
    fx = np.asarray(f(x), dtype=np.float64)
    n_in = x.size
    n_out = fx.size
    J = np.empty((n_out, n_in), dtype=np.float64)
    for j in range(n_in):
        xp = x.copy(); xm = x.copy()
        xp[j] += eps; xm[j] -= eps
        J[:, j] = (np.asarray(f(xp), dtype=np.float64) - np.asarray(f(xm), dtype=np.float64)) / (2 * eps)
    return J


# ---------------------------------------------------------------------------
# Float64 mode is active (required by jax_core's x64 commitment)
# ---------------------------------------------------------------------------

class TestJaxFloat64Active:
    def test_x64_enabled(self):
        from mpa_scale_solver import jax_core  # noqa: F401 — triggers x64
        assert jax.config.read("jax_enable_x64") is True


# ---------------------------------------------------------------------------
# v2 forward map matches v1 closed form (high precision)
# ---------------------------------------------------------------------------

class TestParityWithV1Math:
    @pytest.mark.parametrize("tau_obs", [0.01, 0.5, 1.0, math.e, 10.0])
    def test_tangent_flow_substrate_matches_v1(self, tau_obs):
        field = _tangent_field(delta_chit=0.7, delta_gamma=-0.4)
        canonical = CanonicalState(chit=1.2, gamma_AB=-0.6)

        # v1 path
        s_v1 = apply_translation(canonical, field, tau_obs=tau_obs)
        # v2 path
        s_chit, s_gamma = tangent_flow_substrate_diff(canonical, field, tau_obs)

        assert float(s_chit) == pytest.approx(s_v1.observables["substrate_chit"], abs=1e-12)
        assert float(s_gamma) == pytest.approx(s_v1.observables["substrate_gamma_AB"], abs=1e-12)

    @pytest.mark.parametrize("nu", [0.0, 0.5, 1.0, 2.0, 3.0, 5.0])
    def test_flow_diff_banach_matches_v1(self, nu):
        substrate = BanachSubstrate(chit_0=1.5, gamma_AB_0=-0.5)
        field = substrate.translation_field()

        chit_v1 = flow(substrate.canonical_initial(), nu, field).chit
        chit_v2, _ = flow_diff(substrate.canonical_initial(), nu, field)

        assert float(chit_v2) == pytest.approx(chit_v1, abs=1e-14)

    def test_banach_state_diff_matches_state_at(self):
        substrate = BanachSubstrate(chit_0=1.5, gamma_AB_0=-0.5)
        for nu in (0.0, 0.5, 1.0, 2.0, 5.0):
            truth = substrate.state_at(nu)
            chit, gamma = banach_state_diff(substrate, nu)
            assert float(chit) == pytest.approx(truth.chit, abs=1e-14)
            assert float(gamma) == pytest.approx(truth.gamma_AB, abs=1e-14)


# ---------------------------------------------------------------------------
# Autograd vs finite-difference — per primitive
# ---------------------------------------------------------------------------

class TestAutogradVsFiniteDiff:
    def test_tangent_flow_substrate_grad(self):
        """∂(substrate)/∂(canonical) matches finite difference."""
        params = jnp.array([0.8, -0.4], dtype=jnp.float64)
        delta_chit = jnp.float64(0.5)
        delta_gamma = jnp.float64(-0.3)
        tau = jnp.float64(2.0)
        ref = jnp.float64(1.0)

        def forward(p):
            sc, sg = tangent_flow_substrate(
                p[0], p[1], delta_chit, delta_gamma, tau, ref,
            )
            return jnp.stack([sc, sg])

        autograd_J = np.asarray(jax.jacfwd(forward)(params))
        fd_J = _finite_diff(lambda x: np.asarray(forward(jnp.asarray(x))), np.asarray(params))
        np.testing.assert_allclose(autograd_J, fd_J, rtol=1e-6, atol=1e-9)

    def test_banach_state_grad(self):
        """∂(canonical)/∂(chit_0, gamma_AB_0, lambda_chit, lambda_gamma, nu)
        matches finite difference."""
        params = jnp.array([1.5, -0.5, 1.0, 1.0, 2.0], dtype=jnp.float64)

        def forward(p):
            chit, gamma = banach_state(p[0], p[1], p[2], p[3], p[4])
            return jnp.stack([chit, gamma])

        autograd_J = np.asarray(jax.jacfwd(forward)(params))
        fd_J = _finite_diff(lambda x: np.asarray(forward(jnp.asarray(x))), np.asarray(params))
        np.testing.assert_allclose(autograd_J, fd_J, rtol=1e-6, atol=1e-9)

    def test_inversion_residual_grad(self):
        """Gradient of inversion residual w.r.t. (chit, gamma)."""
        delta_chit = jnp.float64(0.5)
        delta_gamma = jnp.float64(-0.3)
        tau = jnp.float64(2.0)
        ref = jnp.float64(1.0)
        target_chit = jnp.float64(1.2)
        target_gamma = jnp.float64(-0.5)

        def cost(p):
            return tangent_flow_inversion_residual(
                p[0], p[1], target_chit, target_gamma,
                delta_chit, delta_gamma, tau, ref,
            )

        params = jnp.array([0.7, -0.3], dtype=jnp.float64)
        autograd_grad = np.asarray(jax.grad(cost)(params))
        fd_grad = _finite_diff(
            lambda x: np.asarray([cost(jnp.asarray(x))]), np.asarray(params),
        ).flatten()
        np.testing.assert_allclose(autograd_grad, fd_grad, rtol=1e-6, atol=1e-9)


# ---------------------------------------------------------------------------
# JIT compiles the primitives cleanly
# ---------------------------------------------------------------------------

class TestJitCompiles:
    def test_tangent_flow_substrate_jit(self):
        jit_fn = jax.jit(tangent_flow_substrate)
        sc, sg = jit_fn(
            jnp.float64(1.2), jnp.float64(-0.6),
            jnp.float64(0.5), jnp.float64(-0.3),
            jnp.float64(2.0), jnp.float64(1.0),
        )
        # log(2) ≈ 0.6931472; chit -> 1.2 + 0.5*log(2)
        assert float(sc) == pytest.approx(1.2 + 0.5 * math.log(2.0), abs=1e-12)

    def test_banach_state_jit(self):
        jit_fn = jax.jit(banach_state)
        chit, gamma = jit_fn(
            jnp.float64(1.5), jnp.float64(-0.5),
            jnp.float64(1.0), jnp.float64(1.0), jnp.float64(2.0),
        )
        assert float(chit) == pytest.approx(1.5 * math.exp(-2.0), abs=1e-14)


# ---------------------------------------------------------------------------
# CanonicalState as a JAX PyTree
# ---------------------------------------------------------------------------

class TestCanonicalStatePyTree:
    def test_grad_through_canonical_state(self):
        """jax.grad over a function of CanonicalState works after registration."""
        from mpa_scale_solver import jax_pytree  # noqa: F401

        def cost(state: CanonicalState) -> jnp.ndarray:
            return state.chit ** 2 + 3.0 * state.gamma_AB ** 2

        grad_fn = jax.grad(cost)
        state = CanonicalState(chit=jnp.float64(0.7), gamma_AB=jnp.float64(-0.4))
        g = grad_fn(state)
        assert float(g.chit) == pytest.approx(2 * 0.7, abs=1e-12)
        assert float(g.gamma_AB) == pytest.approx(6 * (-0.4), abs=1e-12)

    def test_canonical_state_flatten_roundtrip(self):
        state = CanonicalState(chit=0.7, gamma_AB=-0.4, k_frust=True)
        leaves, treedef = jax.tree_util.tree_flatten(state)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        assert restored.chit == 0.7
        assert restored.gamma_AB == -0.4
        assert restored.k_frust is True


# ---------------------------------------------------------------------------
# Forward Jacobian — closed form vs autograd
# ---------------------------------------------------------------------------

class TestTangentFlowJacobian:
    def test_identity_field_jacobian_is_identity(self):
        """delta=0 -> substrate equals canonical -> Jacobian is identity."""
        field = _tangent_field(delta_chit=0.0, delta_gamma=0.0)
        J = tangent_flow_forward_jacobian(
            CanonicalState(chit=0.7, gamma_AB=-0.3), field, tau_obs=2.0,
        )
        np.testing.assert_allclose(np.asarray(J), np.eye(2), atol=1e-12)

    def test_diagonal_jacobian_under_axis_separable_scaling(self):
        """Tangent flow couples chit/gamma only through their own deltas.

        ∂substrate_chit/∂canonical_gamma_AB == 0  and vice versa.
        ∂substrate_chit/∂canonical_chit == 1.
        ∂substrate_gamma/∂canonical_gamma_AB == (tau/tau_ref)^delta_gamma.
        """
        field = _tangent_field(delta_chit=0.7, delta_gamma=-0.4)
        tau = 3.0
        J = np.asarray(tangent_flow_forward_jacobian(
            CanonicalState(chit=0.8, gamma_AB=-0.2), field, tau_obs=tau,
        ))
        assert J[0, 0] == pytest.approx(1.0, abs=1e-12)
        assert J[0, 1] == pytest.approx(0.0, abs=1e-12)
        assert J[1, 0] == pytest.approx(0.0, abs=1e-12)
        assert J[1, 1] == pytest.approx(tau ** -0.4, abs=1e-12)


# ---------------------------------------------------------------------------
# Exact closed-form inversion on tangent-flow fields
# ---------------------------------------------------------------------------

class TestForwardSweepInvertDiff:
    @pytest.mark.parametrize("tau_obs", [0.1, 1.0, 2.0, 5.0])
    def test_recovers_canonical_under_identity_field(self, tau_obs):
        """Identity tangent-flow: forward and inverse are both identity."""
        field = _tangent_field(delta_chit=0.0, delta_gamma=0.0)
        truth = CanonicalState(chit=0.7, gamma_AB=-0.3)
        target = apply_translation(truth, field, tau_obs=tau_obs)
        recovered = forward_sweep_invert_diff(target, field, tau_obs=tau_obs)
        assert recovered.chit == pytest.approx(truth.chit, abs=1e-12)
        assert recovered.gamma_AB == pytest.approx(truth.gamma_AB, abs=1e-12)

    @pytest.mark.parametrize("tau_obs", [0.5, 1.0, 2.0, 5.0])
    def test_recovers_canonical_under_nonidentity_field(self, tau_obs):
        """Non-trivial deltas: closed-form analytical inverse is exact."""
        field = _tangent_field(delta_chit=0.7, delta_gamma=-0.4)
        truth = CanonicalState(chit=0.8, gamma_AB=-0.25)
        target = apply_translation(truth, field, tau_obs=tau_obs)
        recovered = forward_sweep_invert_diff(target, field, tau_obs=tau_obs)
        assert recovered.chit == pytest.approx(truth.chit, abs=1e-12)
        assert recovered.gamma_AB == pytest.approx(truth.gamma_AB, abs=1e-12)

    def test_target_without_substrate_keys_raises(self):
        field = _tangent_field()
        bad_target = SubstrateState(tau_obs=1.0, observables={})
        with pytest.raises(ValueError, match="substrate_chit"):
            forward_sweep_invert_diff(bad_target, field, tau_obs=1.0)

    def test_inverse_propagates_k_frust(self):
        field = _tangent_field()
        target = apply_translation(
            CanonicalState(chit=0.5, gamma_AB=-0.2, k_frust=True),
            field, tau_obs=1.0,
        )
        recovered = forward_sweep_invert_diff(target, field, tau_obs=1.0, k_frust=True)
        assert recovered.k_frust is True


# ---------------------------------------------------------------------------
# Analytical inverse — exact + differentiable in the target
# ---------------------------------------------------------------------------

class TestTangentFlowAnalyticalInverse:
    def test_forward_inverse_roundtrip_exact(self):
        """forward then inverse returns canonical at float64 precision."""
        delta_chit = jnp.float64(0.7)
        delta_gamma = jnp.float64(-0.4)
        tau_obs = jnp.float64(2.0)
        tau_obs_ref = jnp.float64(1.0)

        canonical_in = jnp.array([0.83, -0.27], dtype=jnp.float64)

        s_chit, s_gamma = tangent_flow_substrate(
            canonical_in[0], canonical_in[1],
            delta_chit, delta_gamma, tau_obs, tau_obs_ref,
        )
        c_chit, c_gamma = tangent_flow_canonical_inverse(
            s_chit, s_gamma,
            delta_chit, delta_gamma, tau_obs, tau_obs_ref,
        )
        assert float(c_chit) == pytest.approx(float(canonical_in[0]), abs=1e-14)
        assert float(c_gamma) == pytest.approx(float(canonical_in[1]), abs=1e-14)

    def test_inverse_grad_matches_finite_diff(self):
        """∂canonical/∂substrate matches finite-difference."""
        delta_chit = jnp.float64(0.5)
        delta_gamma = jnp.float64(-0.3)
        tau_obs = jnp.float64(2.0)
        tau_obs_ref = jnp.float64(1.0)

        def invert(s):
            cc, cg = tangent_flow_canonical_inverse(
                s[0], s[1], delta_chit, delta_gamma, tau_obs, tau_obs_ref,
            )
            return jnp.stack([cc, cg])

        s = jnp.array([1.3, -0.19], dtype=jnp.float64)
        autograd_J = np.asarray(jax.jacfwd(invert)(s))
        fd_J = _finite_diff(lambda x: np.asarray(invert(jnp.asarray(x))), np.asarray(s))
        np.testing.assert_allclose(autograd_J, fd_J, rtol=1e-6, atol=1e-9)

    def test_inverse_identity_at_degenerate_tau(self):
        """tau_obs <= 0 -> identity inverse (matches forward map)."""
        c_chit, c_gamma = tangent_flow_canonical_inverse(
            jnp.float64(1.3), jnp.float64(-0.19),
            jnp.float64(0.5), jnp.float64(-0.3),
            jnp.float64(0.0), jnp.float64(1.0),
        )
        assert float(c_chit) == 1.3
        assert float(c_gamma) == -0.19


# ---------------------------------------------------------------------------
# Lookup squared-distance — differentiable in the query
# ---------------------------------------------------------------------------

class TestLookupSquaredDistance:
    def test_distance_minimum_at_field_canonical(self):
        field_chits = jnp.array([-1.0, 0.0, 1.0], dtype=jnp.float64)
        field_gammas = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float64)
        field_taus = jnp.zeros(3, dtype=jnp.float64)
        has_tau = jnp.zeros(3, dtype=bool)

        # Query exactly at field rule 1 (chit=0, gamma=0)
        d2 = lookup_squared_distance(
            jnp.float64(0.0), jnp.float64(0.0),
            field_chits, field_gammas, field_taus, has_tau,
            jnp.float64(1.0), jnp.float64(1.0),
        )
        assert float(d2[1]) == pytest.approx(0.0, abs=1e-12)
        assert float(d2[0]) == pytest.approx(1.0, abs=1e-12)
        assert float(d2[2]) == pytest.approx(1.0, abs=1e-12)

    def test_distance_grad_w_r_t_query(self):
        """Differentiable in the query coordinates."""
        field_chits = jnp.array([-1.0, 1.0], dtype=jnp.float64)
        field_gammas = jnp.array([0.0, 0.0], dtype=jnp.float64)
        field_taus = jnp.zeros(2, dtype=jnp.float64)
        has_tau = jnp.zeros(2, dtype=bool)

        def total_dist(q):
            d2 = lookup_squared_distance(
                q[0], q[1],
                field_chits, field_gammas, field_taus, has_tau,
                jnp.float64(1.0), jnp.float64(1.0),
            )
            return jnp.sum(d2)

        # ∂(sum (chit - field_chits)^2 + (gamma - field_gammas)^2) / ∂query
        # = (2*query_chit - 2*(-1) + 2*query_chit - 2*(1), 2*query_gamma - 0 + 2*query_gamma - 0)
        # at q=(0.3, 0.0): chit grad = 4*0.3 = 1.2; gamma grad = 0.0
        q = jnp.array([0.3, 0.0], dtype=jnp.float64)
        g = jax.grad(total_dist)(q)
        assert float(g[0]) == pytest.approx(1.2, abs=1e-12)
        assert float(g[1]) == pytest.approx(0.0, abs=1e-12)
