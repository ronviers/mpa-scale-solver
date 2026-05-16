"""v3 — LearnedField translation field (BLOCK_IN §v3).

Third translation-field shape alongside lookup_table (v0) and tangent_flow
(v1). A small MLP `(chit, gamma_AB, log(tau/tau_ref))` -> `(substrate_chit,
substrate_gamma_AB)`, evaluated through `jax_core.learned_field_substrate`.

Acceptance:
  - apply_translation dispatches on shape == "learned".
  - The closed-form forward map composes through jax.grad / jit.
  - parse_translation_field round-trips a JSON-shaped learned field.
  - forward_sweep_invert against a LearnedField works (grid-search path).
  - An identity-ish MLP reproduces the input (small numerical tolerance).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mpa_scale_solver import (
    CanonicalState,
    LearnedField,
    OperatingPoint,
    TranslationRule,
    apply_translation,
    forward_sweep_invert,
    parse_translation_field,
)
from mpa_scale_solver.jax_core import learned_field_substrate, mlp_forward
from mpa_scale_solver.jax_ops import learned_field_substrate_diff


def _origin_rule() -> TranslationRule:
    return TranslationRule(
        operating_point=OperatingPoint(label="origin", gt="c", axes={"tau_obs": 1.0}),
        xdot_choice="default",
        canonical=_canonical_point_at_origin(),
    )


def _canonical_point_at_origin():
    from mpa_scale_solver import CanonicalPoint
    return CanonicalPoint(chit=0.0, gamma_AB=0.0, k_frust=False, method="learned")


def _identity_learned_field() -> LearnedField:
    """Two-layer MLP roughly approximating an identity from (chit, gamma, log_ratio) -> (substrate_chit, substrate_gamma).

    Hidden layer dim 4; the projection matrix in the first layer picks chit
    and gamma_AB onto two of the hidden units, output layer reads them back.
    Activation is tanh, so chit/gamma must be within tanh's linear regime
    (|chit|, |gamma| << 1) for the identity approximation to hold.

    For tau_obs == tau_obs_ref, log_ratio = 0 and tanh(0) = 0, so the third
    input is suppressed by construction.
    """
    # W1: (4, 3) maps [chit, gamma, log_ratio] -> hidden
    # h0 ≈ tanh(small * chit) ≈ small * chit
    # h1 ≈ tanh(small * gamma) ≈ small * gamma
    eps = 0.01
    W1 = (
        (eps, 0.0, 0.0),
        (0.0, eps, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )
    b1 = (0.0, 0.0, 0.0, 0.0)
    # W2: (2, 4) reads first two hidden units back, scales by 1/eps to
    # cancel the linearization scaling.
    W2 = (
        (1.0 / eps, 0.0, 0.0, 0.0),
        (0.0, 1.0 / eps, 0.0, 0.0),
    )
    b2 = (0.0, 0.0)
    return LearnedField(
        direction="forward",
        shape="learned",
        rule_at_origin=_origin_rule(),
        weights=((W1, b1), (W2, b2)),
        architecture=(3, 4, 2),
        activation="tanh",
        tau_obs_ref=1.0,
    )


# ---------------------------------------------------------------------------
# jax_core.mlp_forward primitive
# ---------------------------------------------------------------------------

class TestMLPForward:
    def test_single_layer_linear_passthrough(self):
        # One-layer "MLP" with no hidden units = pure linear map.
        W = jnp.eye(3, dtype=jnp.float64)
        b = jnp.zeros(3, dtype=jnp.float64)
        x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
        y = mlp_forward(x, ((W, b),))
        np.testing.assert_allclose(np.asarray(y), [1.0, 2.0, 3.0])

    def test_two_layer_tanh(self):
        # Two layers with tanh on the hidden.
        W1 = jnp.array([[1.0, 0.0]], dtype=jnp.float64)
        b1 = jnp.array([0.0], dtype=jnp.float64)
        W2 = jnp.array([[1.0]], dtype=jnp.float64)
        b2 = jnp.array([0.0], dtype=jnp.float64)
        x = jnp.array([0.5, 0.0], dtype=jnp.float64)
        y = mlp_forward(x, ((W1, b1), (W2, b2)), activation="tanh")
        expected = float(jnp.tanh(0.5))
        np.testing.assert_allclose(np.asarray(y), [expected])

    def test_relu_activation(self):
        W1 = jnp.array([[1.0]], dtype=jnp.float64)
        b1 = jnp.array([0.0], dtype=jnp.float64)
        W2 = jnp.array([[1.0]], dtype=jnp.float64)
        b2 = jnp.array([0.0], dtype=jnp.float64)
        out_pos = mlp_forward(jnp.array([1.0]), ((W1, b1), (W2, b2)), activation="relu")
        out_neg = mlp_forward(jnp.array([-1.0]), ((W1, b1), (W2, b2)), activation="relu")
        np.testing.assert_allclose(np.asarray(out_pos), [1.0])
        np.testing.assert_allclose(np.asarray(out_neg), [0.0])

    def test_unknown_activation_raises(self):
        with pytest.raises(ValueError, match="unsupported activation"):
            mlp_forward(jnp.array([0.0]), ((jnp.eye(1), jnp.zeros(1)),), activation="sigmoid")


# ---------------------------------------------------------------------------
# learned_field_substrate (jax_core) + learned_field_substrate_diff (jax_ops)
# ---------------------------------------------------------------------------

class TestLearnedFieldSubstrate:
    def test_identity_approximation_at_tau_ref(self):
        field = _identity_learned_field()
        state = CanonicalState(chit=0.1, gamma_AB=-0.05)
        s_chit, s_gamma = learned_field_substrate_diff(state, field, 1.0)
        np.testing.assert_allclose(float(s_chit), 0.1, atol=1e-3)
        np.testing.assert_allclose(float(s_gamma), -0.05, atol=1e-3)

    def test_log_ratio_is_zero_when_tau_equals_ref(self):
        # Identity MLP's third input is suppressed; tau==tau_ref should
        # produce the same output as the canonical-only contribution.
        field = _identity_learned_field()
        state = CanonicalState(chit=0.2, gamma_AB=0.0)
        out_ref = learned_field_substrate_diff(state, field, 1.0)
        # At a different tau, the third input is nonzero but our W1 row 3
        # is zero, so the result is identical.
        out_other = learned_field_substrate_diff(state, field, 2.0)
        np.testing.assert_allclose(float(out_ref[0]), float(out_other[0]), atol=1e-12)
        np.testing.assert_allclose(float(out_ref[1]), float(out_other[1]), atol=1e-12)

    def test_jax_grad_traces(self):
        """jax.grad over learned_field_substrate composes (differentiability)."""
        field = _identity_learned_field()
        weights_jax = tuple(
            (jnp.asarray(W, dtype=jnp.float64), jnp.asarray(b, dtype=jnp.float64))
            for W, b in field.weights
        )

        def f(chit_val):
            s_chit, _ = learned_field_substrate(
                chit=chit_val,
                gamma_AB=jnp.asarray(0.0, dtype=jnp.float64),
                tau_obs=jnp.asarray(1.0, dtype=jnp.float64),
                tau_obs_ref=jnp.asarray(1.0, dtype=jnp.float64),
                weights=weights_jax,
                activation="tanh",
            )
            return s_chit

        # At chit=0 and tanh activation, the derivative is roughly the
        # composition of W1[0,0]*W2[0,0] near linear regime ~ 1.0.
        grad = jax.grad(f)(jnp.asarray(0.0, dtype=jnp.float64))
        np.testing.assert_allclose(float(grad), 1.0, atol=1e-3)

    def test_jit_compiles(self):
        field = _identity_learned_field()
        weights_jax = tuple(
            (jnp.asarray(W, dtype=jnp.float64), jnp.asarray(b, dtype=jnp.float64))
            for W, b in field.weights
        )

        @jax.jit
        def call(chit_val):
            return learned_field_substrate(
                chit=chit_val,
                gamma_AB=jnp.asarray(0.0, dtype=jnp.float64),
                tau_obs=jnp.asarray(1.0, dtype=jnp.float64),
                tau_obs_ref=jnp.asarray(1.0, dtype=jnp.float64),
                weights=weights_jax,
                activation="tanh",
            )

        out = call(jnp.asarray(0.1, dtype=jnp.float64))
        np.testing.assert_allclose(float(out[0]), 0.1, atol=1e-3)


# ---------------------------------------------------------------------------
# apply_translation dispatch on shape == "learned"
# ---------------------------------------------------------------------------

class TestApplyTranslationLearned:
    def test_apply_translation_returns_substrate_state(self):
        field = _identity_learned_field()
        state = CanonicalState(chit=0.1, gamma_AB=0.2)
        substrate = apply_translation(state, field, tau_obs=1.0)
        assert substrate.label == "origin"
        assert "substrate_chit" in substrate.observables
        assert "substrate_gamma_AB" in substrate.observables
        np.testing.assert_allclose(substrate.observables["substrate_chit"], 0.1, atol=1e-3)
        np.testing.assert_allclose(substrate.observables["substrate_gamma_AB"], 0.2, atol=1e-3)
        assert substrate.tau_obs == 1.0

    def test_apply_translation_tau_obs_axis_recorded(self):
        field = _identity_learned_field()
        substrate = apply_translation(
            CanonicalState(chit=0.0, gamma_AB=0.0), field, tau_obs=3.0,
        )
        assert substrate.axes.get("tau_obs") == 3.0


# ---------------------------------------------------------------------------
# parse_translation_field (JSON round-trip)
# ---------------------------------------------------------------------------

class TestParseLearnedField:
    def test_parse_learned_shape(self):
        d = {
            "direction": "forward",
            "shape": "learned",
            "rule_at_origin": {
                "operating_point": {"label": "origin", "gt": "c", "tau_obs": 1.0},
                "xdot_choice": "default",
                "canonical": {
                    "chit": 0.0, "gamma_AB": 0.0,
                    "k_frust": False, "method": "learned",
                },
            },
            "weights": [
                [[[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                 [0.0, 0.0, 0.0, 0.0]],
                [[[100.0, 0.0, 0.0, 0.0], [0.0, 100.0, 0.0, 0.0]],
                 [0.0, 0.0]],
            ],
            "architecture": [3, 4, 2],
            "activation": "tanh",
            "tau_obs_ref": 1.0,
        }
        field = parse_translation_field(d)
        assert isinstance(field, LearnedField)
        assert field.shape == "learned"
        assert field.architecture == (3, 4, 2)
        assert field.activation == "tanh"
        # Round-trip a forward map.
        substrate = apply_translation(CanonicalState(chit=0.1, gamma_AB=0.2), field, 1.0)
        np.testing.assert_allclose(substrate.observables["substrate_chit"], 0.1, atol=1e-3)


# ---------------------------------------------------------------------------
# forward_sweep_invert against a LearnedField
# ---------------------------------------------------------------------------

class TestForwardSweepInvertLearned:
    def test_grid_search_inverts_identity_field(self):
        field = _identity_learned_field()
        truth = CanonicalState(chit=0.15, gamma_AB=-0.25)
        target = apply_translation(truth, field, tau_obs=1.0)
        grid = np.array(
            [[c, g] for c in np.linspace(-0.5, 0.5, 21)
             for g in np.linspace(-0.5, 0.5, 21)],
            dtype=np.float64,
        )
        recovered, residual = forward_sweep_invert(target, field, 1.0, grid)
        # Identity-ish MLP: recovery should land within a grid step.
        assert abs(recovered.chit - truth.chit) <= 0.06
        assert abs(recovered.gamma_AB - truth.gamma_AB) <= 0.06
        assert residual < 0.01
