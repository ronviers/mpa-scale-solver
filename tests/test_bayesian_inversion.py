"""Bayesian inversion tests for v2.1 (BLOCK_IN §v2 cut b).

Validates `operations.forward_sweep_invert_posterior` and its wrapped
variant, plus the `jax_core` Laplace primitives:

  - Tangent-flow posterior fast path: MAP exact, covariance =
    noise_variance * inv(J^T J), log-evidence finite for
    well-conditioned Jacobians.
  - Lookup-table posterior: MAP at brute-force argmin, covariance from
    softmax-weighted moments over top-k candidates.
  - Identity field: posterior covariance is `noise_variance * I` (the
    forward map's Jacobian is identity so J^T J = I).
  - Noise-variance scaling: doubling sigma^2 doubles covariance.
  - Wrapped variant carries Posterior through validation + provenance.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mpa_scale_solver import (
    CanonicalPoint,
    CanonicalState,
    DispatchPath,
    OperatingPoint,
    Posterior,
    ScalingRule,
    SubstrateState,
    TangentFlowField,
    TranslationField,
    TranslationRule,
    apply_translation,
    forward_sweep_invert_posterior,
    forward_sweep_invert_posterior_wrapped,
)
from mpa_scale_solver.jax_core import (
    laplace_covariance_from_jacobian,
    laplace_log_evidence,
)
from mpa_scale_solver.jax_ops import (
    lookup_table_posterior,
    tangent_flow_posterior,
)


# ---------------------------------------------------------------------------
# Fixtures
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


def _three_cell_field() -> TranslationField:
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
# Tangent-flow posterior (closed-form fast path)
# ---------------------------------------------------------------------------

class TestTangentFlowPosterior:
    @pytest.mark.parametrize("tau_obs", [0.5, 1.0, 2.0])
    def test_map_recovers_truth_exactly(self, tau_obs):
        """MAP comes from the exact closed-form inverse."""
        field = _tangent_field(delta_chit=0.7, delta_gamma=-0.4)
        truth = CanonicalState(chit=0.8, gamma_AB=-0.25)
        target = apply_translation(truth, field, tau_obs=tau_obs)

        posterior = tangent_flow_posterior(target, field, tau_obs=tau_obs)
        assert isinstance(posterior, Posterior)
        assert posterior.mean.chit == pytest.approx(truth.chit, abs=1e-12)
        assert posterior.mean.gamma_AB == pytest.approx(truth.gamma_AB, abs=1e-12)

    def test_identity_field_covariance_equals_noise_times_identity(self):
        """delta = 0: J = I, so covariance = noise_variance * I."""
        field = _tangent_field(delta_chit=0.0, delta_gamma=0.0)
        truth = CanonicalState(chit=0.6, gamma_AB=-0.2)
        target = apply_translation(truth, field, tau_obs=1.0)

        posterior = tangent_flow_posterior(
            target, field, tau_obs=1.0, noise_variance=1.0,
        )
        cov = np.array(posterior.covariance)
        np.testing.assert_allclose(cov, np.eye(2), atol=1e-12)

    def test_noise_variance_scales_covariance_linearly(self):
        """Posterior covariance = noise_variance * inv(J^T J)."""
        field = _tangent_field(delta_chit=0.5, delta_gamma=-0.3)
        target = apply_translation(
            CanonicalState(chit=0.7, gamma_AB=-0.3), field, tau_obs=1.5,
        )

        p1 = tangent_flow_posterior(target, field, tau_obs=1.5, noise_variance=1.0)
        p2 = tangent_flow_posterior(target, field, tau_obs=1.5, noise_variance=2.0)

        cov1 = np.array(p1.covariance)
        cov2 = np.array(p2.covariance)
        np.testing.assert_allclose(cov2, 2.0 * cov1, atol=1e-12)

    def test_log_evidence_is_finite_for_well_conditioned_field(self):
        field = _tangent_field(delta_chit=0.5, delta_gamma=-0.3)
        target = apply_translation(
            CanonicalState(chit=0.7, gamma_AB=-0.3), field, tau_obs=1.5,
        )
        posterior = tangent_flow_posterior(target, field, tau_obs=1.5)
        assert posterior.log_evidence is not None
        assert math.isfinite(posterior.log_evidence)

    def test_k_frust_propagated(self):
        field = _tangent_field()
        target = apply_translation(
            CanonicalState(chit=0.5, gamma_AB=-0.2, k_frust=True),
            field, tau_obs=1.0,
        )
        posterior = tangent_flow_posterior(target, field, tau_obs=1.0, k_frust=True)
        assert posterior.mean.k_frust is True


# ---------------------------------------------------------------------------
# Lookup-table posterior (weighted-moment fit)
# ---------------------------------------------------------------------------

class TestLookupTablePosterior:
    def test_map_candidate_recorded_in_modes(self):
        """For an exact-match target the argmin candidate is the chit=+1 rule.

        The mean is a softmax-weighted blend of top-k candidates (at
        noise_variance=1.0 the chit=0 and chit=-1 candidates still
        carry weight despite higher residuals), so MAP-by-argmin lives
        in `posterior.modes` when it differs from the blended mean.
        """
        field = _three_cell_field()
        target = SubstrateState(
            tau_obs=1.0,
            label="chit=+1",
            axes={"chit_label": 1.0},
            observables={},
        )
        grid = np.array([[c, 0.0] for c in [-1.0, 0.0, 1.0]])
        posterior = lookup_table_posterior(
            target, field, tau_obs=1.0, canonical_grid=grid,
        )
        # MAP-by-argmin is in `modes` (mean differs from MAP here).
        assert len(posterior.modes) == 1
        assert posterior.modes[0].chit == pytest.approx(1.0, abs=1e-12)

    def test_low_noise_concentrates_mean_at_argmin(self):
        """At small noise_variance the softmax weight collapses to argmin."""
        field = _three_cell_field()
        target = SubstrateState(
            tau_obs=1.0,
            label="chit=+1",
            axes={"chit_label": 1.0},
            observables={},
        )
        grid = np.array([[c, 0.0] for c in [-1.0, 0.0, 1.0]])
        posterior = lookup_table_posterior(
            target, field, tau_obs=1.0, canonical_grid=grid,
            noise_variance=1e-6,
        )
        # noise_variance << residual-step -> weight collapses to argmin
        assert posterior.mean.chit == pytest.approx(1.0, abs=1e-9)

    def test_covariance_positive_definite(self):
        """Lookup posterior covariance is positive semi-definite."""
        field = _three_cell_field()
        target = SubstrateState(tau_obs=1.0, axes={"chit_label": 0.5}, observables={})
        grid = np.array([[c, 0.0] for c in np.linspace(-1, 1, 21)])
        posterior = lookup_table_posterior(
            target, field, tau_obs=1.0, canonical_grid=grid,
        )
        cov = np.array(posterior.covariance)
        eigvals = np.linalg.eigvalsh(cov)
        assert np.all(eigvals >= -1e-12)  # PSD modulo numerical jitter

    def test_lookup_posterior_via_top_level_dispatch(self):
        """forward_sweep_invert_posterior dispatches lookup_table -> moment fit."""
        field = _three_cell_field()
        target = SubstrateState(tau_obs=1.0, axes={"chit_label": 0.0}, observables={})
        grid = np.array([[c, 0.0] for c in [-1.0, 0.0, 1.0]])
        posterior = forward_sweep_invert_posterior(
            target, field, tau_obs=1.0, canonical_grid=grid, top_k=3,
        )
        assert isinstance(posterior, Posterior)
        # Symmetric three-cell + symmetric residual surface -> mean near 0
        assert posterior.mean.chit == pytest.approx(0.0, abs=0.05)

    def test_lookup_posterior_requires_grid(self):
        field = _three_cell_field()
        target = SubstrateState(tau_obs=1.0, axes={"chit_label": 0.0}, observables={})
        with pytest.raises(ValueError, match="canonical_grid"):
            forward_sweep_invert_posterior(target, field, tau_obs=1.0)


# ---------------------------------------------------------------------------
# Top-level dispatch + wrapped variant
# ---------------------------------------------------------------------------

class TestForwardSweepInvertPosteriorDispatch:
    def test_tangent_flow_dispatch(self):
        field = _tangent_field()
        target = apply_translation(
            CanonicalState(chit=0.5, gamma_AB=-0.2), field, tau_obs=1.0,
        )
        posterior = forward_sweep_invert_posterior(target, field, tau_obs=1.0)
        assert isinstance(posterior, Posterior)
        assert posterior.mean.chit == pytest.approx(0.5, abs=1e-12)

    def test_wrapped_variant_carries_provenance(self):
        field = _tangent_field()
        target = apply_translation(
            CanonicalState(chit=0.5, gamma_AB=-0.2), field, tau_obs=1.0,
        )
        out = forward_sweep_invert_posterior_wrapped(
            target, field, tau_obs=1.0,
        )
        assert isinstance(out.value, Posterior)
        assert out.provenance.operation == "forward_sweep_invert_posterior"
        assert out.provenance.dispatch_path == DispatchPath.DIRECT_COMPUTE
        assert out.provenance.solver_version.startswith("2.")
        assert out.validation.asymptotic_closure_compliant is True


# ---------------------------------------------------------------------------
# jax_core Laplace primitives
# ---------------------------------------------------------------------------

class TestLaplaceCovariance:
    def test_identity_jacobian_returns_noise_times_identity(self):
        I = jnp.eye(2, dtype=jnp.float64)
        cov = laplace_covariance_from_jacobian(I, jnp.float64(3.0))
        np.testing.assert_allclose(np.asarray(cov), 3.0 * np.eye(2), atol=1e-12)

    def test_log_evidence_known_closed_form(self):
        """Known case: J = I, sigma^2 = 1, residual = 0, dim_y = dim_c = 2.

            log p(y) = -0.5 * 0 / 1
                       - 0.5 * 2 * log(2*pi*1)
                       + 0.5 * 2 * log(2*pi)
                       - 0.5 * log det(I)
                     = -2*log(2*pi)/2 + 2*log(2*pi)/2 - 0
                     = 0
        """
        H = jnp.eye(2, dtype=jnp.float64)
        log_ev = laplace_log_evidence(
            residual_at_map=jnp.float64(0.0),
            hessian=H,
            noise_variance=jnp.float64(1.0),
            n_obs=2,
        )
        assert float(log_ev) == pytest.approx(0.0, abs=1e-12)
