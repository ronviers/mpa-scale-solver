"""v5 sensitivity backprop (BLOCK_IN §v5).

Composes the per-op Jacobians in `jax_ops` into the full audit-traversal
chain rule, so driver-profile hyperparameters can be optimized via
gradient descent in a one-liner.

The audit traversal is `apply_translation → forward_sweep_invert →
tau_obs_sweep`: canonical state → substrate observation → recovered
canonical, per tau_obs frame across a sweep. v2.0 already gives us the
per-op differentiable forward map (`tangent_flow_substrate_diff`) and
the closed-form analytical inverse (`forward_sweep_invert_diff`,
exact at float64 precision for tangent-flow). v5 composes these into:

  - `trajectory_substrate_diff` — per-frame substrate observations as
    a JAX-traceable trajectory. Differentiable in canonical state and
    field parameters.
  - `trajectory_substrate_jacobian` — stack of per-frame Jacobians
    `∂substrate / ∂canonical` across the sweep.
  - `field_parameter_sensitivity` — per-frame Jacobian
    `∂substrate / ∂(delta_chit, delta_gamma, tau_obs_ref)` for
    tangent-flow fields; the surface driver-profile hyperparameter
    optimization composes against.
  - `driver_profile_loss_grad` — the one-liner the BLOCK_IN promised:
    given a loss `loss(predicted_substrates, observed_substrates)`,
    returns the gradient w.r.t. the field's hyperparameters.

Scope: tangent_flow and learned fields (the differentiable shapes).
Lookup-table fields have no differentiable forward map and stay on
the v0 grid path; cross-shape composition is the consumer's job.

Native-port note: every function here returns JAX arrays (not nested
dataclasses), so the v6 native port reproduces them as plain
multi-dimensional float64 arrays under autodiff (`enzyme` for Rust,
hand-written for C++). The Python-as-pseudo-code spec rule applies.
"""

from __future__ import annotations

from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .jax_core import (
    learned_field_substrate,
    tangent_flow_substrate,
)
from .jax_ops import (
    _learned_weights_as_jax,
    tangent_flow_substrate_diff,
)
from .types import (
    AnyTranslationField,
    CanonicalState,
    LearnedField,
    TangentFlowField,
    TranslationField,
)


# ---------------------------------------------------------------------------
# Trajectory forward map: substrate observations across a tau_obs sweep
# ---------------------------------------------------------------------------

def trajectory_substrate_diff(
    canonical: CanonicalState,
    field: AnyTranslationField,
    tau_obs_grid: np.ndarray,
) -> jnp.ndarray:
    """Per-frame substrate observations as a JAX-traceable trajectory.

    For each `tau_obs` in the grid, evaluates the differentiable forward
    map at the same `canonical` state and returns the substrate
    observable pair. The canonical state itself does not flow — this is
    the apply_translation surface, not the continuous-flow trajectory.
    For the post-flow version compose with `flow_diff`.

    Returns shape `(T, 2)` JAX array; row t is `(substrate_chit,
    substrate_gamma_AB)` at `tau_obs_grid[t]`.

    Differentiable in `canonical.chit`, `canonical.gamma_AB`, and the
    field's parameters. Lookup-table fields raise NotImplementedError
    (no differentiable forward map).
    """
    if isinstance(field, TranslationField):
        raise NotImplementedError(
            "trajectory_substrate_diff on lookup_table fields: the "
            "nearest-neighbor dispatch is non-differentiable through the "
            "rule choice. Use a TangentFlowField or LearnedField for "
            "differentiable trajectory sensitivity."
        )
    if not isinstance(field, (TangentFlowField, LearnedField)):
        raise TypeError(f"unsupported translation field type: {type(field).__name__}")

    rows = []
    for t in tau_obs_grid:
        s_chit, s_gamma = _forward_map_jax(canonical, field, float(t))
        rows.append(jnp.stack([s_chit, s_gamma]))
    return jnp.stack(rows)


def trajectory_substrate_jacobian(
    canonical: CanonicalState,
    field: AnyTranslationField,
    tau_obs_grid: np.ndarray,
) -> jnp.ndarray:
    """Stack of per-frame Jacobians `∂substrate / ∂canonical` across the sweep.

    Returns shape `(T, 2, 2)`; entry `[t, i, j]` is the partial of the
    i-th substrate observable w.r.t. the j-th canonical coordinate at
    frame t. Composes `tangent_flow_forward_jacobian` (the v2.0 per-op
    Jacobian) through the tau_obs grid; for learned fields the same
    chain rule is taken via `jax.jacfwd` on the closure.

    The sensitivity surface driver-profile hyperparameter optimization
    composes against.
    """
    if isinstance(field, TranslationField):
        raise NotImplementedError(
            "trajectory_substrate_jacobian on lookup_table fields is not "
            "supported; nearest-neighbor dispatch is non-differentiable."
        )

    jacobians = []
    for t in tau_obs_grid:
        jacobians.append(_forward_jacobian_jax(canonical, field, float(t)))
    return jnp.stack(jacobians)


# ---------------------------------------------------------------------------
# Field-parameter sensitivity — tangent_flow specific
# ---------------------------------------------------------------------------

def field_parameter_sensitivity(
    canonical: CanonicalState,
    field: TangentFlowField,
    tau_obs_grid: np.ndarray,
) -> jnp.ndarray:
    """Per-frame Jacobian of substrate w.r.t. `(delta_chit, delta_gamma, tau_obs_ref)`.

    Returns shape `(T, 2, 3)`; entry `[t, i, k]` is the partial of the
    i-th substrate observable w.r.t. the k-th hyperparameter in
    `(delta_chit, delta_gamma, tau_obs_ref)` at frame t.

    The substrate fits the tangent-flow scaling rule analytically:

        substrate_chit  = chit  + delta_chit  * log(tau / tau_ref)
        substrate_gamma = gamma * (tau / tau_ref) ** delta_gamma

    so the per-frame Jacobian has closed form; here we compose via
    `jax.jacfwd` on the closure for parity with the consumer-facing
    sensitivity surface (and so the same code can switch to learned
    fields without bifurcating).

    This is the surface `driver_profile_loss_grad` composes against
    for tangent-flow fields.
    """
    chit = jnp.asarray(canonical.chit, dtype=jnp.float64)
    gamma = jnp.asarray(canonical.gamma_AB, dtype=jnp.float64)

    def forward_params(params: jnp.ndarray, tau: float) -> jnp.ndarray:
        delta_chit, delta_gamma, tau_ref = params[0], params[1], params[2]
        s_chit, s_gamma = tangent_flow_substrate(
            chit, gamma, delta_chit, delta_gamma,
            jnp.asarray(tau, dtype=jnp.float64), tau_ref,
        )
        return jnp.stack([s_chit, s_gamma])

    rule = field.scaling
    params = jnp.array(
        [rule.delta_chit, rule.delta_gamma, rule.tau_obs_ref],
        dtype=jnp.float64,
    )
    jac_fn = jax.jacfwd(forward_params)
    jacobians = [jac_fn(params, float(t)) for t in tau_obs_grid]
    return jnp.stack(jacobians)


# ---------------------------------------------------------------------------
# Driver-profile loss gradient — the one-liner BLOCK_IN §v5 promised
# ---------------------------------------------------------------------------

def driver_profile_loss_grad(
    loss_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    canonical: CanonicalState,
    field: TangentFlowField,
    tau_obs_grid: np.ndarray,
    observed_substrates: np.ndarray,
) -> dict:
    """Gradient of `loss(predicted_substrates, observed)` w.r.t. field hyperparams.

    `loss_fn` takes a `(T, 2)` predicted-substrate trajectory and a
    `(T, 2)` observed-substrate trajectory and returns a scalar. Common
    choices: mean squared residual, max residual, weighted L2.

    Returns a dict:

        {"loss": float,
         "grad_delta_chit": float,
         "grad_delta_gamma": float,
         "grad_tau_obs_ref": float}

    The gradient composes `loss_fn` over the trajectory through the
    tangent-flow forward map. Tangent-flow only at v5 — learned-field
    hyperparameter optimization is the curator-path's job at v3
    (mpa-conform), not the solver's.

    Usage:

        observed = numpy.array([...])  # shape (T, 2)
        result = driver_profile_loss_grad(
            lambda p, o: jnp.mean((p - o) ** 2),
            canonical_initial, field, tau_grid, observed,
        )
        # gradient-descent step: field.scaling.delta_chit -= lr * result["grad_delta_chit"]

    Observe-only: this function does not mutate the field. Consumers
    apply the gradient update themselves (the seven-op surface is
    immutable; fields are frozen dataclasses).
    """
    if not isinstance(field, TangentFlowField):
        raise TypeError(
            "driver_profile_loss_grad currently supports TangentFlowField only "
            "(closed-form parameter Jacobian). Learned-field training lives in "
            "mpa-conform's curator path."
        )

    chit = jnp.asarray(canonical.chit, dtype=jnp.float64)
    gamma = jnp.asarray(canonical.gamma_AB, dtype=jnp.float64)
    observed = jnp.asarray(observed_substrates, dtype=jnp.float64)
    if observed.ndim != 2 or observed.shape[1] != 2:
        raise ValueError(
            f"observed_substrates must have shape (T, 2); "
            f"got {observed.shape}"
        )
    if observed.shape[0] != len(tau_obs_grid):
        raise ValueError(
            f"observed length {observed.shape[0]} != "
            f"tau_obs_grid length {len(tau_obs_grid)}"
        )

    tau_arr = jnp.asarray(tau_obs_grid, dtype=jnp.float64)

    def predicted_from_params(params: jnp.ndarray) -> jnp.ndarray:
        delta_chit, delta_gamma, tau_ref = params[0], params[1], params[2]
        def per_frame(tau: jnp.ndarray) -> jnp.ndarray:
            s_chit, s_gamma = tangent_flow_substrate(
                chit, gamma, delta_chit, delta_gamma, tau, tau_ref,
            )
            return jnp.stack([s_chit, s_gamma])
        return jax.vmap(per_frame)(tau_arr)

    def loss_of_params(params: jnp.ndarray) -> jnp.ndarray:
        predicted = predicted_from_params(params)
        return loss_fn(predicted, observed)

    rule = field.scaling
    params_init = jnp.array(
        [rule.delta_chit, rule.delta_gamma, rule.tau_obs_ref],
        dtype=jnp.float64,
    )
    loss_value, grad = jax.value_and_grad(loss_of_params)(params_init)
    return {
        "loss": float(loss_value),
        "grad_delta_chit": float(grad[0]),
        "grad_delta_gamma": float(grad[1]),
        "grad_tau_obs_ref": float(grad[2]),
    }


# ---------------------------------------------------------------------------
# Inversion sensitivity — closed-form for tangent_flow
# ---------------------------------------------------------------------------

def inversion_sensitivity(
    target_substrate_pair: jnp.ndarray,
    field: TangentFlowField,
    tau_obs: float,
) -> jnp.ndarray:
    """Jacobian `∂(recovered_canonical) / ∂(substrate_observation)`.

    For tangent-flow fields the analytical inverse is closed form
    (`jax_core.tangent_flow_canonical_inverse`); this function returns
    the 2×2 Jacobian at the given target via `jax.jacfwd`. The
    surface composes with `field_parameter_sensitivity` to give the
    full chain rule through the audit traversal:

        ∂recovered / ∂field_params = (∂recovered / ∂substrate)
                                   · (∂substrate / ∂field_params)

    Useful when consumers want to understand how observation noise
    propagates to canonical-state uncertainty (the v2.1 Laplace
    posterior covariance is the integrated form of this).
    """
    from .jax_core import tangent_flow_canonical_inverse

    rule = field.scaling
    delta_chit = jnp.asarray(rule.delta_chit, dtype=jnp.float64)
    delta_gamma = jnp.asarray(rule.delta_gamma, dtype=jnp.float64)
    tau_obs_jax = jnp.asarray(tau_obs, dtype=jnp.float64)
    tau_obs_ref = jnp.asarray(rule.tau_obs_ref, dtype=jnp.float64)

    def invert(s: jnp.ndarray) -> jnp.ndarray:
        cc, cg = tangent_flow_canonical_inverse(
            s[0], s[1], delta_chit, delta_gamma, tau_obs_jax, tau_obs_ref,
        )
        return jnp.stack([cc, cg])

    target = jnp.asarray(target_substrate_pair, dtype=jnp.float64)
    return jax.jacfwd(invert)(target)


# ---------------------------------------------------------------------------
# Internal: dispatch the differentiable forward map / Jacobian by field shape
# ---------------------------------------------------------------------------

def _forward_map_jax(
    canonical: CanonicalState,
    field: AnyTranslationField,
    tau_obs: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Differentiable forward map dispatched on field shape."""
    if isinstance(field, TangentFlowField):
        return tangent_flow_substrate_diff(canonical, field, tau_obs)
    if isinstance(field, LearnedField):
        weights = _learned_weights_as_jax(field)
        return learned_field_substrate(
            chit=jnp.asarray(canonical.chit, dtype=jnp.float64),
            gamma_AB=jnp.asarray(canonical.gamma_AB, dtype=jnp.float64),
            tau_obs=jnp.asarray(tau_obs, dtype=jnp.float64),
            tau_obs_ref=jnp.asarray(field.tau_obs_ref, dtype=jnp.float64),
            weights=weights,
            activation=field.activation,
        )
    raise TypeError(f"unsupported translation field type: {type(field).__name__}")


def _forward_jacobian_jax(
    canonical: CanonicalState,
    field: AnyTranslationField,
    tau_obs: float,
) -> jnp.ndarray:
    """2×2 forward Jacobian dispatched on field shape."""
    def forward(c: jnp.ndarray) -> jnp.ndarray:
        s_chit, s_gamma = _forward_map_jax(
            CanonicalState(chit=c[0], gamma_AB=c[1], k_frust=canonical.k_frust),
            field, tau_obs,
        )
        return jnp.stack([s_chit, s_gamma])

    c0 = jnp.array([canonical.chit, canonical.gamma_AB], dtype=jnp.float64)
    return jax.jacfwd(forward)(c0)
