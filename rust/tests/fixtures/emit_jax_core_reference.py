"""Emit the bit-identity reference fixture for the Rust port.

Runs the Python `mpa_scale_solver` primitives over a small input sweep
and writes input + output pairs to `jax_core_reference.json` next to
this file. The Rust integration test at `rust/tests/bit_identity.rs`
consumes that fixture and asserts each Rust primitive reproduces the
Python output within a per-primitive ULP budget (BLOCK_IN §v6
"byte-identical for deterministic ops" check).

Module coverage:
  * `jax_core` (math.rs) — 12 primitives, session 2.
  * `gfdr_model` (gfdr_model.rs) + `sidecar` + `flow` — session 4.

Regeneration discipline: the fixture is committed. If any source
function changes, rerun this script and commit the JSON diff in the
same change. The Rust test will catch unintentional divergence.

Run with the repo's Python (any cwd is fine — the package is editable-
installed; if not, `pip install -e H:/mpa-scale-solver` once):

    python rust/tests/fixtures/emit_jax_core_reference.py
"""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp

from mpa_scale_solver import jax_core, gfdr_model
from mpa_scale_solver.flow import flow as flow_op
from mpa_scale_solver.operations import (
    intent_map as intent_map_op,
    regime_at_wrapped as regime_at_wrapped_op,
)
from mpa_scale_solver.provenance import make_provenance, provenance_hash
from mpa_scale_solver.sidecar import round_key as sidecar_round_key
from mpa_scale_solver.types import (
    CanonicalPoint,
    CanonicalState,
    DispatchPath,
    GamutSpec,
    OperatingPoint,
    ScalingRule,
    TangentFlowField,
    TranslationRule,
)


# ---------------------------------------------------------------------------
# Conversion helpers — JAX arrays / scalars → plain Python for JSON
# ---------------------------------------------------------------------------


def _f(x) -> float:
    return float(x)


def _row(x) -> list[float]:
    return [float(v) for v in jnp.asarray(x).tolist()]


def _mat(x) -> list[list[float]]:
    return [[float(v) for v in row] for row in jnp.asarray(x).tolist()]


# ---------------------------------------------------------------------------
# Per-primitive case generators. Each returns a list of
# `{"inputs": {...}, "outputs": {...}}` dicts.
# ---------------------------------------------------------------------------


def cases_tangent_flow_substrate() -> list[dict]:
    sweep = [
        (1.234, 2.345, 0.4, 0.6, 3.7, 1.0),       # generic
        (1.0, 2.0, 0.5, 0.3, 4.0, 4.0),           # tau == ref → identity
        (1.5, 2.5, 0.3, 0.7, 0.0, 4.0),           # degenerate tau_obs
        (1.5, 2.5, 0.3, 0.7, 4.0, 0.0),           # degenerate tau_obs_ref
        (0.5, 1.5, -0.3, -0.4, 2.0, 1.0),         # negative deltas
        (1.0, 2.0, 0.2, 0.5, 100.0, 1.0),         # large tau ratio
    ]
    out: list[dict] = []
    for chit, gamma, dchit, dgamma, tau, ref in sweep:
        a, b = jax_core.tangent_flow_substrate(
            jnp.float64(chit), jnp.float64(gamma),
            jnp.float64(dchit), jnp.float64(dgamma),
            jnp.float64(tau), jnp.float64(ref),
        )
        out.append({
            "inputs": {
                "chit": chit, "gamma_ab": gamma,
                "delta_chit": dchit, "delta_gamma": dgamma,
                "tau_obs": tau, "tau_obs_ref": ref,
            },
            "outputs": {"chit": _f(a), "gamma_ab": _f(b)},
        })
    return out


def cases_banach_state() -> list[dict]:
    sweep = [
        (1.5, 2.5, 0.1, 0.2, 1.0),
        (1.5, 2.5, 0.1, 0.2, 0.0),    # nu = 0 → identity
        (1.0, 1.0, 0.5, 0.7, 10.0),   # large nu
        (1.0, 2.0, 0.0, 0.0, 5.0),    # zero lambdas → unchanged
        (2.0, 3.0, 1.3, 0.4, 0.5),    # generic
    ]
    out: list[dict] = []
    for chit_0, gamma_0, lam_c, lam_g, nu in sweep:
        a, b = jax_core.banach_state(
            jnp.float64(chit_0), jnp.float64(gamma_0),
            jnp.float64(lam_c), jnp.float64(lam_g),
            jnp.float64(nu),
        )
        out.append({
            "inputs": {
                "chit_0": chit_0, "gamma_ab_0": gamma_0,
                "lambda_chit": lam_c, "lambda_gamma": lam_g,
                "nu": nu,
            },
            "outputs": {"chit": _f(a), "gamma_ab": _f(b)},
        })
    return out


def cases_tangent_flow_canonical() -> list[dict]:
    # Same math as tangent_flow_substrate (nu in the tau_obs slot).
    # Different sweep to broaden coverage rather than duplicate.
    sweep = [
        (1.0, 2.0, 0.3, 0.4, 1.5, 1.0),
        (1.0, 2.0, 0.3, 0.4, 1.0, 1.0),         # nu == ref → identity
        (1.0, 2.0, 0.3, 0.4, 0.0, 1.0),         # degenerate nu
        (1.234, 2.345, -0.2, 0.8, 5.0, 2.0),    # generic with negative delta_chit
    ]
    out: list[dict] = []
    for c0, g0, dchit, dgamma, nu, ref in sweep:
        a, b = jax_core.tangent_flow_canonical(
            jnp.float64(c0), jnp.float64(g0),
            jnp.float64(dchit), jnp.float64(dgamma),
            jnp.float64(nu), jnp.float64(ref),
        )
        out.append({
            "inputs": {
                "chit_0": c0, "gamma_ab_0": g0,
                "delta_chit": dchit, "delta_gamma": dgamma,
                "nu": nu, "tau_obs_ref": ref,
            },
            "outputs": {"chit": _f(a), "gamma_ab": _f(b)},
        })
    return out


def cases_lookup_squared_distance() -> list[dict]:
    sweep = [
        # 3 rules, no tau-carrying — third term collapses to 0
        {
            "query_chit": 0.0, "query_gamma": 0.0,
            "field_chits": [3.0, 0.0, -2.0],
            "field_gammas": [4.0, 0.0, 1.0],
            "field_taus": [1.0, 1.0, 1.0],
            "has_tau": [False, False, False],
            "tau_obs": 1.0, "tau_obs_weight": 1.0,
        },
        # Mixed has_tau, query at origin
        {
            "query_chit": 0.0, "query_gamma": 0.0,
            "field_chits": [0.0, 0.0],
            "field_gammas": [0.0, 0.0],
            "field_taus": [2.0, 3.0],
            "has_tau": [True, False],
            "tau_obs": 1.0, "tau_obs_weight": 3.0,
        },
        # All tau-carrying, off-origin query
        {
            "query_chit": 0.5, "query_gamma": 0.5,
            "field_chits": [1.0, 0.0, -0.5, 0.25],
            "field_gammas": [0.0, 1.0, 0.5, -0.25],
            "field_taus": [0.5, 2.0, 1.0, 4.0],
            "has_tau": [True, True, True, True],
            "tau_obs": 2.0, "tau_obs_weight": 0.5,
        },
        # Degenerate tau_obs (log_tau_q clamped to 0)
        {
            "query_chit": 1.0, "query_gamma": 1.0,
            "field_chits": [1.0, 0.0],
            "field_gammas": [1.0, 0.0],
            "field_taus": [1.0, 1.0],
            "has_tau": [True, False],
            "tau_obs": 0.0, "tau_obs_weight": 1.0,
        },
    ]
    out: list[dict] = []
    for case in sweep:
        d = jax_core.lookup_squared_distance(
            jnp.float64(case["query_chit"]),
            jnp.float64(case["query_gamma"]),
            jnp.array(case["field_chits"], dtype=jnp.float64),
            jnp.array(case["field_gammas"], dtype=jnp.float64),
            jnp.array(case["field_taus"], dtype=jnp.float64),
            jnp.array(case["has_tau"], dtype=jnp.bool_),
            jnp.float64(case["tau_obs"]),
            jnp.float64(case["tau_obs_weight"]),
        )
        out.append({"inputs": case, "outputs": {"d2": _row(d)}})
    return out


def cases_tangent_flow_canonical_inverse() -> list[dict]:
    # For each canonical pair, run forward then inverse — fixture stores
    # the (substrate_chit, substrate_gamma_ab) as the inverse inputs and
    # the recovered canonical as the outputs.
    sweep = [
        (1.234, 2.345, 0.4, 0.6, 3.7, 1.0),
        (1.0, 2.0, 0.5, 0.3, 4.0, 4.0),
        (1.5, 2.5, 0.3, 0.7, 0.0, 4.0),    # degenerate → identity
        (0.5, 1.5, -0.3, -0.4, 2.0, 1.0),
        (1.0, 2.0, 0.2, 0.5, 100.0, 1.0),
    ]
    out: list[dict] = []
    for chit, gamma, dchit, dgamma, tau, ref in sweep:
        s_chit, s_gamma = jax_core.tangent_flow_substrate(
            jnp.float64(chit), jnp.float64(gamma),
            jnp.float64(dchit), jnp.float64(dgamma),
            jnp.float64(tau), jnp.float64(ref),
        )
        r_chit, r_gamma = jax_core.tangent_flow_canonical_inverse(
            s_chit, s_gamma,
            jnp.float64(dchit), jnp.float64(dgamma),
            jnp.float64(tau), jnp.float64(ref),
        )
        out.append({
            "inputs": {
                "substrate_chit": _f(s_chit),
                "substrate_gamma_ab": _f(s_gamma),
                "delta_chit": dchit, "delta_gamma": dgamma,
                "tau_obs": tau, "tau_obs_ref": ref,
            },
            "outputs": {"chit": _f(r_chit), "gamma_ab": _f(r_gamma)},
        })
    return out


def cases_tangent_flow_inversion_residual() -> list[dict]:
    # All `(candidate, target)` pairs are specified explicitly so Python
    # and Rust evaluate the same inputs — avoids the libm-cancellation
    # near-zero residual that arises when target = forward(candidate) is
    # generated by one implementation and consumed by the other (cross-
    # implementation cancellation is not a porting bug; the MAP=0
    # property is covered analytically by tests/math.rs:
    # `tangent_flow_inversion_residual_zero_at_map`).
    sweep = [
        {"cc": 1.0, "cg": 2.0, "tc": 5.0, "tg": 5.0,
         "dchit": 0.2, "dgamma": 0.3, "tau": 2.0, "ref": 1.0},
        {"cc": 1.5, "cg": 2.5, "tc": 1.0, "tg": 2.0,
         "dchit": 0.3, "dgamma": 0.7, "tau": 0.0, "ref": 4.0},
        {"cc": 0.5, "cg": 1.5, "tc": 2.7, "tg": 3.1,
         "dchit": -0.4, "dgamma": 0.5, "tau": 3.0, "ref": 1.0},
        {"cc": 2.0, "cg": -1.0, "tc": 1.0, "tg": -2.0,
         "dchit": 0.1, "dgamma": 0.2, "tau": 1.5, "ref": 1.5},  # tau == ref
    ]
    out: list[dict] = []
    for case in sweep:
        cc = jnp.float64(case["cc"])
        cg = jnp.float64(case["cg"])
        tc = jnp.float64(case["tc"])
        tg = jnp.float64(case["tg"])
        dchit = jnp.float64(case["dchit"])
        dgamma = jnp.float64(case["dgamma"])
        tau = jnp.float64(case["tau"])
        ref = jnp.float64(case["ref"])
        res = jax_core.tangent_flow_inversion_residual(
            cc, cg, tc, tg, dchit, dgamma, tau, ref,
        )
        out.append({
            "inputs": {
                "candidate_chit": case["cc"], "candidate_gamma": case["cg"],
                "target_substrate_chit": case["tc"], "target_substrate_gamma": case["tg"],
                "delta_chit": case["dchit"], "delta_gamma": case["dgamma"],
                "tau_obs": case["tau"], "tau_obs_ref": case["ref"],
            },
            "outputs": {"residual": _f(res)},
        })
    return out


def cases_laplace_covariance_from_jacobian() -> list[dict]:
    # The Rust port hardcodes the 2x2 (J^T J) case; Jacobians are (n_obs, 2).
    sweep = [
        {"jacobian": [[1.0, 0.0], [0.0, 1.0]], "noise_variance": 0.25},
        {"jacobian": [[1.0, 0.5], [0.0, 2.0], [-0.5, 1.0]], "noise_variance": 1.0},
        {"jacobian": [[2.0, 1.0], [1.0, 2.0], [-1.0, 1.0], [0.5, -0.5]],
         "noise_variance": 0.1},
    ]
    out: list[dict] = []
    for case in sweep:
        cov = jax_core.laplace_covariance_from_jacobian(
            jnp.array(case["jacobian"], dtype=jnp.float64),
            jnp.float64(case["noise_variance"]),
        )
        out.append({"inputs": case, "outputs": {"covariance": _mat(cov)}})
    return out


def cases_laplace_covariance_from_hessian() -> list[dict]:
    sweep = [
        {"hessian": [[1.0, 0.0], [0.0, 1.0]], "noise_variance": 1.0},
        {"hessian": [[2.0, 0.5], [0.5, 3.0]], "noise_variance": 0.25},
        {"hessian": [[4.0, -1.0], [-1.0, 2.0]], "noise_variance": 0.5},
    ]
    out: list[dict] = []
    for case in sweep:
        cov = jax_core.laplace_covariance_from_hessian(
            jnp.array(case["hessian"], dtype=jnp.float64),
            jnp.float64(case["noise_variance"]),
        )
        out.append({"inputs": case, "outputs": {"covariance": _mat(cov)}})
    return out


def cases_caputo_flow() -> list[dict]:
    sweep = [
        # Single term [(1, 1)] → Markovian Banach (BLOCK_IN §v2.4 acceptance).
        {"chit_0": 1.0, "gamma_ab_0": 2.0,
         "lambda_chit": 0.3, "lambda_gamma": 0.4, "nu": 1.5,
         "prony_amplitudes": [1.0], "prony_decays": [1.0]},
        # Two-term
        {"chit_0": 1.0, "gamma_ab_0": 1.0,
         "lambda_chit": 0.3, "lambda_gamma": 0.4, "nu": 1.5,
         "prony_amplitudes": [0.4, 0.6], "prony_decays": [1.0, 2.5]},
        # Four-term (stresses pairwise-vs-sequential sum-order divergence)
        {"chit_0": 2.0, "gamma_ab_0": 1.5,
         "lambda_chit": 0.5, "lambda_gamma": 0.7, "nu": 2.0,
         "prony_amplitudes": [0.25, 0.25, 0.25, 0.25],
         "prony_decays": [0.5, 1.0, 2.0, 4.0]},
        # nu = 0 → all exponentials are 1; result = chit_0 * sum(amplitudes)
        {"chit_0": 1.0, "gamma_ab_0": 1.0,
         "lambda_chit": 0.3, "lambda_gamma": 0.4, "nu": 0.0,
         "prony_amplitudes": [0.5, 0.5], "prony_decays": [1.0, 2.0]},
    ]
    out: list[dict] = []
    for case in sweep:
        a, b = jax_core.caputo_flow(
            jnp.float64(case["chit_0"]), jnp.float64(case["gamma_ab_0"]),
            jnp.float64(case["lambda_chit"]), jnp.float64(case["lambda_gamma"]),
            jnp.float64(case["nu"]),
            jnp.array(case["prony_amplitudes"], dtype=jnp.float64),
            jnp.array(case["prony_decays"], dtype=jnp.float64),
        )
        out.append({"inputs": case, "outputs": {"chit": _f(a), "gamma_ab": _f(b)}})
    return out


def cases_mlp_forward() -> list[dict]:
    sweep = [
        # Single linear layer (output layer is always linear).
        {"x": [1.0, 1.0],
         "weights": [{"w": [[2.0, 0.0], [0.0, 3.0]], "b": [0.5, -0.5]}],
         "activation": "tanh"},
        # 2-layer tanh
        {"x": [0.5, 0.5],
         "weights": [
             {"w": [[1.0, 0.0], [0.0, 1.0]], "b": [0.0, 0.0]},
             {"w": [[1.0, 1.0]], "b": [0.0]},
         ],
         "activation": "tanh"},
        # 2-layer relu
        {"x": [-0.5, 1.0],
         "weights": [
             {"w": [[1.0, 1.0], [0.5, -0.5]], "b": [0.1, -0.1]},
             {"w": [[1.0, -1.0]], "b": [0.2]},
         ],
         "activation": "relu"},
        # 3-layer tanh, wider
        {"x": [0.3, -0.2, 0.7],
         "weights": [
             {"w": [[0.5, 0.1, -0.2], [0.1, 0.4, 0.3], [-0.1, 0.2, 0.1]],
              "b": [0.05, -0.05, 0.0]},
             {"w": [[0.7, 0.2, -0.1], [0.1, -0.3, 0.5]],
              "b": [0.0, 0.1]},
             {"w": [[1.0, -1.0]], "b": [0.0]},
         ],
         "activation": "tanh"},
    ]
    out: list[dict] = []
    for case in sweep:
        weights_jax = tuple(
            (jnp.array(layer["w"], dtype=jnp.float64),
             jnp.array(layer["b"], dtype=jnp.float64))
            for layer in case["weights"]
        )
        y = jax_core.mlp_forward(
            jnp.array(case["x"], dtype=jnp.float64),
            weights_jax,
            activation=case["activation"],
        )
        out.append({"inputs": case, "outputs": {"y": _row(y)}})
    return out


def cases_learned_field_substrate() -> list[dict]:
    sweep = [
        # Identity-passthrough on (chit, gamma + log_ratio).
        {"chit": 1.5, "gamma_ab": 2.5, "tau_obs": 4.0, "tau_obs_ref": 2.0,
         "weights": [{"w": [[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]],
                      "b": [0.0, 0.0]}],
         "activation": "tanh"},
        # Degenerate tau_obs → log_ratio clamped to 0.
        {"chit": 1.5, "gamma_ab": 2.5, "tau_obs": 0.0, "tau_obs_ref": 4.0,
         "weights": [{"w": [[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]],
                      "b": [0.0, 0.0]}],
         "activation": "tanh"},
        # 2-layer tanh
        {"chit": 0.3, "gamma_ab": 0.4, "tau_obs": 2.0, "tau_obs_ref": 1.0,
         "weights": [
             {"w": [[0.5, 0.1, -0.2], [0.1, 0.4, 0.3], [-0.1, 0.2, 0.5]],
              "b": [0.0, 0.0, 0.0]},
             {"w": [[1.0, 0.5, -0.3], [0.2, 1.0, 0.1]],
              "b": [0.05, -0.05]},
         ],
         "activation": "tanh"},
        # 2-layer relu with degenerate tau_obs_ref
        {"chit": 0.5, "gamma_ab": -0.3, "tau_obs": 4.0, "tau_obs_ref": 0.0,
         "weights": [
             {"w": [[1.0, 0.5, 0.5], [-0.5, 1.0, 0.5]],
              "b": [0.0, 0.0]},
             {"w": [[1.0, 0.0], [0.0, 1.0]],
              "b": [0.0, 0.0]},
         ],
         "activation": "relu"},
    ]
    out: list[dict] = []
    for case in sweep:
        weights_jax = tuple(
            (jnp.array(layer["w"], dtype=jnp.float64),
             jnp.array(layer["b"], dtype=jnp.float64))
            for layer in case["weights"]
        )
        a, b = jax_core.learned_field_substrate(
            jnp.float64(case["chit"]), jnp.float64(case["gamma_ab"]),
            jnp.float64(case["tau_obs"]), jnp.float64(case["tau_obs_ref"]),
            weights_jax,
            activation=case["activation"],
        )
        out.append({"inputs": case, "outputs": {"chit": _f(a), "gamma_ab": _f(b)}})
    return out


def cases_laplace_log_evidence() -> list[dict]:
    sweep = [
        {"residual_at_map": 0.0,
         "hessian": [[1.0, 0.0], [0.0, 1.0]],
         "noise_variance": 1.0, "n_obs": 4},
        {"residual_at_map": 0.5,
         "hessian": [[2.0, 0.5], [0.5, 3.0]],
         "noise_variance": 0.25, "n_obs": 8},
        {"residual_at_map": 1.5,
         "hessian": [[4.0, -1.0], [-1.0, 2.0]],
         "noise_variance": 0.5, "n_obs": 16},
    ]
    out: list[dict] = []
    for case in sweep:
        ev = jax_core.laplace_log_evidence(
            jnp.float64(case["residual_at_map"]),
            jnp.array(case["hessian"], dtype=jnp.float64),
            jnp.float64(case["noise_variance"]),
            case["n_obs"],
        )
        out.append({"inputs": case, "outputs": {"log_evidence": _f(ev)}})
    return out


# ---------------------------------------------------------------------------
# Session 4 — gfdr_model.py primitives
# ---------------------------------------------------------------------------


def cases_gfdr_alpha_s() -> list[dict]:
    sweep = [-1.5, -0.7, -0.3, -0.05, 0.0, 0.05, 0.3, 0.7, 1.5]
    return [
        {"inputs": {"chit": c}, "outputs": {"alpha_s": gfdr_model.alpha_s(c)}}
        for c in sweep
    ]


def cases_gfdr_plateau_height() -> list[dict]:
    sweep = [-2.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.5]
    return [
        {"inputs": {"chit": c},
         "outputs": {"plateau_height": gfdr_model.plateau_height(c)}}
        for c in sweep
    ]


def cases_gfdr_vertex_regime() -> list[dict]:
    # String-equality test (no ULP budget needed). Hits each branch.
    sweep = [-1.2, -0.7, -0.5, -0.2, 0.0, 0.2, 0.5, 0.7, 1.0]
    return [
        {"inputs": {"chit": c},
         "outputs": {"regime": gfdr_model.vertex_regime(c)}}
        for c in sweep
    ]


def cases_gfdr_generate_locus() -> list[dict]:
    # One chit per regime — covers all four locus branches. The full
    # 80-point locus is captured per case; that's the bulk of the new
    # fixture, but each point is small.
    sweep_chits = [1.0, 0.4, 0.0, -0.4, -1.0]
    out: list[dict] = []
    for c in sweep_chits:
        regime = gfdr_model.vertex_regime(c)
        locus = gfdr_model.generate_locus(c, regime)
        out.append({
            "inputs": {"chit": c, "regime": regime},
            "outputs": {
                "tau": [p["tau"] for p in locus],
                "chi": [p["chi"] for p in locus],
                "C": [p["C"] for p in locus],
            },
        })
    return out


def cases_gfdr_interp_locus() -> list[dict]:
    # Use one canonical locus and query at sub-grid taus including the
    # endpoint-clamp cases.
    base_chit = 0.0
    locus = gfdr_model.generate_locus(base_chit, gfdr_model.vertex_regime(base_chit))
    tau_queries = [0.001, 0.01, 0.1, 1.0, 10.0, 500.0, 1000.0, 5000.0]
    out: list[dict] = []
    for tau in tau_queries:
        r = gfdr_model.interp_locus(locus, tau)
        out.append({
            "inputs": {"chit": base_chit, "tau": tau},
            "outputs": {"C": r["C"], "chi": r["chi"]},
        })
    return out


def cases_gfdr_locus_residual() -> list[dict]:
    # Per session 2 lesson: do NOT seed `empirical` from
    # `gfdr_model.generate_locus(candidate)` — that creates a
    # self-residual that Python computes as exact 0 but Rust as ~1e-33
    # due to cross-impl libm cancellation. Synthetic invented rows
    # produce non-trivial residuals on both sides → no cancellation
    # collision, ULP tolerance covers the rest.
    empirical = [
        {"tau": 0.1, "C": 0.9, "chi": 0.05},
        {"tau": 1.0, "C": 0.7, "chi": 0.2},
        {"tau": 10.0, "C": 0.4, "chi": 0.5},
        {"tau": 100.0, "C": 0.2, "chi": 0.7},
        {"tau": 500.0, "C": 0.1, "chi": 0.8},
    ]
    candidate_chits = [-1.0, -0.5, -0.1, 0.0, 0.3, 0.5, 1.0]
    out: list[dict] = []
    for c in candidate_chits:
        out.append({
            "inputs": {"empirical": empirical, "candidate_chit": c},
            "outputs": {"residual": gfdr_model.locus_residual(empirical, c)},
        })
    return out


# ---------------------------------------------------------------------------
# Session 4 — sidecar.round_key (cross-language rounding sanity)
# ---------------------------------------------------------------------------


def cases_sidecar_round_key() -> list[dict]:
    # Mostly bulk-of-input cases where Python's banker's rounding and
    # Rust's `(x * 10^n).round_ties_even() / 10^n` agree. Includes a
    # mid-range halfway case where divergence could surface — the test
    # tolerates the documented cross-impl rounding caveat (1 ULP at the
    # rounded precision).
    # Infinity-key cases are excluded — `Infinity` is not valid JSON
    # (CPython emits the token by default; serde_json refuses to parse
    # it). Round-trip non-finite handling is covered by the Rust unit
    # test `sidecar::tests::round_decimal_passes_non_finite_through`.
    # Halfway-at-decimal-N cases where the input is not exactly
    # representable in f64 (e.g. 2.345_5 at decimals=3) are excluded:
    # Python's banker-rounding via dtoa and Rust's
    # `(x * 10^n).round_ties_even() / 10^n` disagree once the binary
    # multiplication shifts the value off the exact halfway. The
    # divergence is documented in `sidecar.rs`; the cases here are
    # exactly-representable (0.125, 0.375, ...) or far from any
    # halfway, where both impls agree bit-for-bit.
    sweep = [
        ((1.234_567_891_234, -2.345_678_912_345, 10.123_456_789), 6),
        ((0.0, 0.0, 0.0), 6),
        ((1e-9, -1e-9, 1.5e-9), 6),     # rounds to 0
        ((1.5, 2.5, -1.5), 0),          # banker's: 2, 2, -2
        ((0.125, 0.375, 0.625), 2),     # exactly representable halfways
    ]
    out: list[dict] = []
    for key, decimals in sweep:
        rounded = sidecar_round_key(key, decimals)
        out.append({
            "inputs": {
                "chit": key[0], "gamma_AB": key[1], "tau_obs": key[2],
                "decimals": decimals,
            },
            "outputs": {
                "chit": rounded[0],
                "gamma_AB": rounded[1],
                "tau_obs": rounded[2],
            },
        })
    return out


# ---------------------------------------------------------------------------
# Session 4 — flow() dispatch (banach_exponential / generic / Caputo)
# ---------------------------------------------------------------------------


def _make_tangent_flow_field(
    delta_chit: float,
    delta_gamma: float,
    tau_obs_ref: float,
    refinement: dict | None,
) -> TangentFlowField:
    origin = TranslationRule(
        operating_point=OperatingPoint(label="origin", gt="s", axes={}),
        xdot_choice="default",
        canonical=CanonicalPoint(
            chit=0.0, gamma_AB=0.0, k_frust=False, method="test", extras={},
        ),
    )
    return TangentFlowField(
        direction="forward",
        shape="tangent_flow",
        rule_at_origin=origin,
        scaling=ScalingRule(
            tau_obs_ref=tau_obs_ref,
            delta_chit=delta_chit,
            delta_gamma=delta_gamma,
            refinement=refinement,
        ),
    )


def cases_flow() -> list[dict]:
    """flow() across all three TangentFlowField dispatch branches.

    Each case carries the full field-construction inputs so Rust can
    rebuild an identical `TangentFlowField` and run `flow::flow()`.
    """
    cases_in: list[dict] = [
        # Generic tangent-flow (no refinement, no banach_exponential):
        # routes through `tangent_flow_canonical`.
        {
            "delta_chit": 0.3, "delta_gamma": 0.5,
            "tau_obs_ref": 1.0, "refinement": None,
            "chit_0": 1.5, "gamma_AB_0": 2.5, "nu": 2.0,
        },
        # Generic at nu=tau_ref: identity in math (log(1)=0, 1^x=1).
        {
            "delta_chit": 0.3, "delta_gamma": 0.5,
            "tau_obs_ref": 1.0, "refinement": None,
            "chit_0": 1.5, "gamma_AB_0": 2.5, "nu": 1.0,
        },
        # Banach exponential branch.
        {
            "delta_chit": 0.0, "delta_gamma": 0.0,
            "tau_obs_ref": 1.0,
            "refinement": {
                "flow_kind": "banach_exponential",
                "lambda_chit": 0.3, "lambda_gamma": 0.4,
            },
            "chit_0": 2.0, "gamma_AB_0": 3.0, "nu": 1.5,
        },
        # Caputo branch (beta_mem < 1) with multi-term prony.
        {
            "delta_chit": 0.0, "delta_gamma": 0.0,
            "tau_obs_ref": 1.0,
            "refinement": {
                "beta_mem": 0.7,
                "lambda_chit": 0.3, "lambda_gamma": 0.4,
                "prony_terms": [[0.4, 1.0], [0.6, 2.5]],
            },
            "chit_0": 1.0, "gamma_AB_0": 1.0, "nu": 1.5,
        },
        # Caputo single-term reduces to Banach exp (sanity case).
        {
            "delta_chit": 0.0, "delta_gamma": 0.0,
            "tau_obs_ref": 1.0,
            "refinement": {
                "beta_mem": 0.999,
                "lambda_chit": 0.3, "lambda_gamma": 0.4,
                "prony_terms": [[1.0, 1.0]],
            },
            "chit_0": 1.5, "gamma_AB_0": 2.5, "nu": 1.0,
        },
    ]
    out: list[dict] = []
    for case in cases_in:
        field = _make_tangent_flow_field(
            case["delta_chit"], case["delta_gamma"],
            case["tau_obs_ref"], case["refinement"],
        )
        initial = CanonicalState(
            chit=case["chit_0"], gamma_AB=case["gamma_AB_0"], k_frust=False,
        )
        result = flow_op(initial, case["nu"], field)
        out.append({
            "inputs": case,
            "outputs": {"chit": result.chit, "gamma_AB": result.gamma_AB},
        })
    return out


# ---------------------------------------------------------------------------
# Session 6 — intent algebra cross-language JSON parity
# ---------------------------------------------------------------------------
#
# Not bit-identity in the math-primitive sense (intent_map is pure
# arithmetic dispatch, no libm); this is a *schema parity* check. Python
# is the producer of sacrifice records in the wrapped-variant path
# (session 7); Rust deserializes Python's sac-dict JSON into
# `SacrificeRecord` and asserts field-by-field agreement.
#
# Asymmetric parity by design: Python's sac dict contains a
# `preserved_invariant` STRING key (stored). Rust's `SacrificeRecord`
# makes that field a derived method (`.preserved_invariant()`) — the
# string is statically determined by the `IntentDiagnostics` variant
# per the BLOCK_IN-prep rationale. serde silently drops the unknown
# field on Python→Rust read. Rust→Python round-trip would lose the
# key; if a future consumer ever needs the symmetric path, add a
# custom serializer on SacrificeRecord (the BLOCK_IN note flags this
# option).


def _sacrifice_case(
    chit: float, gamma_AB: float, k_frust: bool,
    chit_range: tuple[float, float], gamma_AB_range: tuple[float, float],
    intent_id: str,
    notes: str,
) -> dict:
    state = CanonicalState(chit=chit, gamma_AB=gamma_AB, k_frust=k_frust)
    gamut = GamutSpec(chit_range=chit_range, gamma_AB_range=gamma_AB_range)
    mapped, sac = intent_map_op(state, 1.0, gamut, intent_id)
    # Normalize the sac dict for JSON: tuples → lists.
    sac_json = dict(sac)
    if "out_of_gamut_axes" in sac_json:
        sac_json["out_of_gamut_axes"] = list(sac_json["out_of_gamut_axes"])
    return {
        "inputs": {
            "chit": chit, "gamma_AB": gamma_AB, "k_frust": k_frust,
            "tau_obs": 1.0,
            "gamut": {
                "chit_range": list(chit_range),
                "gamma_AB_range": list(gamma_AB_range),
            },
            "intent_id": intent_id,
            "notes": notes,
        },
        "outputs": {
            "mapped": {
                "chit": mapped.chit,
                "gamma_AB": mapped.gamma_AB,
                "k_frust": mapped.k_frust,
            },
            "sacrifice": sac_json,
        },
    }


def cases_sacrifice_record() -> list[dict]:
    return [
        # I1 — regime ∧ sign(γ) ∧ k_frust
        _sacrifice_case(2.0, 0.3, False, (-1.0, 1.0), (-1.0, 1.0),
                        "I1", "deep_c preserved on unit gamut"),
        _sacrifice_case(2.0, 0.0, False, (-0.5, 0.5), (-1.0, 1.0),
                        "I1", "regime unreachable — invariant breaks"),
        _sacrifice_case(0.0, 0.5, False, (-1.0, 1.0), (-1.0, -0.1),
                        "I1", "gamma sign flip"),
        _sacrifice_case(2.0, 0.0, True, (-1.0, 1.0), (-1.0, 1.0),
                        "I1", "k_frust propagates"),
        # I2 — drive-faithful
        _sacrifice_case(0.3, -0.2, False, (-1.0, 1.0), (-1.0, 1.0),
                        "I2", "in-gamut passthrough"),
        _sacrifice_case(2.0, 0.0, False, (-1.0, 1.0), (-1.0, 1.0),
                        "I2", "chit out-of-gamut"),
        _sacrifice_case(2.0, 2.0, False, (-1.0, 1.0), (-1.0, 1.0),
                        "I2", "both axes out-of-gamut"),
        # I3 — capacity_class ∧ k_frust
        _sacrifice_case(0.9, 0.0, False, (-1.0, 1.0), (-1.0, 1.0),
                        "I3", "deep preserved"),
        _sacrifice_case(0.9, 0.0, False, (-0.5, 0.5), (-1.0, 1.0),
                        "I3", "deep demoted to shallow"),
        # I4 — sign(γ_AB)
        _sacrifice_case(0.0, 2.0, False, (-1.0, 1.0), (-1.0, 1.0),
                        "I4", "positive gamma kept positive"),
        _sacrifice_case(0.0, 0.5, False, (-1.0, 1.0), (-1.0, -0.1),
                        "I4", "sign flip flagged"),
        # I5 — regime label
        _sacrifice_case(0.5, 0.0, False, (-1.0, 1.0), (-1.0, 1.0),
                        "I5", "regime preserved (c_near_s)"),
        _sacrifice_case(2.0, 0.0, False, (-0.5, 0.5), (-1.0, 1.0),
                        "I5", "regime broken (deep_c → c_near_s)"),
    ]


# ---------------------------------------------------------------------------
# Session 7 — provenance / wrapped-output parity (schema, not bit-id)
# ---------------------------------------------------------------------------
#
# `provenance_hash` is the load-bearing cross-language hash: the four
# (solver_version, operation, dispatch_path, table_version) inputs must
# produce the same 4-byte blake2b digest in Python and Rust. The fixture
# emits the hash and Rust replays the input + asserts byte-equality.
#
# `operation_output_regime_at` is the wrapped-variant wire-format parity
# (matches the session-6 sacrifice_record approach). Timestamps and note
# strings are excluded from the fixture — see the session-7 note in
# CLAUDE.md / BLOCK_IN about the documented diagnostic-string asymmetry.


def cases_provenance_hash() -> list[dict]:
    """Hash inputs that cross every Python branch of the encoding."""
    rows = [
        ("apply_translation", DispatchPath.DIRECT_COMPUTE, None),
        ("apply_translation", DispatchPath.TABLE_HIT, "banach-1.0.0"),
        ("apply_translation", DispatchPath.COMPUTE_FALLBACK, "banach-1.0.0"),
        ("forward_sweep_invert", DispatchPath.DIRECT_COMPUTE, None),
        ("forward_sweep_invert", DispatchPath.TABLE_HIT, "sidecar-v2-A"),
        ("tau_obs_sweep", DispatchPath.DIRECT_COMPUTE, None),
        ("regime_at", DispatchPath.DIRECT_COMPUTE, None),
        ("gamut_classify", DispatchPath.DIRECT_COMPUTE, None),
        ("intent_map", DispatchPath.DIRECT_COMPUTE, None),
        ("intent_compose", DispatchPath.DIRECT_COMPUTE, None),
        ("validate_driver_profile", DispatchPath.DIRECT_COMPUTE, None),
        # Cross-table-version case so Rust-side tests catch the
        # version-string contribution to the hash.
        ("apply_translation", DispatchPath.TABLE_HIT, "banach-2.0.0"),
    ]
    out = []
    for op, dp, tv in rows:
        prov = make_provenance(op, dispatch_path=dp, table_version=tv)
        out.append({
            "inputs": {
                "operation": op,
                "dispatch_path": dp.value,
                "table_version": tv,
            },
            "output": provenance_hash(prov),
        })
    return out


def cases_operation_output_regime_at() -> list[dict]:
    """End-to-end wrapped-variant JSON parity for `regime_at_wrapped`.

    Chosen because `regime_at` is the simplest wrapped op (no sidecar,
    no round-trip, no per-cell aggregation) — it exercises the
    `OperationOutput<T>` wire format with the least incidental surface.
    Timestamps and note strings are excluded from the emitted JSON; the
    Rust test compares the structured fields. The float-formatting
    divergence between Python `f"{0.0}"` ("0.0") and Rust `format!("{}", 0.0)`
    ("0") is the documented reason notes don't ride on this fixture.
    """
    out = []
    for (chit, gamma_AB, k_frust, tau_obs, label) in [
        (0.5, -0.2, False, 1.0, "c_near_s clean"),
        (0.95, 0.1, True, 1.0, "deep_c with k_frust"),
        (-0.4, 0.3, False, 1.0, "s_critical region"),
        (-0.8, 0.0, False, 1.0, "deep_r"),
    ]:
        cs = CanonicalState(chit=chit, gamma_AB=gamma_AB, k_frust=k_frust)
        op_out = regime_at_wrapped_op(cs, tau_obs)
        out.append({
            "label": label,
            "inputs": {
                "chit": chit,
                "gamma_AB": gamma_AB,
                "k_frust": k_frust,
                "tau_obs": tau_obs,
            },
            "output": {
                "value": {
                    "regime": op_out.value.regime,
                    "k_frust": op_out.value.k_frust,
                },
                "validation": {
                    "asymptotic_closure_compliant":
                        op_out.validation.asymptotic_closure_compliant,
                    "k_frust_invariant": op_out.validation.k_frust_invariant,
                    "round_trip_residual": op_out.validation.round_trip_residual,
                },
                "provenance": {
                    "solver_version": op_out.provenance.solver_version,
                    "operation": op_out.provenance.operation,
                    "dispatch_path": op_out.provenance.dispatch_path.value,
                    "table_version": op_out.provenance.table_version,
                },
            },
        })
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


PRIMITIVES = {
    "tangent_flow_substrate": cases_tangent_flow_substrate,
    "banach_state": cases_banach_state,
    "tangent_flow_canonical": cases_tangent_flow_canonical,
    "lookup_squared_distance": cases_lookup_squared_distance,
    "tangent_flow_canonical_inverse": cases_tangent_flow_canonical_inverse,
    "tangent_flow_inversion_residual": cases_tangent_flow_inversion_residual,
    "laplace_covariance_from_jacobian": cases_laplace_covariance_from_jacobian,
    "laplace_covariance_from_hessian": cases_laplace_covariance_from_hessian,
    "caputo_flow": cases_caputo_flow,
    "mlp_forward": cases_mlp_forward,
    "learned_field_substrate": cases_learned_field_substrate,
    "laplace_log_evidence": cases_laplace_log_evidence,
    # session 4 — gfdr_model.rs + sidecar.rs + flow.rs
    "gfdr_alpha_s": cases_gfdr_alpha_s,
    "gfdr_plateau_height": cases_gfdr_plateau_height,
    "gfdr_vertex_regime": cases_gfdr_vertex_regime,
    "gfdr_generate_locus": cases_gfdr_generate_locus,
    "gfdr_interp_locus": cases_gfdr_interp_locus,
    "gfdr_locus_residual": cases_gfdr_locus_residual,
    "sidecar_round_key": cases_sidecar_round_key,
    "flow": cases_flow,
    # session 6 — operations.py intent algebra (schema parity, not bit-id)
    "sacrifice_record": cases_sacrifice_record,
    # session 7 — provenance + wrapped-variant wire parity
    "provenance_hash": cases_provenance_hash,
    "operation_output_regime_at": cases_operation_output_regime_at,
}


def main() -> None:
    fixture = {name: gen() for name, gen in PRIMITIVES.items()}
    out_path = Path(__file__).parent / "jax_core_reference.json"
    out_path.write_text(json.dumps(fixture, indent=2) + "\n")
    n_cases = sum(len(v) for v in fixture.values())
    size = out_path.stat().st_size
    print(f"wrote {out_path} -- {len(fixture)} primitives, {n_cases} cases, {size} bytes")


if __name__ == "__main__":
    main()
