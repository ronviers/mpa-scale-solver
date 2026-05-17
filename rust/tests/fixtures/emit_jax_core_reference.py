"""Emit the bit-identity reference fixture for the Rust math.rs port.

Runs the Python `mpa_scale_solver.jax_core` primitives over a small input
sweep and writes the input + output pairs to `jax_core_reference.json`
next to this file. The Rust integration test at
`rust/tests/bit_identity.rs` consumes that fixture and asserts each
Rust primitive in `src/math.rs` reproduces the Python output within a
per-primitive ULP budget (BLOCK_IN §v6 "byte-identical for deterministic
ops" check, scoped to math.rs).

Regeneration discipline: the fixture is committed. If `jax_core` math
changes, rerun this script and commit the JSON diff in the same change.
The Rust test will catch unintentional divergence.

Run with the repo's Python (any cwd is fine — the package is editable-
installed; if not, `pip install -e H:/mpa-scale-solver` once):

    python rust/tests/fixtures/emit_jax_core_reference.py
"""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp

from mpa_scale_solver import jax_core


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
