# mpa-scale-solver

MPA scale-management kernel: τ_obs projection and canonical-frame
operations. Sibling to `mpa-solver` (forward physics + observable
extraction). Consumed by `mpa-conform` (orchestration → signed
`declaration_bundle.json`). Read by `mpa-auditor` (display).

The Python implementation is the production artifact through v5
(shipped 2026-05-16). A native (Rust / C++ + WASM + Python bindings)
port lands at v6, matching the Python under the per-seed
reproducibility discipline.

## The seven operations (surface stable across v0–v6)

| Operation | Role |
|---|---|
| `apply_translation` | canonical state → substrate-native at τ_obs (dispatches on `lookup_table` / `tangent_flow`) |
| `forward_sweep_invert` | substrate observation → canonical state at τ_obs (brute-force grid; sidecar fast-path via wrapped variant) |
| `tau_obs_sweep` | per-frame fan-out across a τ_obs grid |
| `regime_at` | five-bucket vertex regime classifier |
| `gamut_classify` | in-gamut / out-of-gamut diagnosis |
| `intent_map` | All five intents (I1–I5) + composition (`intent_compose`) per RFC-S §3 |
| `validate_driver_profile` | RFC-S §5 round-trip residuals + per-intent metric dispatch (v3) |

Plus the v1 addition `flow(canonical_initial, nu, field)` — continuous-form
`C^nu = exp(nu * ln C)`. Markovian by default (`beta_mem = 1`); v2.4 adds
non-Markovian Caputo flow via curator-supplied Prony approximation of the
Mittag-Leffler kernel when `beta_mem < 1`. See
[`docs/CONTINUOUS_FLOW.md`](docs/CONTINUOUS_FLOW.md).

Each of the seven has a `*_wrapped` variant returning
`OperationOutput[T]` with a `ValidationReport` and `Provenance`
alongside the value (handoff §A.2 / §C.5 / §C.6). v0 sigs are unchanged;
v1 consumers that want validation + provenance call the wrapped
variants.

## v2 differentiable surface

v2.0 adds a JAX-backed differentiable surface alongside the v0/v1
operations (which keep their `math.*` / numpy implementations and
fixture byte-identity). See `mpa_scale_solver.jax_core` for the pure
math primitives and `mpa_scale_solver.jax_ops` for the consumer surface:

- `tangent_flow_substrate_diff(canonical, field, tau_obs)` — JAX-array
  forward map. Differentiable in canonical coordinates and field
  parameters.
- `flow_diff(canonical_initial, nu, field)` — continuous-form flow on
  tangent-flow fields. Banach exponential and generic tangent-flow
  branches mirror `flow.flow()` dispatch.
- `tangent_flow_forward_jacobian(canonical, field, tau_obs)` — 2x2
  Jacobian via `jax.jacfwd`; the substrate sensitivity surface v2.1's
  Laplace approximation and v5's `sensitivity_backprop` will compose
  against.
- `forward_sweep_invert_diff(target, field, tau_obs)` — exact
  closed-form inverse of the tangent-flow forward map (monotonic and
  analytically invertible at `tau_obs > 0`); routes through
  `jax_core.tangent_flow_canonical_inverse` so gradients flow.
- `banach_state_diff(substrate, nu)` — differentiable Banach
  analytical canonical state.

v2.1 adds Bayesian inversion on top of the v2.0 foundation:

- `forward_sweep_invert_posterior(target, field, tau_obs, ...)` —
  returns a `Posterior` (mean, covariance, log-evidence) via Laplace
  approximation around the MAP. Tangent-flow uses the closed-form fast
  path (`covariance = noise_variance * inv(J^T J)`); lookup-table uses
  a softmax-weighted moment fit over the top-k lowest-residual
  candidates.
- `forward_sweep_invert_posterior_wrapped(...)` — `OperationOutput[Posterior]`
  with validation + provenance, following the established `*_wrapped`
  pattern.

`CanonicalState` is registered as a JAX PyTree (leaves: `(chit,
gamma_AB)`; aux: `k_frust`) so `jax.grad` / `jax.jacobian` /
`jax.hessian` work directly on CanonicalState-typed callbacks. Float64
is enabled at `jax_core` import.

The v0/v1 `apply_translation`, `flow`, `forward_sweep_invert`, and
`tau_obs_sweep` (and their `*_wrapped` variants) are unchanged. The
JAX surface is parallel and opt-in — consumers that don't need
gradients call the v0/v1 surface as before.

## v3 additions

v3 lands four new capability slices alongside the seven-operation API
(which stays stable per CLAUDE.md):

- **Cross-substrate ops** (`cross_substrate.py`) —
  `gamut_overlap(gamut_a, gamut_b)`, `canonical_distance(a, b, metric)`
  (`l2` / `l1` / `regime` / `universality`), and
  `universality_agreement(field_a, gamut_a, field_b, gamut_b,
  canonical_grid, tau_obs)` — the framework's primary cross-substrate
  test (s→r migration per cdv1 §gFDR signatures) as a direct call.
  Each has a `*_wrapped` variant with validation + provenance.
- **Active learning** (`active_learning.py`) —
  `suggest_measurements(field, gamut, canonical_grid, tau_obs, n)`
  returns `MeasurementCandidate`s ranked by a composite score: posterior
  covariance-trace (v2.1 surface) + gamut-edge proximity + intent-
  invariance fragility (v2.3 surface). Weights are configurable.
- **MCP server** (`mcp_server.py`) — the seven core operations plus the
  three cross-substrate ops plus `suggest_measurements` exposed as 11
  MCP tools over stdio. JSON-shape I/O. Read-only. Console script
  `mpa-scale-solver-mcp` or `python -m mpa_scale_solver.mcp_server`.
- **Learned translation field** (`LearnedField`) — third field shape
  alongside `lookup_table` and `tangent_flow`. Small JAX MLP
  `(chit, gamma_AB, log(tau/tau_ref))` → `(substrate_chit,
  substrate_gamma_AB)`. Curators train (mpa-conform); the solver
  evaluates. `apply_translation` dispatches on `shape == "learned"`.

v3 also tightens `validate_driver_profile` to the per-intent metrics
spelled out in RFC-S §5 (Hamming for I1, L²-on-drive for I2, ‖Γ*‖
deviation for I3, ε-sequence-distance for I4, universality-class
agreement for I5). v2.3's regime-agreement-rate / forward-residual /
round-trip-residual keys are preserved for back-compat; the per-intent
aggregate rides as `summary["per_intent"]`.

## v5 additions

v5 ships three capability slices alongside the unchanged seven-operation
API:

- **Continuous Banach self-test cadence** (`self_test.py`) —
  `SelfTestCadence(k=100)` runs a `BanachDriftReport` on every k-th
  tick against the analytical Banach truth (drift tolerance
  `1e-10`). Drift is reported, not raised. `forward_sweep_invert_stream`
  takes `self_test_cadence` + `self_test_callback` kwargs that advance
  per emitted frame — state-local, no feedback into the primary
  inversion.
- **Sensitivity backprop** (`sensitivity.py`) —
  `trajectory_substrate_diff(canonical, field, tau_obs_grid)` gives a
  per-frame substrate trajectory as a JAX array;
  `trajectory_substrate_jacobian` returns per-frame
  `∂substrate / ∂canonical` (shape `(T, 2, 2)`);
  `field_parameter_sensitivity` returns per-frame
  `∂substrate / ∂(delta_chit, delta_gamma, tau_obs_ref)` for
  tangent-flow; `inversion_sensitivity` returns
  `∂canonical / ∂substrate` via the closed-form analytical inverse.
  The one-liner `driver_profile_loss_grad(loss_fn, canonical, field,
  tau_obs_grid, observed)` returns the gradient of any user-supplied
  loss w.r.t. the field's hyperparameters — composes through
  `jax.value_and_grad` on the trajectory forward map.
- **Gradient-based inversion** — `forward_sweep_invert` gains
  `method: Literal["auto", "grid", "gradient"] = "auto"`. The new
  `"auto"` default routes `TangentFlowField` to the closed-form
  analytical inverse (sub-grid-resolution; exact at float64),
  `LearnedField` to L-BFGS (scipy's L-BFGS-B + `jax.grad`-provided
  gradients, warm-started from the grid argmin), and lookup-table
  fields to the v0–v4 grid path. `method="grid"` preserves byte-
  identical v4 behavior. `method="gradient"` works on differentiable
  shapes only. The wrapped variant accepts the same kwarg.

## Install

```
pip install -e .
```

Python 3.10+. Hard deps are `numpy`, `jax`, `jaxlib` (CPU JAX is
sufficient for correctness; GPU is optional, available only on Linux/
WSL — see CLAUDE.md). Tests additionally need `pytest` and optionally
`matplotlib` (used by the camera test for the visual plot; the test
still passes without it).

## A minimal example

```python
import json
import numpy as np
from mpa_scale_solver import (
    CanonicalState, apply_translation, forward_sweep_invert,
    parse_translation_field,
)

# Load a seed driver profile from mpa-conform's curator output.
with open("mpa-conform/output/seed-corpus/neural-population/driver-profile.json") as f:
    profile = json.load(f)
field = parse_translation_field(profile["translation_field"])

# Forward project a canonical state.
state = CanonicalState(chit=1.0, gamma_AB=-0.5)
substrate = apply_translation(state, field, tau_obs=1.0)
print(substrate.label)        # "committed"
print(substrate.axes["gt"])   # "c"

# Invert: substrate observation → canonical recovery.
grid = np.array([[c, g] for c in np.linspace(-1, 1, 11) for g in [-0.5, 0.0, 0.5]])
recovered, residual = forward_sweep_invert(substrate, field, 1.0, grid)
```

## The camera tests

Two camera tests now live alongside each other:

- **`tests/test_banach_camera.py`** — the v1 acceptance test. The Banach
  substrate's `state_at(nu)` is the framework analytical truth (closed-form
  exponential decay per Q1 of the v1 build session); `forward_sweep_invert`
  recovers canonical state per frame and we score against the truth.
  Pass criterion: `max |residual| <= 0.001` per axis. See
  [`docs/BANACH_SUBSTRATE.md`](docs/BANACH_SUBSTRATE.md).
- **`tests/test_camera_migration.py`** — the v0 legacy test, kept passing
  as back-compat coverage of the lookup-table dispatch path. Synthetic
  aging_log fixture, tolerance ≤ 0.05.

```
pytest tests/test_banach_camera.py
pytest tests/test_camera_migration.py
```

The v0 test outputs `tests/out/migration_compare.png` (numerical curve
overlaid on analytical) and `tests/out/result.json`.

## How this composes

- `mpa-solver` (sibling) does trajectory integration and observable
  extraction (`fit_invariants` → `{X_c, X_r, alpha_s, P_s, N_f, regime}`).
- `mpa-scale-solver` (this repo) does τ_obs projection and canonical-frame
  operations on the resulting observables.
- `mpa-conform` (parent of this repo's testing seed corpus) orchestrates:
  declare → call mpa-solver → call mpa-scale-solver → assemble bundle →
  sign.
- `mpa-auditor` displays the bundle.

See [`docs/ORDER_OF_OPERATIONS.md`](docs/ORDER_OF_OPERATIONS.md) for the
five-step skeleton and the three named inner traversals.
See [`docs/PREREQUISITES.md`](docs/PREREQUISITES.md) for the upstream
binding gap (mpa-solver's `fit_invariants` Python binding).
See [`docs/EXR_CHANNEL_MANIFEST.md`](docs/EXR_CHANNEL_MANIFEST.md) for the
per-frame EXR channels mpa-conform assembles using outputs from this repo.

## What this repo is not

`mpa-scale-solver` does not:

- Extract observables from raw substrate signals. That's `mpa-solver`.
- Orchestrate bundles or sign declarations. That's `mpa-conform`.
- Render audits. That's `mpa-auditor`.
- Produce driver profiles. That's `mpa-conform`'s curator / researcher path.
- Carry RG-flow physics defaults. The trivial-baseline default holds at v0;
  non-trivial flow content arrives via driver-profile lookup tables.

## Session log

| Date | Session | Outcome |
|---|---|---|
| 2026-05-15 | Python v0.1.0 build | Seven operations shipped; gfdr_model.js ported (5-bucket); camera test passes max\|residual\| = 0.012 vs tolerance 0.05; all three seed-corpus profiles pass round-trip closure. |
| 2026-05-16 | Python v1.0.0 build | Tangent-flow translation field + continuous `flow()` + Banach calibration substrate + inverse-lookup-table sidecar dispatch + per-call self-validation + full provenance trail. Seven wrapped variants (`*_wrapped`) returning `OperationOutput[T]`. Banach camera test passes max\|residual\| < 0.001. All v0 fixtures pass unchanged. |
| 2026-05-16 | Python v2.0.0 build (BLOCK_IN §v2 cut (a)) | JAX foundation + differentiability. New modules: `jax_core` (pure JAX math primitives — tangent-flow forward/inverse, Banach analytical state, lookup-table squared distance, inversion residual), `jax_ops` (consumer surface: `*_diff` entry points returning JAX arrays, exact closed-form `forward_sweep_invert_diff` for tangent-flow, `tangent_flow_forward_jacobian`, `banach_state_diff`), `jax_pytree` (CanonicalState as JAX PyTree). Float64 enabled. v0/v1 surfaces unchanged; all v0/v1 fixtures pass unchanged. JAX added as hard dep; Windows CPU + WSL GPU both work (Windows CPU is the dev-time canonical environment). Cuts (b)–(e) remain: Bayesian inversion, N-mode, I1–I4 intents, non-Markovian Caputo flow. |
| 2026-05-16 | Python v2.1.0 build (BLOCK_IN §v2 cut (b)) | Bayesian inversion via Laplace approximation. New `Posterior` dataclass + `forward_sweep_invert_posterior` / `_wrapped` (separate function rather than `posterior=True` kwarg — cleaner return-type contract; sanctioned at session time). Tangent-flow fast path: MAP exact, covariance = `noise_variance * inv(J^T J)`, finite log-evidence. Lookup-table path: weighted-moment fit over top-k lowest-residual candidates. New `jax_core.laplace_covariance_from_jacobian` / `laplace_covariance_from_hessian` / `laplace_log_evidence` primitives. All prior tests still green plus 14 new bayesian-inversion tests. Cuts (c)–(e) remain. |
| 2026-05-16 | Python v2.3.0 build (BLOCK_IN §v2 cut (d)) | All five intents implemented: I1 regime-preserving (regime ∧ sign(γ) ∧ k_frust), I2 drive-faithful (no-adjust + completeness sacrifice), I3 capacity-preserving (deep/shallow capacity class ∧ k_frust), I4 persistence-preserving (sign(γ_AB) contraction-ordering proxy), I5 signature-preserving (5-bucket regime, v0/v1 contract preserved verbatim). New `intent_compose` + `intent_compose_wrapped` enforce RFC-S §3 composition algebra: I2 rejected in chains, other intents sequence freely with per-intent sacrifices in the trace. `validate_driver_profile` accepts all five intent ids (5-bucket agreement metric; per-intent metric tightening deferred to v3 alongside cross-substrate ops). 28 new tests in `test_intents.py`; 202 tests total green. §v2.2 (N-mode) remains cancelled. Cut (e) Caputo flow remains. |
| 2026-05-16 | Python v2.4.0 build (BLOCK_IN §v2 cut (e)) | Non-Markovian Caputo flow via Prony sum-of-exponentials approximation of the Mittag-Leffler kernel. New `jax_core.caputo_flow` primitive (differentiable in all parameters); `flow()` and `flow_diff()` dispatch on `refinement['beta_mem']` (β < 1 → Caputo; β = 1 → v1 Markovian unchanged). `ScalingRule.refinement` accepts `beta_mem: float` and `prony_terms: list[tuple[float, float]]` (curator-supplied amplitude / decay pairs; on-the-fly Mittag-Leffler fitting is mpa-conform's curator-path job). 10 new tests in `test_caputo_flow.py`: β=1 byte-identity vs v1 Markovian, per-axis lambda independence, jax/python parity, jax.grad finite-difference, jit compile, error path. 212 tests total green. **v2 complete** — all cuts (a)–(e) shipped except cancelled (c). |
| 2026-05-16 | Python v3.0.0 build (BLOCK_IN §v3) | Five capabilities in one bundled cut. (1) **Cross-substrate ops** — `gamut_overlap`, `canonical_distance` (l2/l1/regime/universality metrics), `universality_agreement` (the framework's primary s→r migration test per cdv1 §gFDR) live in `cross_substrate.py` with `*_wrapped` variants. The seven-operation API is unchanged; these are cross-substrate compositions per CLAUDE.md. (2) **Active learning** — `suggest_measurements(field, gamut, grid, tau_obs, n)` returns ranked `MeasurementCandidate`s scored by composite (posterior covariance-trace + gamut-edge proximity + intent-invariance fragility); weights configurable. Builds on v2.1 `Posterior` + v2.3 intent algebra. (3) **Per-intent RFC-S §5 metrics** — `validate_driver_profile` now dispatches per intent: I1 Hamming-on-regime + edge-type, I2 L²-on-drive + max-γ, I3 ‖Γ*‖-deviation + capacity-class, I4 ε-sequence-distance + survival, I5 universality-class + intra-class-L². v2.3 back-compat keys preserved. (4) **MCP server** — 11 tools (7 core + 3 cross-substrate + 1 active-learning) exposed via stdio in `mcp_server.py`; thin (one dispatch function per tool, hardcoded JSON schemas); console script `mpa-scale-solver-mcp`. `mcp>=1.0` added as a hard dep. (5) **LearnedField** — third translation-field shape; small JAX MLP `(chit, gamma_AB, log(tau/tau_ref))` → `(substrate_chit, substrate_gamma_AB)`; `jax_core.mlp_forward` + `learned_field_substrate` primitives; `jax_ops.learned_field_substrate_diff` consumer entry; `apply_translation` dispatches on `shape == "learned"`; `parse_translation_field` parses the `learned` JSON shape (forward-compat under driver-profile schema's `additionalProperties` until mpa-atlas bumps the schema). 74 new tests: `test_cross_substrate.py` (34), `test_active_learning.py` (11), `test_mcp_server.py` (17), `test_learned_field.py` (12). 286 tests total green; all v0/v1/v2 fixtures pass unchanged. |
| 2026-05-16 | Python v4.0.0 build (BLOCK_IN §v4) | Three capability slices in one bundled cut. (1) **Streaming inversion** — `streaming.py` carries `InversionResult` + `forward_sweep_invert_stream(observations, field, canonical_grid, *, tau_obs=None, score_fn=None)`, a generator yielding `InversionResult` per consumed observation. State-local per frame (no caching, smoothing, or cross-frame state). `tau_obs=None` falls back to each `obs.tau_obs` for streams where the observer scale varies frame-to-frame. Thin source adapters: `from_iterable` (passthrough) and `from_stdin` (JSON-per-line); WebSocket / polling deferred per thin-discipline — any iterable fits the same interface. Wrapped streaming variant not exposed — consumers wanting per-frame `OperationOutput` call `forward_sweep_invert_wrapped` in their loop. Streaming MCP tool variant deferred per BLOCK_IN §v4 open-watch (the v3 stdio call-per-request shape covers the common agentic workflow). (2) **Symbolic-query DSL** — `symbolic_query.py` carries `query()` + `QueryResult` + `QueryParseError`. Five patterns max (the BLOCK_IN cap): `regime at chit=A gamma=B [tau=T]`, `gamut at chit=A gamma=B [tau=T]`, `translate chit=A gamma=B at tau=T`, `invert chit_obs=X gamma_obs=Y at tau=T` (tangent_flow only — closed-form algebraic inverse exists; lookup_table / learned consumers call `forward_sweep_invert` directly), and `tau where regime crosses B for chit=A gamma=B` (closed-form for tangent_flow with `delta_chit != 0`; bisection over `tau_range=(lo, hi)` kwarg for other shapes). Structural context (field, gamut, grid, tau_range) rides as kwargs; the DSL parses intent and literal parameters only — Mathematica-style. Closed-form expressions returned alongside numerical evaluations where both exist. Read-only (no operation mutates the field). (3) **Notebook ergonomics + default plot hooks** — `_repr_html_` methods on 11 user-facing dataclasses (CanonicalState, SubstrateState, GamutSpec, RegimeReading, TranslationField, TangentFlowField, LearnedField, Posterior, OperationOutput, MeasurementCandidate, InversionResult) render compact HTML tables in Jupyter via a small `_html_table` helper at the top of `types.py`. `__repr__` overridden on `Posterior` (was dumping nested covariance tuple), `LearnedField` (was dumping entire weight tuple), and `OperationOutput` (was inlining nested ValidationReport + Provenance dataclasses) — defaults preserved on every other dataclass for back-compat. `plotting.py` ships four default plot helpers (`plot_trajectory`, `plot_gamut`, `plot_residual_field`, `plot_posterior`) covering the north-star §Visualization-first list, each accepting `backend="matplotlib"` (default) or `"plotly"`. Both backends import lazily; matplotlib stays in test-extras, plotly is fully optional (no project dep). 52 new tests: `test_streaming.py` (14), `test_symbolic_query.py` (13), `test_notebook_repr.py` (25). 338 tests total green; all v0/v1/v2/v3 fixtures pass unchanged. |
| 2026-05-16 | Python v5.0.0 build (BLOCK_IN §v5) | Three capability slices in one bundled cut; the last functional release before the v6 native port. (1) **Continuous Banach self-test cadence** — new `self_test.py` carries `BanachDriftReport` (frozen flat dataclass — call_index, sample_count, drift floats, asymptotic / k_frust flags, timestamp_ns, solver_version, notes tuple), `run_banach_self_test(*, substrate=None, nu_samples=None, call_index=0)` (synchronous, compares `jax_ops.banach_state_diff` vs `BanachSubstrate.state_at` at five default nu samples spanning the migration interior; drift tolerance `DRIFT_TOLERANCE = 1e-10`), and `SelfTestCadence(k=100, nu_samples=None, substrate=None)` (call-counter object whose `.tick(callback=...)` returns a `BanachDriftReport` on every k-th tick). Streaming integration: `forward_sweep_invert_stream` gains `self_test_cadence` + `self_test_callback` kwargs; the cadence advances per emitted frame; state-locality preserved (with-cadence recovery is byte-identical to without-cadence at `k=1`). The "async / out-of-band" framing in BLOCK_IN is satisfied in spirit at v5 — pure-Python microsecond ops — and gets a true second-thread implementation at v6 native. (2) **Sensitivity backprop** — new `sensitivity.py` carries `trajectory_substrate_diff(canonical, field, tau_obs_grid)` (per-frame substrate trajectory, shape `(T, 2)`, JAX-traceable, supports tangent_flow + learned), `trajectory_substrate_jacobian` (`∂substrate / ∂canonical` per frame, shape `(T, 2, 2)`), `field_parameter_sensitivity` (closed-form `∂substrate / ∂(delta_chit, delta_gamma, tau_obs_ref)` per frame, shape `(T, 2, 3)` — tangent-flow-only since the parameters live on `ScalingRule`), `inversion_sensitivity` (`∂canonical / ∂substrate` via `jax.jacfwd` through `tangent_flow_canonical_inverse`), and the BLOCK_IN-promised one-liner `driver_profile_loss_grad(loss_fn, canonical, field, tau_obs_grid, observed_substrates)` returning `{loss, grad_delta_chit, grad_delta_gamma, grad_tau_obs_ref}`. Lookup-table raises NotImplementedError on differentiable paths; learned-field hyperparameter optimization is curator-side (mpa-conform) by design. (3) **Gradient-based inversion** — `forward_sweep_invert` gains `method: Literal["auto", "grid", "gradient"] = "auto"`. Dispatch: `"auto"` (new default) routes `TangentFlowField` to closed-form (`jax_ops.forward_sweep_invert_diff`, exact at float64), `LearnedField` to L-BFGS (`scipy.optimize.minimize(method="L-BFGS-B")` with `jax.grad`-provided gradients, warm-started from grid argmin), `TranslationField` (lookup_table) to grid (v0–v4 byte-identical). `"grid"` forces v0–v4 behavior on any shape (byte-identity preserved across all 338 prior tests); `"gradient"` raises ValueError for lookup_table; `"invalid"` raises ValueError. `return_residual_field=True` always evaluates the grid (that's what the field IS), but `best_state` is from the method-chosen driver. `forward_map=` override forces grid (the override is opaque to the gradient driver). Wrapped variant (`forward_sweep_invert_wrapped`) accepts the same kwarg and forwards through both sidecar miss + direct compute branches. JAX 0.10 dropped `jax.scipy.optimize`; we route through scipy proper (already a transitive dep via JAX). Banach camera test residual sharpens from `< 0.001` (grid resolution) to `< 1e-12` (float64 closed-form) under method="auto". 54 new tests: `test_continuous_self_test.py` (20), `test_sensitivity_backprop.py` (18), `test_gradient_inversion.py` (16). 392 tests total green; all v0/v1/v2/v3/v4 fixtures pass unchanged. v5 is the last functional release; v6 is one-shot native port (zero new features). |
| 2026-05-16 | Rust v6.0.0 build — sessions 1-3 (BLOCK_IN §v6) | Native Rust port bootstrap, bundled because each session was thin enough that the per-version tag granularity is sessions 1-3 together. (1) **Session 1 — toolchain + math**: rustup + stable 1.95 + wasm32-unknown-unknown target; `rust/` scaffolded (single crate, workspace split deferred); `rust/src/math.rs` ported from `mpa_scale_solver/jax_core.py` — all 12 primitives (tangent_flow_substrate, banach_state, tangent_flow_canonical, lookup_squared_distance, tangent_flow_canonical_inverse, tangent_flow_inversion_residual, Laplace 2x2 covariance pair + log evidence, caputo_flow, mlp_forward, learned_field_substrate). `cargo build --release` (native rlib) and `cargo build --release --target wasm32-unknown-unknown` both clean; analytic sanity tests at `rust/tests/math.rs` — 17/17 pass after MSVC VC Tools workload landed via GUI Modify (per `feedback_msvc_workload_gui_install.md`). Doctests disabled (`doctest = false` in Cargo.toml) since the `///` blocks carry math notation. (2) **Session 2 — bit-identity infra**: emitter at `rust/tests/fixtures/emit_jax_core_reference.py` walks all 12 jax_core primitives over a small input sweep (48 cases, 22 KB JSON), committed at `rust/tests/fixtures/jax_core_reference.json`; Rust integration test at `rust/tests/bit_identity.rs` consumes via serde_json (first dev-dep) and asserts each Rust primitive in `src/math.rs` reproduces the Python output within a per-primitive ULP budget — **LIBM = 4 ULPs** for primitives composing few libm calls, **LIBM_WIDE = 16 ULPs** for primitives composing many libm calls or whose JAX-pairwise vs Rust-sequential reduction order can differ. Coverage-guard test ensures the fixture lists all expected primitives. 13/13 bit-identity + 17/17 analytic = 30/30 total. Load-bearing lesson: do NOT generate `target = python_forward(candidate)` and ask Rust to compute `residual = (rust_forward(candidate) - target)^2` — libm cancellation makes Rust's residual ~1e-32 instead of Python's exact 0; specify all paired inputs explicitly. (3) **Session 3 — types**: `mpa_scale_solver/types.py` → `rust/src/types.rs`. 17 structs + 5 enums + `TranslationField` tagged enum + `Activation`/`MlpLayer` re-exported from `math` so `LearnedField.weights` passes straight into `math::learned_field_substrate`. **Naming divergence (load-bearing):** Python's `TranslationField` is the lookup-table struct (with `LookupTableField` as alias and `AnyTranslationField` as the union); Rust uses `LookupTableField` for the struct and `TranslationField` for the tagged enum over `{LookupTable, TangentFlow, Learned}` — i.e. Rust's `TranslationField` corresponds to Python's `AnyTranslationField`. Serde tag is `shape`. `SidecarKey` is a `[u64; 3]` newtype over `f64::to_bits` with custom serde to a `':'`-joined string (works as a JSON map key). `_repr_html_` / `__repr__` overrides are Python display-only and do not port. serde + serde_json promoted from dev-deps to runtime deps. Smoke test (`rust/tests/types_smoke.rs`) covers serde round-trip on every public type and the `shape`-tag discriminator. Cross-language JSON parity (Python writes → Rust reads) remains deferred — it lands with whichever module ports first that actually serializes the relevant type to JSON (sidecar.py / mcp_server.py). 48/48 total tests (13 bit-identity + 17 analytic + 18 types_smoke); Python 392/392 still green; WASM build still clean. |
| 2026-05-16 | Rust v6.1.0 build — session 4 (BLOCK_IN §v6) | `operations.py` **raw forward path** plus three small deps. Four new modules: (1) `rust/src/gfdr_model.rs` — 5 pure functions ported from `mpa_scale_solver/gfdr_model.py` (`vertex_regime`, `alpha_s`, `plateau_height`, `generate_locus`, `interp_locus`, `locus_residual`) plus `LocusPoint` + `EmpiricalRow` structs as typed shapes of Python's `list[dict]`. (2) `rust/src/sidecar.rs` — `round_key` via `(x*10^n).round_ties_even()/10^n` matches Python `round`'s banker's-rounding for the bulk of inputs but diverges on `.x5`-decimal cases whose binary representation shifts off the exact halfway (documented in the module docstring as a cross-language wire-format caveat); `lookup_inverse` / `lookup_forward` as `BTreeMap<SidecarKey, _>::get`; cross-language wire-format parity still deferred per BLOCK_IN. (3) `rust/src/flow.rs` — `flow()` dispatching banach_exponential / generic / Caputo via `serde_json::Value`-typed refinement; returns `Result<CanonicalState, FlowError>` in lieu of Python's `NotImplementedError` / `ValueError`. (4) `rust/src/operations.rs` carries the raw forward path: `TranslationFieldIndex` + `apply_translation` + 3 field-shape helpers, `forward_sweep_invert_grid` (`method="grid"` only — gradient dispatch deferred to session 5), `tau_obs_sweep_grid`, `regime_at`, `regime_display_band`, and `gamut_classify` (returning typed `GamutClassification` / `GamutDiagnosis` rather than Python's dict). `score_fn` / `forward_map` Python kwargs surface as `Option<&dyn Fn(...)>` via type aliases (`ScoreFn` / `ForwardMap`). `default_substrate_score` iterates intersected keys in sorted (BTreeMap) order for deterministic float-sum; Python's hash-randomized set iteration produces ULP-level divergence absorbed by `LIBM_WIDE` in the bit-identity tests. Bit-identity fixture extended from 12 to 20 primitives (8 new fixture entries: `gfdr_alpha_s`, `gfdr_plateau_height`, `gfdr_vertex_regime`, `gfdr_generate_locus`, `gfdr_interp_locus`, `gfdr_locus_residual`, `sidecar_round_key`, `flow`); coverage-guard list bumped accordingly. Session-2 fixture lesson load-bearing again twice: `sidecar_round_key` excludes `.x5`-decimal inputs (Python `round` dtoa vs Rust `round_ties_even` disagree off exact binary halfway); `gfdr_locus_residual` empirical rows are synthetic invented values, NOT generated from `gfdr_model.generate_locus`, to dodge the candidate=truth self-residual cross-impl cancellation (Python exact 0, Rust ~5e-33). 75/75 Rust tests pass (19 src unit + 21 bit-identity + 17 math + 18 types_smoke); Python 392/392 still green; WASM build still clean. |
| 2026-05-16 | Rust v6.2.0 build — session 5 (BLOCK_IN §v6) | `forward_sweep_invert` **gradient inversion dispatcher** completes the `operations.py` method-dispatch surface. New: (1) `rust/src/optim.rs` — hand-rolled 2D damped-Newton solver (`minimize_smooth_2d`) with numerical finite-difference gradient + Hessian + backtracking line search. Substitutes Python's `scipy.optimize.minimize(method="L-BFGS-B") + jax.grad`. Justification: the problem dimension is fixed at 2, BLOCK_IN §v6 session-5 explicitly carves out non-byte-identity vs scipy (the optimizer just needs to converge to the same MAP within ~0.005 per axis from the grid-argmin warm start), and the deviation from the BLOCK_IN-noted `argmin` candidate avoids the dep footprint (compile time + WASM size + API surface) for ~80 lines of code on a 2D problem with a smooth near-quadratic cost. Newton converges in 2-3 outer iterations on the identity-MLP test; the Hessian inversion reuses `math::inv_2x2` (session 1). (2) **operations.rs additions**: `Method` enum (`Auto`/`Grid`/`Gradient`), `InversionResult` struct (slimmer than `GridInversionResult` — residuals optional because closed-form skips the grid), `forward_sweep_invert` dispatcher that routes per Python's `method` kwarg semantics — `Auto` → closed-form for TangentFlow / L-BFGS-equivalent for Learned / grid for LookupTable; `Grid` → grid on any shape; `Gradient` → closed-form/L-BFGS for differentiable shapes, errors on LookupTable. `forward_map` override forces grid (matches Python). New `OperationError::GradientOnLookupTable` mirror of Python's `ValueError("method='gradient' requires a differentiable field …")`. Private helpers: `invert_tangent_flow_closed_form` wraps `math::tangent_flow_canonical_inverse` (session-1 bit-identity tested); `invert_learned_bfgs` builds the squared-residual cost over `(chit, gamma)` and hands it to `optim::minimize_smooth_2d`, warm-started from the grid argmin. `_residual_at` mirror returns `sqrt(score)` at the recovered point — same scale as Python's `best_residual`. (3) **Type-alias API fix (load-bearing)**: `ScoreFn` / `ForwardMap` type aliases keep their declarations for documentation but the public signatures inline `&dyn Fn(...)` directly. Reason: `type Alias = dyn Fn(...)` defaults to `dyn Fn(...) + 'static`, locking callers to `'static` closures — local-state closures (like the camera-test `forward_map` capturing a call counter) wouldn't compile. Inlining lets the trait-object lifetime default to the enclosing reference's lifetime per Rust reference §"Default trait object lifetimes". 9 new gradient-inversion tests in `operations.rs` + 3 new `optim.rs` unit tests = 87/87 Rust tests total (up from 75); Python 392/392 still green; `cargo build --release --target wasm32-unknown-unknown` still clean. **BLOCK_IN-noted divergence to flag at session 6 read-cold:** the L-BFGS budget passed at ~0.005 per axis on the identity-MLP test (BLOCK_IN tolerance was ~0.005; test tolerance is < 0.01 — landed inside both with room). Cross-language verification against Python's `test_learned_field.py::TestForwardSweepInvertLearned` recovery set deferred — would require materializing weights from a fixture, which is the natural touch when `cross_substrate.py` / `active_learning.py` port (they exercise learned-field MAP). BanachSubstrate-based test (`TestBanachAutoSharpening`) skipped here — `banach.py` belongs to the curator-path port, which lives in mpa-conform; the Rust scale-solver consumes `BanachSubstrate` shape only, doesn't recreate it. Wrapped-variant test (`TestWrappedMethodKwarg`) deferred — wrapped variants land in session 7 with validation + provenance. |
| 2026-05-16 | Rust v6.3.0 build — session 6 (BLOCK_IN §v6) | **Intent algebra port** — `intent_map` + `intent_compose` + the five `_intent_iN` handlers + helpers from Python's `operations.py` (RFC-S §3 cut d). Four new `types.rs` public additions: (1) `IntentId` enum (`I1`..`I5` — typed rather than stringly so Python's runtime "unknown intent" ValueError becomes a compile-time impossibility), (2) `CapacityClass` enum (`Deep`/`Shallow` with lowercase serde matching Python's string convention), (3) `SacrificeRecord` struct, and (4) `IntentDiagnostics` tagged enum. The `#[serde(flatten)] + #[serde(tag = "intent")]` combination makes the JSON wire format a single flat dict matching Python's `sac` output shape verbatim — the three truly-common fields (`invariant_preserved`, `delta_chit`, `delta_gamma_AB`) live on the outer struct, intent-specific fields are typed per-variant. `intent` and `preserved_invariant` are derived methods on `SacrificeRecord` (not stored) since they're statically determined by which handler emitted the record; the BLOCK_IN-prep sketch held end-to-end. **operations.rs additions**: public `intent_map(state, tau_obs, gamut, IntentId)` returning `(CanonicalState, SacrificeRecord)`; public `intent_compose(state, tau_obs, gamut, &[IntentId])` returning `Result<(CanonicalState, Vec<SacrificeRecord>), OperationError>`; five private `intent_iN` handlers; helpers (`sign_i`, `capacity_class`, `clamp_to_gamut`, `regime_chit_interval`, `nearest_in_gamut_chit_for_regime`, `sign_preserving_clamp`) ported 1:1 from Python — `_REGIME_CHIT_INTERVALS` dict becomes a function with a `match RegimeLabel` body (typed lookup, same intervals); I3's deep→shallow recovery branch (try-the-same-side-endpoint logic) ports verbatim. Two new errors: `OperationError::IntentComposeEmpty` and `I2InComposition` mirror Python's two ValueErrors. **26 new src unit tests** mirror `tests/test_intents.py`'s TestI1RegimePreserving / I2 / I3 / I4 / I5 / TestComposition classes (TestValidation defers to session 7 alongside `*_wrapped` — same deferral pattern as session 5's wrapped-test skip); plus a serde-flatten round-trip smoke that catches the `#[serde(flatten)] + tag="intent"` combination. **Schema-parity test** added at session-6 end (folded into the same commit): new `sacrifice_record` section in `jax_core_reference.json` (13 cases, all 5 intents + edge cases — `_sacrifice_case` helper in the emitter), new `sacrifice_record_python_to_rust_json_parity` test in `bit_identity.rs` asserting Python's `sac` dict deserializes field-by-field into Rust `SacrificeRecord` (common fields + derived `.intent()` / `.preserved_invariant()` + per-variant diagnostics). **Asymmetric-parity is documented design**: Python's stored `preserved_invariant` STRING is silently dropped on Python→Rust read (serde default); Rust's `.preserved_invariant()` reconstructs it byte-for-byte from the variant — the test asserts the reconstructed string equals Python's emitted string, so symmetric round-trip is a one-line custom `Serialize` away if a future consumer ever needs it (none does as of session 6 — Python is the producer in the wrapped-variant path). **114/114 Rust tests pass** (57 src unit + 22 bit-identity — was 21, +1 sacrifice parity + 17 analytic math + 18 types_smoke); Python 392/392 still green; `cargo build --release --target wasm32-unknown-unknown` still clean. No new bit-identity *math* fixtures — session 6 is pure arithmetic, no math primitives added (BLOCK_IN-prep prediction confirmed). The session-7 carry-over for sacrifice JSON parity is closed pre-session; what remains for session 7 is the `OperationOutput<T>` wire-format parity (the wrapped variants stamp `OperationOutput<(CanonicalState, Vec<SacrificeRecord>)>`, which will need its own fixture + parity test mirroring the session-6 sacrifice_record approach). |
