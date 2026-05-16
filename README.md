# mpa-scale-solver

MPA scale-management kernel: τ_obs projection and canonical-frame
operations. Sibling to `mpa-solver` (forward physics + observable
extraction). Consumed by `mpa-conform` (orchestration → signed
`declaration_bundle.json`). Read by `mpa-auditor` (display).

The Python implementation is the production artifact through v5. A
native (Rust / C++ + WASM + Python bindings) port lands at v6, matching
the Python under the per-seed reproducibility discipline.

## The seven operations (surface stable across v0–v6)

| Operation | Role |
|---|---|
| `apply_translation` | canonical state → substrate-native at τ_obs (dispatches on `lookup_table` / `tangent_flow`) |
| `forward_sweep_invert` | substrate observation → canonical state at τ_obs (brute-force grid; sidecar fast-path via wrapped variant) |
| `tau_obs_sweep` | per-frame fan-out across a τ_obs grid |
| `regime_at` | five-bucket vertex regime classifier |
| `gamut_classify` | in-gamut / out-of-gamut diagnosis |
| `intent_map` | All five intents (I1–I5) + composition (`intent_compose`) per RFC-S §3 |
| `validate_driver_profile` | RFC-S §5 round-trip residuals |

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
