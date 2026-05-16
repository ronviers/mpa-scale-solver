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
| `intent_map` | I5 signature-preserving remap (I1–I4: v2) |
| `validate_driver_profile` | RFC-S §5 round-trip residuals |

Plus the v1 addition `flow(canonical_initial, nu, field)` — continuous-form
`C^nu = exp(nu * ln C)` in Markovian scope. See
[`docs/CONTINUOUS_FLOW.md`](docs/CONTINUOUS_FLOW.md).

Each of the seven has a `*_wrapped` variant returning
`OperationOutput[T]` with a `ValidationReport` and `Provenance`
alongside the value (handoff §A.2 / §C.5 / §C.6). v0 sigs are unchanged;
v1 consumers that want validation + provenance call the wrapped
variants.

## Install

```
pip install -e .
```

Python 3.10+. Hard dep is `numpy`. Tests additionally need `pytest` and
optionally `matplotlib` (used by the camera test for the visual plot;
the test still passes without it).

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
