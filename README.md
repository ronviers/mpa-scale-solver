# mpa-scale-solver

MPA scale-management kernel: τ_obs projection and canonical-frame
operations. Sibling to `mpa-solver` (forward physics + observable
extraction). Consumed by `mpa-conform` (orchestration → signed
`declaration_bundle.json`). Read by `mpa-auditor` (display).

The Python implementation is the v0 shipping artifact. A native
(Rust / C++ + WASM + Python bindings) port comes later, byte-identical
to this Python.

## The seven operations

| Operation | Role |
|---|---|
| `apply_translation` | canonical state → substrate-native at τ_obs (forward) |
| `forward_sweep_invert` | substrate observation → canonical state at τ_obs |
| `tau_obs_sweep` | per-frame fan-out across a τ_obs grid |
| `regime_at` | five-bucket vertex regime classifier |
| `gamut_classify` | in-gamut / out-of-gamut diagnosis |
| `intent_map` | I5 signature-preserving remap (I1–I4: `NotImplementedError`) |
| `validate_driver_profile` | RFC-S §5 round-trip residuals |

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

## The camera migration test

`tests/test_camera_migration.py` is the visual end-to-end test. It builds a
synthetic substrate whose tau_obs-camera sweep traces the framework's
c → s → r migration, then verifies forward_sweep_invert recovers analytical
canonical truth across 80 log-spaced frames.

```
pytest tests/test_camera_migration.py
```

Outputs `tests/out/migration_compare.png` (numerical curve overlaid on
analytical) and `tests/out/result.json`. Pass criterion: max |residual| ≤ 0.05.

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
| 2026-05-15 | Python v0.1.0 build (this session) | Seven operations shipped; gfdr_model.js ported (5-bucket); camera test passes max\|residual\| = 0.012 vs tolerance 0.05; all three seed-corpus profiles pass round-trip closure. |
