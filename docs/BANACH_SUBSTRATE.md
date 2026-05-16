# Banach substrate (local pointer)

The framework's canonical calibration reference. Lives in **mpa-conform**
as the spec artifact; this repo vendors the `BanachSubstrate` class for
the v1 camera test and sidecar dispatch.

## Authoritative spec

**[H:/mpa-conform/docs/banach-substrate-reference.md](../../mpa-conform/docs/banach-substrate-reference.md)** —
state space, generator, normalization manifest, operations, use cases,
implementation falsifiers, provenance.

## v1 implementation here

`mpa_scale_solver.banach.BanachSubstrate` is the runtime class used by
the camera test and as a reference producer for inverse-lookup sidecars.

```python
from mpa_scale_solver import BanachSubstrate, flow

substrate = BanachSubstrate(chit_0=1.5, gamma_AB_0=-0.5)
field = substrate.translation_field()

# Analytical truth (closed-form exponential decay):
truth_at_nu = substrate.state_at(2.0)

# Solver-side computation (same closed form via flow()):
flowed = flow(substrate.canonical_initial(), 2.0, field)

# Substrate observation at depth nu (identity translation):
obs = substrate.substrate_at(2.0)

# Inverse-lookup sidecar for the dispatch fast-path:
sidecar = substrate.build_sidecar(np.logspace(-2, 1, 80))
```

## Closed-form trajectory (Q1 v1 normalization)

```
chit(nu)    = chit_0 * exp(-lambda_chit * nu)
gamma_AB(nu) = gamma_AB_0 * exp(-lambda_gamma * nu)
```

Defaults `lambda_chit = lambda_gamma = 1.0` correspond to the v1
normalization (spectral-gap eigenvalue `exp(-1)` of `ln C`). v2 derives
the lambdas from the `flow_spectrum` directly via the closed
Wilson–Kadanoff construction in `v9_receipts §RG closure`.

Defaults `chit_0 = 1.5`, `gamma_AB_0 = -0.5` are a c-band start with
cooperative gamma so the trajectory traces the full c → s migration
interior as `nu` sweeps. Asymptotic-Closure-compliant: `chit -> 0` as
`nu -> infinity` but never reaches 0 at any finite `nu`.

## What `apply_translation` does for Banach

Identity translation. Substrate observables equal canonical values:

```
substrate.observables["substrate_chit"]    = canonical.chit
substrate.observables["substrate_gamma_AB"] = canonical.gamma_AB
```

This is the `delta_chit = delta_gamma = 0` case of the tangent-flow
scaling rule. The RG flow happens in canonical space and is read by
`flow()`, not by the translation.

## Camera test pass criterion (handoff §E item 2)

`max |residual| <= 0.001` per axis across the 80-frame log-spaced
tau_obs sweep. See `tests/test_banach_camera.py`. The legacy v0 camera
test (`tests/test_camera_migration.py`) is kept passing as back-compat
coverage of the lookup-table dispatch path.
