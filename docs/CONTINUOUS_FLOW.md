# Continuous flow `flow(canonical_initial, nu, field)`

The v1 addition to the seven-operation surface. Returns the canonical
state at depth `nu` under the field's flow generator.

## Mathematical form

`C^nu = exp(nu * ln C)` — continuous-form RG flow. Closed by composition
in Markovian scope (`beta_mem = 1`) per `v9_receipts.md` §RG closure
(Wilson–Kadanoff structural equivalence; cite: cdv1 §Heat-tax tower,
cdv1 §Universal two-mode kernel, Wilson–Polchinski functional RG). The
Banach substrate sits exactly at the Markovian boundary by construction
and is the v1 reference instance.

For integer `nu = N`, the closed form is equivalent to `N` successive
applications of the discrete generator. For real `nu`, the closed form
is the spectral functional calculus on `C`.

## v1 dispatch

`flow()` dispatches on `field.shape`:

| Shape | v1 behavior |
|---|---|
| `tangent_flow` with `refinement.flow_kind == "banach_exponential"` | Closed-form exp decay: `chit(nu) = chit_0 * exp(-lambda_chit * nu)`, `gamma_AB(nu) = gamma_AB_0 * exp(-lambda_gamma * nu)`. Defaults `lambda_chit = lambda_gamma = 1.0` (v1 normalization; v2 derives from `flow_spectrum`). |
| `tangent_flow` without a `flow_kind` | Apply the `ScalingRule` treating `nu` as `tau_obs`. For `delta_chit = delta_gamma = 0` (the canonical default) this is identity. |
| `lookup_table` | `NotImplementedError`. Lookup tables sample the flow; reconstructing a continuous generator requires the fractional-RG generalization, which lands at v2 alongside JAX. |

## v1 scope, v2+ extensions

| Capability | Lands at |
|---|---|
| Markovian scope (`beta_mem = 1`) closed form | v1 |
| Banach substrate exponential decay (closed form, this version) | v1 |
| Generic tangent-flow scaling rule | v1 |
| Non-Markovian Caputo (`beta_mem < 1`) fractional-RG generalization | v2 |
| Differentiability via JAX | v2 |
| Spectral functional calculus on lookup-table fields | v2 |

## Why this is what `flow()` does at v1

The v1 acceptance test is the Banach camera test (`tests/test_banach_camera.py`).
The Banach substrate's `state_at(nu)` is the framework's analytical
truth; `flow()` returns the same closed form so the test can compare
solver pipelines against an algebraic expression. Per Q1 of the v1 build
session this resolves the "two integrations of the same generator" risk
flagged at scope confirmation.

## Asymptotic-Closure compliance

`flow()` on the Banach substrate is exp decay toward `(0, 0)`. The
asymptote is never reached at any finite `nu` — Asymptotic-Closure-
compliant per v9 §Asymptotic closure. Validation reports on the wrapped
operations flag `flow()` outputs that land on exact 0.0 or 1.0 floats
(which would require either `nu = infinity` or `chit_0 = 0` — both
outside the Banach substrate's domain).
