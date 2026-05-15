# Order of operations

Restatement of handoff §B for pipeline integrators. The shape is not "five
compute steps" — it is **five ordering constraints** plus **three named
inner traversals**.

## The five-step skeleton

From `mpa-auditor/docs/foundational-answers.md` §Q13:

```
declare (class, columns, units, τ_obs)
  → τ_obs selects the canonical frame
  → forward-project (apply_translation at this τ_obs)
  → sweep-fit (forward_sweep_invert: substrate observation → canonical state)
  → audit (compare to prediction over validity_range ∩ gamut)
```

## Ordering constraints

| Constraint | What it means |
|---|---|
| τ_obs declared before any projection | τ_obs is an observer-fact, not a substrate-unknown. The camera frame is an input to `apply_translation`, not an output. There is no "infer τ_obs from the data" path. |
| Translation field applied before inversion | The inversion is forward-sweep search through forward projections. You cannot fit without first being able to forward-project. |
| Regime classification after projection | Regime label is τ_obs-conditional. Classifying pre-projection is meaningless. |
| Gamut check needs canonical state | Gamut is the image of the RG trajectory in canonical space. Substrate state must be in canonical coords first. |
| Intent map only fires when gamut check fails | Intents enumerate which invariants are preserved when out-of-gamut. In-gamut states pass through. |

## Three named inner traversals

### Audit traversal (single τ_obs declared)

```
substrate-native observable
  → window-average at the declared τ_obs                  (substrate_signal.window_average_at_tau_obs)
  → apply_translation^(-1) via forward_sweep_invert       (operations.forward_sweep_invert)
  → canonical state (chit, gamma_AB, ...)
  → regime_at                                              (operations.regime_at)
  → gamut_classify                                         (operations.gamut_classify)
  → in-gamut: pass through; out-of-gamut: intent_map      (operations.intent_map, I5 at v0)
  → compare to predicted canonical state
```

### s → r migration traversal (τ_obs swept)

The framework's primary cross-substrate test. **Per-frame fan-out**, not
a pipeline applied once (handoff §C.1):

```
substrate-native multi-window observables
  → for each τ_obs in the grid:
       window_average_at_tau_obs(signal, τ)               # per-frame, one window per τ
       apply_translation                                  # candidate forward map at τ
       forward_sweep_invert                               # recovered canonical at τ
       regime_at                                          # classification at τ
  → trajectory of (τ_obs, canonical_state, regime)
  → trajectory shape IS the audit signature (c → s → r migration)
```

The camera test (`tests/test_camera_migration.py`) instances this with a
synthetic substrate whose analytical truth is known.

### Driver-profile validation traversal (RFC-S §5)

```
reference canonical state at τ_obs_ref
  → forward-project (apply_translation)
  → predicted substrate-native observation
  → compare to known reference substrate observation (forward residual)
  → invert (forward_sweep_invert)
  → compare to original canonical state (round-trip residual)
  → both residuals must be within intent-specific threshold
```

Instanced by `validate_driver_profile`. The seed-corpus integration tests
(`tests/test_seed_corpus.py`) check this traversal at substrate-cell
closure on neural-population, ck-glassy, and surface-code-qec profiles.

## What is NOT in the order of operations

- No backward-direction translation field. The map is forward-only by
  architecture (§Q13). The backward direction is ill-posed and never built.
- No silent τ_obs inference. Declared input, not derived output.
- No alpha_s extraction here. That lives in `mpa-solver`'s
  `fit_invariants`; this repo consumes the result.
