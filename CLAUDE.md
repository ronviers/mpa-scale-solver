# mpa-scale-solver — session discipline

## What lives here

- The seven operations (`apply_translation`, `forward_sweep_invert`,
  `tau_obs_sweep`, `regime_at`, `gamut_classify`, `intent_map`,
  `validate_driver_profile`).
- The ported gFDR analytical forward model (`gfdr_model.py` ←
  `mpa-auditor/math/gfdr-model.js`).
- The synthetic substrate signal generator + per-frame
  `window_average_at_tau_obs` used only by the camera test.
- Schema dataclasses mirroring `mpa-atlas/schema/driver-profile.v2.0.json`.

Named family of operations, parallel to `mpa-solver`. Sibling, not nested.

## What does NOT live here

| Concern | Belongs to |
|---|---|
| Observable extraction (`correlator`, `response_direct`, `gfdr_locus`, `fit_invariants`) | `mpa-solver` |
| Bundle orchestration, curator path, researcher path | `mpa-conform` |
| Display / audit-engine rendering | `mpa-auditor` |
| Driver-profile production (substrate-class characterization) | `mpa-conform` curator path |
| Physics integration / trajectory ensembles | `mpa-solver` |
| RFC text, schemas, framework prose | `mpa-atlas` |

The line you will most want to cross and must not: "I'll just compute
alpha_s here, it's only a few lines." No. `fit_invariants` lives in
`mpa-solver`.

## Math caveats

- **Five-bucket regime classifier is canonical** (`vertex_regime` in
  `gfdr_model.py`). The three-bucket cut (`regime_display_band`) is a
  display-only helper for renderers.
- **`apply_translation` is lookup + nearest-neighbor over the discrete
  rules**, not parametric dispatch. The v2.0 schema is forward-only /
  `lookup_table`. The parametric `aging_log` lives in
  `_test_fixtures.py` and is used only by the camera test.
- **`forward_sweep_invert` is brute-force grid search at v0**. Adaptive
  refinement, ambiguity-set reporting, and Bayesian primitives are v1.
- **Recovery resolution = table resolution**. Discrete tables partition
  canonical space into Voronoi cells; recovery is exact only when the
  candidate grid includes the rule canonical. Round-trip closure on
  seed-corpus profiles is checked at the substrate-cell level (matching
  `operating_point.label`), not at the scalar-canonical level.
- **τ_obs is an observer-fact, not a substrate-unknown** (mpa-auditor
  §Q13). It is declared and passed in; the operations never infer it
  from the data.

## Reproducibility

Stateless free functions on plain dataclasses. Same inputs →
byte-identical outputs. Fixtures under `tests/fixtures/` lock behavior;
changing the math intentionally requires bumping `__version__` plus a
commit note explaining the change.

## Sibling-repo relationships

| Repo | Lives there | We read | We write |
|---|---|---|---|
| `mpa-atlas` | Specs, schemas, framework prose | RFC-S §§0–5, driver-profile.v2.0 schema, v9_compressed / cdv1_compressed | — |
| `mpa-solver` | Forward physics, observable extraction | C++/WASM `fit_invariants` (via mpa-conform Python bindings) | — |
| `mpa-conform` | Orchestration, curator + researcher path | Seed-corpus driver profiles (tests only) | — |
| `mpa-auditor` | Display, audit engine | `gfdr-model.js` (porting source for `gfdr_model.py`) | — |

Read-only consumer of `mpa-atlas` and `mpa-auditor`. Sibling-symmetric
with `mpa-solver`. Output is consumed by `mpa-conform`.

## Thin discipline borrowings

- **Thin-RFC discipline** (mpa-atlas/CLAUDE.md): we do not thicken the
  schema or the operation set. The seven operations are the entire v0
  surface; no eighth without a foundational-questions entry.
- **No declared virtues in user-facing copy** (memory): README and CLI
  output describe behavior, not virtue.
- **Document size by function, not percentage**: this file is short
  because the load-bearing distinctions fit short.

## Acceptance for the v0 build session (handoff §E)

All ten items met as of 2026-05-15:

1. `H:/mpa-scale-solver/` exists, `git init`d.
2. Seven operations implemented per handoff §B + §C.
3. `gfdr_model.py` ported with the five-bucket classifier.
4. Unit tests + fixture regression + camera test pass (max\|residual\| = 0.012
   vs tolerance 0.05).
5. Seed corpus integration: neural-population, ck-glassy, surface-code-qec
   all close round-trip at substrate-cell level.
6. README, CLAUDE.md, docs/* written.
7. `pip install -e .` works in a fresh venv.
8. Sdist tarball built.
9. Pushed to `github.com/ronviers/mpa-scale-solver` (public, MIT).
10. Session log row appended to README.

## Session handoff

This is a single-session build per the v0 commit. Next sessions are
shaped by handoff §C.5:

- **v1**: I1–I4 intent operations, residual-field return-from-invert
  refinements, migration-trajectory analytics, compactification-point
  detection.
- **v2**: native build (Rust + WASM + Python bindings, byte-identical to
  this Python), non-trivial RG-flow defaults, tangent-flow translation
  field (RFC-S Appendix B item 1), N-mode, sensitivity, learned form,
  cross-substrate gamut operations.

Each is its own session, sequenced by the user.
