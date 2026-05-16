# mpa-scale-solver — session discipline

## What lives here

- The seven operations (`apply_translation`, `forward_sweep_invert`,
  `tau_obs_sweep`, `regime_at`, `gamut_classify`, `intent_map`,
  `validate_driver_profile`) plus the v1 addition `flow` for continuous
  `C^nu = exp(nu * ln C)`.
- The seven `*_wrapped` variants returning `OperationOutput[T]` with
  `ValidationReport` + `Provenance` riding alongside the value (handoff
  §A.2 / §C.5 / §C.6).
- Two translation-field shapes: `TranslationField` (lookup_table, v0)
  and `TangentFlowField` (v1, RFC-S Appendix B item 1, with the Banach
  γ-scaling as the canonical leading-order rule). `apply_translation`
  dispatches on `field.shape`.
- The `BanachSubstrate` calibration class + closed-form `state_at(nu)`
  used as the v1 camera-test fixture. Authoritative spec lives in
  mpa-conform; this repo vendors the runtime class.
- The `InverseLookupSidecar` type + dispatch helpers (sidecar production
  is mpa-conform's curator-path job; we only consume).
- The ported gFDR analytical forward model (`gfdr_model.py` ←
  `mpa-auditor/math/gfdr-model.js`).
- The synthetic substrate signal generator + per-frame
  `window_average_at_tau_obs` used only by the legacy v0 camera test.
- Schema dataclasses mirroring `mpa-atlas/schema/driver-profile.v2.0.json`.

Named family of operations, parallel to `mpa-solver`. Sibling, not nested.

## What does NOT live here

| Concern | Belongs to |
|---|---|
| Observable extraction (`correlator`, `response_direct`, `gfdr_locus`, `fit_invariants`) | `mpa-solver` |
| Bundle orchestration, curator path, researcher path | `mpa-conform` |
| Display / audit-engine rendering | `mpa-auditor` |
| Driver-profile production (substrate-class characterization) | `mpa-conform` curator path |
| Inverse-lookup-table sidecar production | `mpa-conform` curator path |
| Physics integration / trajectory ensembles | `mpa-solver` |
| RFC text, schemas, framework prose | `mpa-atlas` |

The line you will most want to cross and must not: "I'll just compute
alpha_s here, it's only a few lines." No. `fit_invariants` lives in
`mpa-solver`.

## Math caveats

- **Five-bucket regime classifier is canonical** (`vertex_regime` in
  `gfdr_model.py`). The three-bucket cut (`regime_display_band`) is a
  display-only helper for renderers.
- **`apply_translation` dispatches on `field.shape`**. lookup_table is
  the v0 nearest-neighbor over discrete rules; tangent_flow is the
  closed-form auto-remap with the Banach γ-scaling as the canonical
  leading-order rule (RFC-S Appendix B item 1).
- **`forward_sweep_invert` is brute-force grid search**. v1 adds the
  opt-in `sidecar=` kwarg on the wrapped variant for table-first
  dispatch; the v0 sig is unchanged. Adaptive refinement, ambiguity-set
  reporting, and gradient-based / Bayesian inversion are v2.
- **`flow()` is continuous in Markovian scope only** (`beta_mem = 1`).
  Non-Markovian Caputo (`beta_mem < 1`) is v2's fractional-RG
  generalization. v1 supports tangent-flow fields directly; lookup-table
  flow raises NotImplementedError.
- **The Banach substrate's `state_at(nu)` is the framework analytical
  truth**: closed-form `chit_0 * exp(-lambda_chit * nu)` per Q1 of the
  v1 build session. v2 derives the lambdas from `flow_spectrum` via the
  closed Wilson-Kadanoff construction.
- **Recovery resolution = grid resolution**. Discrete tables and brute-
  force grids partition canonical space into Voronoi cells; recovery is
  exact only when the candidate grid includes the rule canonical.
  Round-trip closure on seed-corpus profiles is checked at the
  substrate-cell level. Sub-grid-resolution recovery is v2 (JAX +
  gradient-based optimization).
- **τ_obs is an observer-fact, not a substrate-unknown** (mpa-auditor
  §Q13). It is declared and passed in; the operations never infer it
  from the data.

## Reproducibility

Stateless free functions on plain frozen dataclasses (the `BanachSubstrate`
methods are pure functions of the stored parameters — no mutable state).
Same inputs → byte-identical outputs. Fixtures under `tests/fixtures/`
lock behavior; changing the math intentionally requires bumping
`__version__` plus a commit note explaining the change. Per-seed
reproducibility (parallel-friendly) is the discipline the v6 native port
will match.

## Back-compat (v0 → v1)

- **v0 sigs unchanged**. Every v0 fixture passes unchanged in v1.
  `TranslationField` is still the lookup_table dataclass; `LookupTableField`
  is an alias for the handoff-spelled name.
- **`apply_translation` accepts `Union[TranslationField, TangentFlowField]`**
  (via `AnyTranslationField`). Old code passing a `TranslationField`
  works unmodified.
- **`*_wrapped` variants are additive**. They call the unwrapped v0
  operation and stamp validation + provenance onto an `OperationOutput`.
  Consumers that don't care call the v0 op as before.
- **`sidecar=` is opt-in everywhere**. Not passing one falls through to
  the v0 brute-force path.
- **The thread-local `get_last_call_metadata()` pattern from handoff
  §A.2 alternative is NOT implemented** (it would conflict with the
  stateless commitment in §A.3). Wrapped variants are the single
  back-compat-safe path for validation + provenance.

## Sibling-repo relationships

| Repo | Lives there | We read | We write |
|---|---|---|---|
| `mpa-atlas` | Specs, schemas, framework prose | RFC-S §§0–5, driver-profile.v2.0 schema, v9_compressed / cdv1_compressed, v9_receipts §RG closure + §Asymptotic closure | — |
| `mpa-solver` | Forward physics, observable extraction | C++/WASM `fit_invariants` (via mpa-conform Python bindings) | — |
| `mpa-conform` | Orchestration, curator + researcher path | Seed-corpus driver profiles (tests only); banach-substrate-reference.md (authoritative Banach spec) | — |
| `mpa-auditor` | Display, audit engine | `gfdr-model.js` (porting source for `gfdr_model.py`) | — |

Read-only consumer of `mpa-atlas` and `mpa-auditor`. Sibling-symmetric
with `mpa-solver`. Output is consumed by `mpa-conform`.

## Thin discipline borrowings

- **Thin-RFC discipline** (mpa-atlas/CLAUDE.md): we do not thicken the
  schema or the operation set. The seven operations + `flow` are the
  entire v1 surface; no eighth without a foundational-questions entry.
- **No declared virtues in user-facing copy** (memory): README and CLI
  output describe behavior, not virtue.
- **Document size by function, not percentage**: this file is short
  because the load-bearing distinctions fit short.

## Trajectory

- **v1 (this version)**: continuous `flow()` + tangent-flow translation
  field + Banach calibration substrate + inverse-lookup-table sidecar
  dispatch + per-call self-validation + full provenance trail. Seven
  wrapped variants. Pure numpy + Python.
- **v2**: JAX adoption, full differentiability, Bayesian inversion
  primitives, N-mode generalization, full I1–I5 intent operations,
  non-Markovian Caputo (`beta_mem < 1`) fractional-RG generalization.
- **v3**: cross-substrate operations, active learning, MCP server
  interface, learned translation-field form.
- **v4**: streaming / online operation, symbolic query interface,
  Mathematica-style exploration.
- **v5**: continuous self-test cadence, sensitivity backprop, gradient-
  based inversion replacing grid search where invertible.
- **v6**: one-shot native port (Rust or C++; language picked at session
  time). Matches the v5 Python under per-seed reproducibility. Zero new
  features.

Each is its own session, sequenced by the user via
`mpa-conform/docs/ROADMAP.md`.

## Acceptance for the v1 build session (handoff §E)

All twelve items met as of 2026-05-16:

1. v0 fixture regression passes unchanged.
2. Banach camera test passes with `max |residual| < 0.001` per axis.
3. v0 camera test (legacy `aging_log`) still passes — back-compat
   intact.
4. Seed-corpus integration passes (three real profiles, unchanged).
5. Sidecar dispatch test passes (with + without; provenance correctly
   records dispatch path).
6. Validation test passes (each flag fires when triggered).
7. Provenance test passes.
8. `pip install -e .` works in a fresh venv.
9. README, CLAUDE.md, docs/* updated; CONTINUOUS_FLOW.md, TANGENT_FLOW.md,
   SIDECAR_FORMAT.md, BANACH_SUBSTRATE.md added.
10. Sdist tarball built.
11. Tagged `v1.0.0` and pushed to `github.com/ronviers/mpa-scale-solver`.
12. Session log row appended to README.

## Session handoff

This v1 session is the second build on the v0→v6 trajectory. Next
sessions per the trajectory list above; each is its own handoff in
`mpa-conform/docs/`.
