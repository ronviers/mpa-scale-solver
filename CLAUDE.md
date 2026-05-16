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
- **v2.0 JAX surface (parallel to v0/v1; opt-in):**
  - `jax_core.py` — pure JAX math primitives, float64-enabled,
    JIT-able, differentiable. Mirrors the v0/v1 closed forms in
    `jax.numpy`. Is the math source the v6 native port reads.
  - `jax_ops.py` — consumer surface returning JAX arrays:
    `tangent_flow_substrate_diff`, `flow_diff`,
    `tangent_flow_forward_jacobian`, `banach_state_diff`,
    `forward_sweep_invert_diff` (exact closed-form inverse on
    tangent-flow). Composes cleanly under `jax.grad` /
    `jax.jacobian` / `jax.hessian`.
  - `jax_pytree.py` — `CanonicalState` registered as a JAX PyTree
    (leaves: `(chit, gamma_AB)`; aux: `k_frust`). Idempotent
    side-effect on import.

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
  dispatch; the v0 sig is unchanged. v2.0 adds
  `jax_ops.forward_sweep_invert_diff` — exact closed-form analytical
  inverse for tangent-flow fields (differentiable through the target),
  routed through `jax_core.tangent_flow_canonical_inverse`. Bayesian
  inversion is v2.1; generic gradient-based (BFGS / L-BFGS / Newton)
  for non-tangent-flow differentiable forward maps is v5.
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
  substrate-cell level. For tangent-flow fields v2.0 provides
  sub-grid-resolution recovery via the closed-form
  `jax_ops.forward_sweep_invert_diff` (exact at float64 precision); for
  lookup-table fields the grid remains the binding constraint until v5
  lands a smooth-surrogate dispatch.
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

## Back-compat (v0 → v1 → v2.0)

- **v0 sigs unchanged**. Every v0 fixture passes unchanged in v1 and
  v2.0. `TranslationField` is still the lookup_table dataclass;
  `LookupTableField` is an alias for the handoff-spelled name.
- **`apply_translation` accepts `Union[TranslationField, TangentFlowField]`**
  (via `AnyTranslationField`). Old code passing a `TranslationField`
  works unmodified.
- **`*_wrapped` variants are additive**. They call the unwrapped v0
  operation and stamp validation + provenance onto an `OperationOutput`.
  Consumers that don't care call the v0 op as before.
- **`sidecar=` is opt-in everywhere**. Not passing one falls through to
  the v0 brute-force path.
- **v2.0 JAX surface is parallel and opt-in**. `jax_core` / `jax_ops`
  add new entry points; the existing `operations.*` / `flow.*` /
  `banach.*` paths still use Python `math.*` for byte-identity with the
  fixture suite. Consumers that don't need gradients call the v0/v1
  surface as before.
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
  entire surface; no eighth without a foundational-questions entry.
  v2.0's `*_diff` variants are not eighth operations — they are
  differentiable shapes of the existing seven, returning JAX arrays
  rather than dataclasses, and dispatched on consumer opt-in.
- **No declared virtues in user-facing copy** (memory): README and CLI
  output describe behavior, not virtue.
- **Document size by function, not percentage**: this file is short
  because the load-bearing distinctions fit short.

## Trajectory

- **v1**: continuous `flow()` + tangent-flow translation field + Banach
  calibration substrate + inverse-lookup-table sidecar dispatch +
  per-call self-validation + full provenance trail. Seven wrapped
  variants. Pure numpy + Python. (Shipped 2026-05-16.)
- **v2.0**: JAX foundation + differentiability. New modules: `jax_core`
  (math primitives), `jax_ops` (consumer surface), `jax_pytree`
  (CanonicalState as JAX PyTree). v0/v1 surfaces unchanged. JAX
  becomes a hard dep. (Shipped 2026-05-16.)
- **v2.1–v2.4**: remaining v2 slices per BLOCK_IN cuts (b)–(e):
  Bayesian inversion, N-mode generalization, I1–I4 intents,
  non-Markovian Caputo. Each its own session; each builds on the v2.0
  jax_core / jax_ops foundation.
- **v3**: cross-substrate operations, active learning, MCP server
  interface, learned translation-field form (LearnedField uses
  jax_core / jax_ops directly).
- **v4**: streaming / online operation, symbolic query interface,
  Mathematica-style exploration.
- **v5**: continuous self-test cadence, sensitivity backprop (composes
  jax_ops Jacobians into the full trajectory chain rule), gradient-
  based inversion replacing grid search where invertible (jax_ops
  already provides the tangent-flow closed-form; v5 generalizes to
  learned / lookup-table-smooth-surrogate cases).
- **v6**: one-shot native port (Rust or C++; language picked at session
  time). `jax_core.py` is the math source the port reads; the v0/v1
  operations are wrapper-shape only. Matches the v5 Python under
  per-seed reproducibility. Zero new features.

Each is its own session, sequenced by the user via
`mpa-conform/docs/ROADMAP.md`.

## Acceptance for the v1 build session (handoff §E)

All twelve items met as of 2026-05-16. (Detail moved to README session
log; the v1 acceptance contract is locked.)

## Acceptance for the v2.0 build session

Five items met as of 2026-05-16:

1. v0 + v1 fixture regression passes unchanged (125 prior tests green
   plus 32 new differentiability tests).
2. JAX 0.10.0 installed and active on Windows CPU; float64 enabled
   at `jax_core` import.
3. New `mpa_scale_solver.jax_core` / `jax_ops` / `jax_pytree` modules
   exposed; CanonicalState round-trips through `jax.tree_util` and
   `jax.grad` works directly on CanonicalState-typed callbacks.
4. Differentiability tests pass: forward map matches v1 closed form
   at abs=1e-12, autograd matches finite-difference at rtol=1e-6 /
   atol=1e-9, JIT compiles cleanly, analytical inverse exact at
   abs=1e-12 over a sweep of tau_obs values.
5. pyproject bumped to 2.0.0 with JAX as a hard dep; README +
   CLAUDE.md + BLOCK_IN.md updated.

## Session handoff

This v2.0 session is the third build on the v0→v6 trajectory. The
v2.1 → v6 trajectory is governed by the **self-evolving block-in
handoff** at [`docs/BLOCK_IN.md`](docs/BLOCK_IN.md). Each session that
lands a version deletes its own §vN section from that doc and refines
the remaining sections in place. Historical "what shipped" stays in
this repo's `README.md` § Session Log.

When opening a scale-solver session, read [`docs/BLOCK_IN.md`](docs/BLOCK_IN.md)
§vN for the version being built. Read [`docs/NORTH_STAR.md`](docs/NORTH_STAR.md)
for the destination context.
