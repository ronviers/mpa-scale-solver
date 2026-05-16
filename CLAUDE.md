# mpa-scale-solver — session discipline

## What lives here

- The seven operations (`apply_translation`, `forward_sweep_invert`,
  `tau_obs_sweep`, `regime_at`, `gamut_classify`, `intent_map`,
  `validate_driver_profile`) plus the v1 addition `flow` for continuous
  `C^nu = exp(nu * ln C)`.
- The seven `*_wrapped` variants returning `OperationOutput[T]` with
  `ValidationReport` + `Provenance` riding alongside the value (handoff
  §A.2 / §C.5 / §C.6).
- Three translation-field shapes: `TranslationField` (lookup_table, v0),
  `TangentFlowField` (v1, RFC-S Appendix B item 1, with the Banach
  γ-scaling as the canonical leading-order rule), and `LearnedField`
  (v3, small JAX MLP). `apply_translation` dispatches on `field.shape`.
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
    `jax.numpy`. Is the math source the v6 native port reads. v2.1
    extends with Laplace-approximation primitives
    (`laplace_covariance_from_jacobian` / `_from_hessian` /
    `laplace_log_evidence`). v2.4 adds `caputo_flow`
    (Prony sum-of-exponentials approximation of the Mittag-Leffler
    kernel; differentiable in all parameters).
  - `jax_ops.py` — consumer surface returning JAX arrays:
    `tangent_flow_substrate_diff`, `flow_diff`,
    `tangent_flow_forward_jacobian`, `banach_state_diff`,
    `forward_sweep_invert_diff` (exact closed-form inverse on
    tangent-flow). Composes cleanly under `jax.grad` /
    `jax.jacobian` / `jax.hessian`. v2.1 adds `tangent_flow_posterior`
    + `lookup_table_posterior` (Laplace approximation).
  - `jax_pytree.py` — `CanonicalState` registered as a JAX PyTree
    (leaves: `(chit, gamma_AB)`; aux: `k_frust`). Idempotent
    side-effect on import.
- **v2.1 Bayesian inversion surface:** `Posterior` dataclass in
  `types.py` + `forward_sweep_invert_posterior` /
  `forward_sweep_invert_posterior_wrapped` in `operations.py`.
  Separate functions, not a kwarg on the existing wrapped variant
  (cleaner return-type contract). The wrapped variant follows the
  established `*_wrapped` validation + provenance pattern.
- **v2.3 intent algebra:** `intent_map` accepts all five RFC-S §3
  intents (I1–I5); each handler is a thin free function in
  `operations.py` returning `(mapped, sacrifice)`. New `intent_compose`
  + `intent_compose_wrapped` apply intents sequentially; I2 (drive-
  faithful) is rejected in any composition with other intents per §3.
  Sacrifice dicts carry `preserved_invariant` + `invariant_preserved`
  uniform keys plus intent-specific diagnostics; v1's I5 keys
  (`regime_preserved`, `original_regime`, `mapped_regime`) are
  preserved verbatim.
- **v3 cross-substrate compositions:** `cross_substrate.py` carries
  `gamut_overlap`, `canonical_distance` (l2 / l1 / regime /
  universality), `universality_agreement` (the framework's s→r
  migration test as a direct call) plus `*_wrapped` variants. The
  seven-operation API is unchanged — these are *cross-substrate
  compositions* per the thin-discipline rule below.
- **v3 active learning:** `active_learning.py` carries
  `suggest_measurements` returning `MeasurementCandidate`s ranked by
  composite score (posterior covariance-trace + gamut-edge proximity
  + intent-invariance fragility); weights configurable. Builds on
  v2.1 `Posterior` and v2.3 intent algebra.
- **v3 per-intent RFC-S §5 metrics:** `validate_driver_profile`
  dispatches per intent — I1 Hamming-on-regime + edge-type, I2
  L²-on-drive + max-γ, I3 ‖Γ*‖-deviation + capacity-class, I4
  ε-sequence-distance + survival, I5 universality-class + intra-class
  L². Aggregator helpers in `validation.py`
  (`per_intent_cell_metric`, `aggregate_per_intent_metrics`). v2.3
  back-compat keys (`forward_residuals`, `round_trip_residuals`,
  `regime_agreements`, `regime_agreement_rate`) preserved.
- **v3 MCP server:** `mcp_server.py` exposes 11 tools (7 core + 3
  cross-substrate + 1 active-learning) over stdio. Stateless, JSON
  I/O, read-only. Thin (one dispatch function per tool, hardcoded
  schemas; no framework gymnastics). Console script
  `mpa-scale-solver-mcp` declared in pyproject. `mcp>=1.0` hard dep.
- **v3 LearnedField:** small JAX MLP forward map under
  `jax_core.mlp_forward` + `jax_core.learned_field_substrate`;
  consumer entry `jax_ops.learned_field_substrate_diff`. Weights ride
  as nested tuples for JSON serialization; training is curator-side
  (mpa-conform). Forward-compat: the mpa-atlas driver-profile schema
  bump to admit `shape: "learned"` lands in a separate session;
  until then curators ship the learned-field block under the schema's
  `additionalProperties` allowance, parsed by
  `_parse_learned_field`.

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
- **`flow()` supports Markovian (`beta_mem = 1`) and non-Markovian
  Caputo (`beta_mem < 1`) scopes.** Markovian is the v1 closed-form
  exponential / generic tangent-flow path. Caputo is the v2.4 Prony
  sum-of-exponentials approximation of the Mittag-Leffler kernel; the
  curator pre-fits `prony_terms` and ships them on
  `ScalingRule.refinement`. Lookup-table flow still raises
  NotImplementedError (lookup tables sample the flow without an
  explicit generator).
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
- **v2.1**: Bayesian inversion via Laplace approximation. `Posterior`
  dataclass + `forward_sweep_invert_posterior` / `_wrapped`. Tangent-
  flow closed-form fast path; lookup-table weighted-moment fit.
  (Shipped 2026-05-16.)
- **v2.3**: All five RFC-S §3 intents (I1–I5) + composition algebra.
  Five thin handlers in `operations.py`; `intent_compose` enforces the
  I2-doesn't-compose rule. Uniform sacrifice-dict keys
  (`preserved_invariant`, `invariant_preserved`) coexist with v1's I5
  keys for back-compat. (Shipped 2026-05-16.) §v2.2 cut (c) N-mode
  cancelled.
- **v2.4**: non-Markovian Caputo flow via Prony sum-of-exponentials
  approximation of the Mittag-Leffler kernel. `flow()` and `flow_diff()`
  dispatch on `refinement['beta_mem']`; β = 1 stays on the v1 Markovian
  path byte-identically. (Shipped 2026-05-16; v2 complete.)
- **v3**: cross-substrate operations + active learning + MCP server +
  LearnedField (third translation-field shape, small JAX MLP) +
  per-intent RFC-S §5 metrics in `validate_driver_profile`. Seven-op
  API unchanged; cross-substrate ops live as compositions in their own
  module. (Shipped 2026-05-16; 286 tests green.)
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

## Acceptance for the v3 build session

Five items met as of 2026-05-16:

1. v0 + v1 + v2 fixtures pass unchanged (212 prior tests green plus 74
   new v3 tests: 34 cross-substrate, 11 active-learning, 17 MCP server,
   12 learned-field). 286 tests total green.
2. Cross-substrate ops in `cross_substrate.py` (`gamut_overlap`,
   `canonical_distance`, `universality_agreement`) + `*_wrapped`
   variants; the seven-operation API stays stable (these are
   compositions, not new fundamental ops).
3. Active learning: `suggest_measurements` returns ranked
   `MeasurementCandidate`s using v2.1 Posterior + v2.3 intent algebra.
4. Per-intent RFC-S §5 metrics in `validate_driver_profile`; v2.3
   back-compat keys preserved.
5. MCP server (`mcp_server.py`, stdio, 11 tools) +
   `mpa-scale-solver-mcp` console script; `mcp>=1.0` hard dep.
   LearnedField third shape with MLP forward map in jax_core/jax_ops;
   parser extension in `parse_translation_field`. README + CLAUDE.md
   updated; BLOCK_IN §v3 deleted; §v4/v5 refined.

## Acceptance for the v2.4 build session

Four items met as of 2026-05-16:

1. v0 + v1 + v2.0–v2.3 fixtures + 10 new `test_caputo_flow.py` tests
   pass (212 tests green total).
2. β=1 with `prony_terms=[(1.0, 1.0)]` is byte-identical to v1's
   Markovian Banach exponential (float-exact comparison across a
   representative ν grid).
3. New `jax_core.caputo_flow` primitive (differentiable, JIT-compiles);
   `flow()` / `flow_diff()` dispatch on `refinement['beta_mem']` (β < 1
   → Caputo; β = 1 → v1 unchanged); missing `prony_terms` with β < 1
   raises.
4. README + CLAUDE.md updated; BLOCK_IN §v2.4 deleted; v2 trajectory
   complete (block-in carries v3 → v6 only).

## Acceptance for the v2.3 build session

Three items met as of 2026-05-16:

1. v0 + v1 + v2.0/v2.1 fixtures + 28 new `test_intents.py` tests pass
   (202 tests green total). The deprecated `TestIntentMap::test_i1_to_i4_not_implemented`
   placeholder was removed; positive coverage replaces it in
   `test_intents.py`.
2. All five intents implemented as thin free functions in `operations.py`;
   `intent_compose` enforces the I2 composition rule.
3. README + CLAUDE.md updated; BLOCK_IN §v2.3 deleted.

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
