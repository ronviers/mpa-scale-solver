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
- **v4 streaming inversion:** `streaming.py` carries `InversionResult`
  + `forward_sweep_invert_stream(observations, field, canonical_grid,
  *, tau_obs=None, score_fn=None)` (generator yielding
  `InversionResult` per consumed observation). State-local per frame;
  `tau_obs=None` uses each `obs.tau_obs` (per-frame observer scale).
  Thin source adapters: `from_iterable` (passthrough) and `from_stdin`
  (JSON-per-line). WebSocket / polling deferred — any iterable fits
  the same interface. Wrapped streaming variant not exposed —
  consumers that want `OperationOutput` per frame call
  `forward_sweep_invert_wrapped` in their own loop.
- **v4 symbolic query DSL:** `symbolic_query.py` carries `query()` +
  `QueryResult` + `QueryParseError`. Five patterns max (BLOCK_IN
  §v4 cap): `regime at`, `gamut at`, `translate`, `invert`
  (tangent_flow only — closed-form algebraic inverse), and
  `tau where regime crosses` (closed-form for tangent_flow with
  delta_chit ≠ 0; bisection over `tau_range` for lookup_table /
  learned). Structural context (field, gamut, grid, tau_range) rides
  as kwargs; the DSL parses intent and literal parameters only.
  Mathematica-style: closed-form expression + numerical evaluation
  where both exist. Read-only — no operation mutates the field.
- **v4 notebook ergonomics:** `_repr_html_` methods on every
  user-facing dataclass (CanonicalState, SubstrateState, GamutSpec,
  RegimeReading, TranslationField, TangentFlowField, LearnedField,
  Posterior, OperationOutput, MeasurementCandidate, InversionResult)
  render compact HTML tables in Jupyter. `__repr__` overridden on
  `Posterior`, `LearnedField`, `OperationOutput` to avoid dumping
  nested tuples / weight matrices in REPL output. The default
  dataclass `__repr__` is preserved everywhere else for back-compat.
  A small `_html_table` helper at the top of `types.py` keeps the
  HTML method bodies short.
- **v4 default plot hooks:** `plotting.py` carries
  `plot_trajectory(trajectory, ...)` (regime-banded canonical curve),
  `plot_gamut(gamut, *, points=...)` (gamut envelope + overlay),
  `plot_residual_field(residuals, canonical_grid, *, recovered=...)`,
  and `plot_posterior(posterior, *, n_sigma=2.0)` (MAP + Laplace
  ellipse). Each takes `backend="matplotlib"` (default) or
  `"plotly"`. Both backends are lazily imported (`try: import
  matplotlib`) — matplotlib stays in test-extras; plotly is fully
  optional (no project dep). Animation / scrubber UIs deferred to
  mpa-auditor per the suite block-in.
- **v5 continuous Banach self-test cadence:** `self_test.py` carries
  `BanachDriftReport` + `run_banach_self_test` +
  `SelfTestCadence(k=100, ...)`. Synchronous per-tick check —
  microsecond-scale pure-Python ops; the "out-of-band where backend
  permits" framing in BLOCK_IN is satisfied in spirit, with v6 native
  getting a true second-thread implementation. Streaming integration:
  `forward_sweep_invert_stream` takes `self_test_cadence` +
  `self_test_callback` kwargs; the cadence advances per **emitted
  frame** and the callback receives the `BanachDriftReport` on every
  k-th frame. State-locality preserved — the self-test does NOT
  feed back into the primary inversion.
- **v5 sensitivity backprop:** `sensitivity.py` carries
  `trajectory_substrate_diff(canonical, field, tau_obs_grid)` (per-frame
  substrate observations, shape `(T, 2)`),
  `trajectory_substrate_jacobian` (`∂substrate / ∂canonical` per frame,
  shape `(T, 2, 2)`), `field_parameter_sensitivity`
  (`∂substrate / ∂(delta_chit, delta_gamma, tau_obs_ref)` per frame,
  shape `(T, 2, 3)`), `inversion_sensitivity`
  (`∂canonical / ∂substrate` via the closed-form analytical inverse),
  and the one-liner `driver_profile_loss_grad(loss_fn, canonical,
  field, tau_obs_grid, observed)` that the BLOCK_IN §v5 promised.
  Composes v2.0's per-op Jacobians from `jax_ops` through the full
  audit traversal. Tangent-flow + learned only — lookup-table raises
  NotImplementedError (no differentiable forward map). Observe-only:
  no mutation of fields (frozen dataclasses).
- **v5 gradient-based inversion:** `forward_sweep_invert` gains a
  `method` kwarg with dispatch table — `"auto"` (default) routes
  `TangentFlowField` to closed-form
  (`jax_core.tangent_flow_canonical_inverse`, exact at float64),
  `LearnedField` to L-BFGS (scipy's L-BFGS-B with `jax.grad`-provided
  gradients, warm-started from grid argmin),
  `TranslationField` (lookup_table) to grid (v0–v4 behavior).
  `"grid"` forces v0–v4 byte-identical behavior on any shape;
  `"gradient"` works on differentiable shapes only and raises for
  lookup_table. `return_residual_field=True` always runs the grid
  (that's what the field IS), but the best_state under `method="auto"`
  remains the closed-form / gradient result. `forward_map=` override
  forces grid (the override is opaque to the gradient driver).
  Same kwarg lands on `forward_sweep_invert_wrapped`.

Named family of operations, parallel to `mpa-solver`. Sibling, not nested.

**v6 Rust port (in progress, started 2026-05-16):** `rust/` is the
native+WASM port tree. `rust/src/math.rs` mirrors `jax_core.py` 1:1,
and `rust/src/types.rs` mirrors `types.py` (the runtime dataclass
shapes). Future modules land alongside (`operations.rs`,
`sensitivity.rs`, `sidecar.rs`, etc.) and bindings (pyo3,
wasm-bindgen) land later. Per-seed reproducibility is the contract;
the Python remains the executable spec until v6 ships, then this
CLAUDE.md section flips to "Rust is canonical, Python is reference."

Bit-identity scaffold (session 2):
`rust/tests/fixtures/emit_jax_core_reference.py` emits per-primitive
input / output JSON from the Python `jax_core`;
`rust/tests/fixtures/jax_core_reference.json` is the committed
fixture; `rust/tests/bit_identity.rs` is the integration test that
loads it and asserts per-primitive ULP equality. The pattern extends
to each new Rust module — add a per-primitive case generator + a
top-level fixture key in the emitter, regenerate the JSON, add a
matching `#[test]` function in `bit_identity.rs`. The coverage-guard
test in `bit_identity.rs` will need its expected-primitive list
extended when new modules land. Lesson from session 2: do NOT
generate `target = python_forward(candidate)` and ask Rust to
compute `residual = (rust_forward - target)^2` — libm cancellation
makes Rust's residual ~1e-32 instead of Python's exact 0. Specify
all paired inputs explicitly in the fixture.

types.rs (session 3): the 17 Python dataclasses + 5 enums, plus
`Activation` and `MlpLayer` re-exported from `math` so that
`LearnedField.weights: Vec<MlpLayer>` passes directly into
`math::learned_field_substrate` with no conversion shim. **Naming
divergence from Python (load-bearing):** Python's `TranslationField`
is the lookup-table struct, with `LookupTableField` as alias and
`AnyTranslationField` as the union. In Rust the struct is named
`LookupTableField` (matching its shape) and `TranslationField` is
the tagged enum over `{LookupTable, TangentFlow, Learned}` — i.e.
Rust's `TranslationField` corresponds to Python's
`AnyTranslationField`, not Python's `TranslationField`. The
serde tag is `shape` (matches the Python `.shape` field
discriminator). `SidecarKey` is a `[u64; 3]` newtype over
`f64::to_bits` of the rounded floats, with custom serde to a
`':'`-joined string so it works as a JSON map key. The smoke
test (`rust/tests/types_smoke.rs`) covers serde round-trip on
every public type. Cross-language JSON parity (Python writes →
Rust reads) is unproven at types.rs alone; it lands when the
first module with actual JSON I/O ports (`sidecar.py` /
`mcp_server.py`).

gfdr_model.rs + sidecar.rs + flow.rs + operations.rs (session 4):
the raw forward path. `gfdr_model.rs` mirrors `gfdr_model.py` 1:1
(5 pure functions; `LocusPoint` / `EmpiricalRow` as the typed
shapes of Python's `list[dict]`). `sidecar.rs` carries `round_key`
+ `lookup_inverse` + `lookup_forward`; the round function uses
Rust's `round_ties_even` which agrees with Python `round`'s
banker's-rounding for the bulk of inputs but diverges on
`.x5`-decimal cases whose binary representation shifts off the
exact halfway — documented in the module docstring as a
cross-language wire-format caveat. `flow.rs` carries `flow()` with
banach-exponential / generic / Caputo dispatch, returning
`Result<CanonicalState, FlowError>` in lieu of Python's
`NotImplementedError` / `ValueError`. `operations.rs` carries the
raw forward path: `TranslationFieldIndex`, `apply_translation` +
3 field-shape helpers, `forward_sweep_invert_grid` (Python
`method="grid"` only; gradient dispatch lands session 5),
`tau_obs_sweep_grid`, `regime_at`, `regime_display_band`, and
`gamut_classify` (returns typed `GamutClassification` /
`GamutDiagnosis` rather than Python's `dict`). Python's
`score_fn` / `forward_map` callable kwargs surface as
`Option<&dyn Fn(...)>` so callers can pass `None` cleanly.
`default_substrate_score` iterates intersected keys in sorted
(BTreeMap) order for deterministic float-sum across Rust runs;
Python's hash-randomized set-iteration order is absorbed by
`LIBM_WIDE = 16 ULPs` in the bit-identity tests. Bit-identity
fixture extended from 12 to 20 primitives; **75/75 Rust tests**
green (19 src unit + 21 bit-identity + 17 analytic math + 18
types smoke). Session-2 fixture lesson kicked in twice: see the
`sidecar_round_key` (drop `.x5`-decimal cases) and
`gfdr_locus_residual` (synthetic empirical, not generator-seeded)
comments in `emit_jax_core_reference.py`.

optim.rs + operations.rs gradient dispatcher (session 5): the
`forward_sweep_invert` `method` kwarg completes. New `optim.rs`
carries `minimize_smooth_2d` — a hand-rolled 2D damped-Newton
solver (numerical FD gradient + Hessian + backtracking line search;
Hessian inversion via `math::inv_2x2` from session 1). Substitutes
Python's `scipy.optimize.minimize(method="L-BFGS-B") + jax.grad`.
**Deliberate divergence from the BLOCK_IN-noted `argmin` candidate:**
the problem dimension is fixed at 2, BLOCK_IN §v6 session-5
carves out non-byte-identity vs scipy (the optimizer just needs
to converge to the same MAP within ~0.005 per axis from the
grid-argmin warm start), and ~80 lines of hand-rolled code beats
pulling `argmin + argmin-math` for the compile-time / WASM-size /
API-surface cost. Newton converges in 2-3 outer iterations on a
smooth near-quadratic cost (the identity-MLP case). `operations.rs`
adds `Method::{Auto, Grid, Gradient}` enum + `InversionResult`
struct (residuals optional; closed-form path skips the grid) +
`forward_sweep_invert` dispatcher routing per Python's `method`
kwarg semantics. New `OperationError::GradientOnLookupTable`
mirrors Python's `ValueError`. Private helpers:
`invert_tangent_flow_closed_form` wraps
`math::tangent_flow_canonical_inverse` (session-1 bit-identity
tested); `invert_learned_bfgs` hands the squared-residual cost
to `optim::minimize_smooth_2d` warm-started from grid argmin.
**Type-alias API fix (load-bearing for future module ports):**
`ScoreFn` / `ForwardMap` type aliases are now documentation-only;
public signatures inline `&dyn Fn(...)` directly. Reason:
`type Alias = dyn Fn(...)` defaults to `dyn Fn(...) + 'static`,
which locks callers to `'static` closures and blocks any closure
that captures local state (e.g. the camera-test `forward_map`
that captures a call counter). Inlining lets the trait-object
lifetime default to the enclosing reference's lifetime per Rust
reference §"Default trait object lifetimes". Apply the same
inlining pattern to any future `&dyn Trait` parameter. **87/87
Rust tests** green (31 src unit including 3 new optim + 9 new
gradient-inversion + 21 bit-identity + 17 analytic math + 18
types smoke). No new bit-identity fixtures — session 5 composes
existing math primitives; cross-language verification against
Python's `test_learned_field.py::TestForwardSweepInvertLearned`
recovery set defers to the session that ports `cross_substrate.py`
/ `active_learning.py` (they instantiate learned fields with
materialized weights).


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
- **v4**: streaming inversion + symbolic query DSL + notebook
  ergonomics (`_repr_html_` + selective `__repr__` overrides) +
  default plot hooks (matplotlib + lazy plotly). New modules:
  `streaming` (`InversionResult` + `forward_sweep_invert_stream` +
  thin source adapters), `symbolic_query` (5-pattern DSL),
  `plotting` (4 default plot helpers). Seven-op API unchanged.
  (Shipped 2026-05-16; 338 tests green.)
- **v5**: continuous Banach self-test cadence + sensitivity backprop
  + gradient-based inversion. New modules: `self_test`
  (`BanachDriftReport` + `SelfTestCadence` + `run_banach_self_test`;
  streaming cadence hook on `forward_sweep_invert_stream`),
  `sensitivity` (trajectory + per-frame Jacobian composers + the
  `driver_profile_loss_grad` one-liner). `forward_sweep_invert`
  gains the `method` kwarg dispatching tangent_flow → closed-form,
  learned → L-BFGS, lookup_table → grid. Seven-op API surface
  unchanged. (Shipped 2026-05-16; 392 tests green.)
- **v6**: native port to **Rust + WASM** (browser is the load-bearing
  consumer per user direction). `jax_core.py` is the math source the
  port reads; the v0/v1 operations are wrapper-shape only. Matches the
  v5 Python under per-seed reproducibility. Zero new features.
  - *Session 1 (2026-05-16):* toolchain bootstrapped; `rust/`
    scaffolded; `rust/src/math.rs` ported from `jax_core.py` (all 12
    primitives). `cargo build` (native + wasm32) clean;
    `cargo test --release` green (17/17 analytic sanity tests).
  - *Session 2 (2026-05-16):* bit-identity test infrastructure
    landed. Emitter at `rust/tests/fixtures/emit_jax_core_reference.py`
    walks all 12 `jax_core` primitives over a small input sweep
    (48 cases, 22 KB JSON committed at
    `rust/tests/fixtures/jax_core_reference.json`). Rust integration
    test at `rust/tests/bit_identity.rs` consumes the JSON via
    `serde_json` (first dev-dependency) and asserts each Rust
    primitive in `src/math.rs` reproduces the Python output within
    per-primitive ULP budgets (LIBM = 4 ULPs for primitives
    composing a small number of libm calls; LIBM_WIDE = 16 ULPs
    for primitives composing many libm calls or sums whose
    JAX-pairwise vs Rust-sequential reduction order can differ).
    `cargo test --release` green: **13/13 bit-identity +
    17/17 analytic = 30/30 total.** Next: port `types.py` then
    `operations.py`; extend the bit-identity infrastructure with
    fixture entries per new module.
  - *Session 3 (2026-05-16):* `types.py` → `rust/src/types.rs`.
    17 structs + 5 enums; `Activation` / `MlpLayer` re-exported
    from `math` so `LearnedField.weights` passes directly into
    `math::learned_field_substrate`. Naming divergence:
    `TranslationField` is now the Rust tagged enum (Python's
    `AnyTranslationField`); `LookupTableField` is the lookup-table
    struct (Python's `TranslationField`); serde tag is `shape`.
    `SidecarKey([u64; 3])` with custom serde to ':'-joined string
    so it works as a JSON map key. `_repr_html_` / `__repr__`
    overrides do not port (Python display only). `serde` +
    `serde_json` promoted to runtime deps. New smoke test at
    `rust/tests/types_smoke.rs` (18 tests) — round-trip every
    public type. `cargo test --release` green: **13/13
    bit-identity + 17/17 analytic + 18/18 types smoke = 48/48
    total**; Python 392/392 still green; WASM build still clean.
    Next: port `operations.py`. Details in
    [`docs/BLOCK_IN.md`](docs/BLOCK_IN.md) §v6.
  - *Session 4 (2026-05-16):* `operations.py` **raw forward path**
    plus three small deps. Four new modules:
    `rust/src/gfdr_model.rs` (5 pure functions: `vertex_regime`,
    `alpha_s`, `plateau_height`, `generate_locus`, `interp_locus`,
    `locus_residual`; `LocusPoint` + `EmpiricalRow` typed shapes),
    `rust/src/sidecar.rs` (`round_key` via
    `(x*10^n).round_ties_even()/10^n` matching Python for the bulk
    of inputs; cross-language wire-format parity for sidecars
    Python-emitted-Rust-consumed still deferred per BLOCK_IN),
    `rust/src/flow.rs` (banach_exp / generic / Caputo dispatch
    on the `serde_json::Value`-typed refinement map),
    `rust/src/operations.rs` (raw forward path:
    `TranslationFieldIndex`, `apply_translation` + 3 helpers,
    `forward_sweep_invert_grid` — grid path only, gradient
    dispatch lands session 5 — `tau_obs_sweep_grid`, `regime_at`,
    `regime_display_band`, `gamut_classify` returning a typed
    `GamutClassification`). `score_fn` / `forward_map` Python
    kwargs surface as `Option<&dyn Fn(...)>`.
    `default_substrate_score` sorts intersected keys (BTreeMap)
    for deterministic float-sum; Python's hash-randomized
    set iteration is covered by `LIBM_WIDE` in the bit-identity
    tests. **75/75 Rust tests pass** (19 src unit + 21
    bit-identity + 17 math + 18 types_smoke); Python 392/392
    still green; WASM build still clean. Session-2 fixture
    lesson load-bearing twice: `sidecar_round_key` fixture
    excludes `.x5`-decimal inputs (Python `round` vs Rust
    `round_ties_even` disagree off the exact binary halfway);
    `gfdr_locus_residual` empirical rows are SYNTHETIC, not
    generated from `gfdr_model.generate_locus`, to dodge the
    candidate=truth cross-impl cancellation that collapses
    Python's residual to exact 0 but leaves Rust's at ~5e-33.
    Next: gradient inversion (`method="auto" / "gradient"` via a
    native L-BFGS optimizer — session 5). Five-session
    operations.py-port-completion breakdown lives in
    [`docs/BLOCK_IN.md`](docs/BLOCK_IN.md) §v6.
  - *Session 5 (2026-05-16):* `forward_sweep_invert` **gradient
    inversion dispatcher** lands. New `rust/src/optim.rs` carries
    `minimize_smooth_2d` — a hand-rolled 2D damped-Newton solver
    (numerical FD gradient + Hessian + backtracking line search;
    Hessian inversion via `math::inv_2x2`). Substitutes Python's
    `scipy.optimize.minimize(method="L-BFGS-B")` + `jax.grad`.
    Deliberate divergence from the BLOCK_IN-noted `argmin` candidate
    (justified at module-doc resolution): problem is 2D, BLOCK_IN
    carves out non-byte-identity vs scipy, ~80 LOC beats the dep
    footprint for compile-time / WASM-size / API-surface costs. New
    `Method::{Auto, Grid, Gradient}` enum + `InversionResult` struct
    (residuals optional — closed-form skips grid) +
    `forward_sweep_invert` dispatcher in `operations.rs`. Closed-form
    path wraps `math::tangent_flow_canonical_inverse` (session-1
    bit-identity tested); L-BFGS-equivalent path warm-starts from
    grid argmin. New `OperationError::GradientOnLookupTable` mirrors
    Python's `ValueError`. **Type-alias API fix (forward-relevant):**
    `ScoreFn` / `ForwardMap` aliases become documentation-only;
    public signatures inline `&dyn Fn(...)` so the trait-object
    lifetime defaults to the enclosing reference's, allowing local-
    state closures (a `type Alias = dyn Trait` defaults to
    `dyn Trait + 'static` which locks callers to `'static`). Apply
    this inlining pattern to any future `&dyn Trait` parameter on the
    Rust port. **87/87 Rust tests pass** (31 src unit including
    3 new optim + 9 new gradient-inversion + 21 bit-identity + 17
    math + 18 types_smoke); Python 392/392 still green; WASM build
    still clean. No new bit-identity fixtures — session 5 composes
    existing math primitives. Cross-language convergence vs Python's
    `test_learned_field.py::TestForwardSweepInvertLearned` recovery
    set deferred to whichever session ports `cross_substrate.py` /
    `active_learning.py` (they instantiate learned fields with
    materialized weights). BanachSubstrate-based test deferred —
    `banach.py` belongs to the curator-path port in mpa-conform.
    Wrapped-variant test deferred — wrapped variants land in session 7
    with validation + provenance. Next per BLOCK_IN: session 6 —
    intent algebra. Details in
    [`docs/BLOCK_IN.md`](docs/BLOCK_IN.md) §v6.

Each is its own session, sequenced by the user via
`mpa-conform/docs/ROADMAP.md`.

## Acceptance for the v1 build session (handoff §E)

All twelve items met as of 2026-05-16. (Detail moved to README session
log; the v1 acceptance contract is locked.)

## Acceptance for the v4 build session

Four items met as of 2026-05-16:

1. v0 + v1 + v2 + v3 fixtures pass unchanged (286 prior tests green
   plus 52 new v4 tests: 14 streaming, 13 symbolic-query, 25
   notebook-repr / plot helpers). 338 tests total green.
2. Streaming surface (`mpa_scale_solver.streaming`): `InversionResult`
   + `forward_sweep_invert_stream` generator + thin source adapters
   (`from_iterable`, `from_stdin`). State-local per frame; lazy
   generator; supports per-frame `tau_obs` via `tau_obs=None`.
3. Symbolic-query DSL (`mpa_scale_solver.symbolic_query`): five
   patterns, regex-parsed, closed-form expressions returned for
   tangent_flow translate / invert / tau-crossing; bisection
   fallback for non-tangent-flow tau-crossing via `tau_range` kwarg.
4. Notebook ergonomics: `_repr_html_` on 11 user-facing dataclasses;
   `__repr__` overridden on Posterior / LearnedField /
   OperationOutput to suppress nested-tuple / weight-matrix dumps.
   `plotting.py` ships four default plot helpers
   (`plot_trajectory`, `plot_gamut`, `plot_residual_field`,
   `plot_posterior`) with matplotlib (lazy) and plotly (optional).
   README + CLAUDE.md updated; BLOCK_IN §v4 deleted; §v5 refined
   with the streaming-side self-test cadence note (cadence k applies
   per emitted frame).

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

## Acceptance for the v5 build session

Five items met as of 2026-05-16:

1. v0 + v1 + v2 + v3 + v4 fixtures pass unchanged (338 prior tests
   green plus 54 new v5 tests: 20 continuous-self-test, 18
   sensitivity-backprop, 16 gradient-inversion). 392 tests total
   green.
2. Continuous self-test (`self_test.py`): `SelfTestCadence(k=100)`
   default; per-tick `BanachDriftReport` agrees with the analytical
   Banach truth at `<= DRIFT_TOLERANCE = 1e-10`; streaming hook
   advances per emitted frame; state-locality verified (with-cadence
   recovery byte-identical to without-cadence at `k=1`).
3. Sensitivity backprop (`sensitivity.py`): five entries
   (`trajectory_substrate_diff`, `trajectory_substrate_jacobian`,
   `field_parameter_sensitivity`, `inversion_sensitivity`,
   `driver_profile_loss_grad`). Closed-form chit/delta-chit
   sensitivity verified against `log(tau/tau_ref)`; gradient direction
   verified as loss-descent on a synthetic mis-specified field.
4. Gradient-based inversion: `forward_sweep_invert(..., method="auto"|
   "grid"|"gradient")`. `"auto"` (new default) routes tangent_flow
   to closed-form (sub-grid-resolution recovery; `< 1e-12` per axis
   on the Banach camera test where v4 was at `< 0.001`), learned to
   L-BFGS (sub-grid recovery on the identity learned-field test),
   lookup_table to grid (v4 byte-identical). Wrapped variant accepts
   the same kwarg. `forward_map=` override forces grid;
   `return_residual_field=True` always evaluates grid plus the
   method-chosen best_state. `method="grid"` byte-identity preserved
   across all 338 prior tests.
5. README + CLAUDE.md updated; BLOCK_IN §v5 deleted; §v6 refined
   with the v5 surface (self-test + sensitivity + method-dispatch)
   that the native port must reproduce. Tagged `v5.0.0`.

## Session handoff

The v0→v6 trajectory is governed by the **self-evolving block-in
handoff** at [`docs/BLOCK_IN.md`](docs/BLOCK_IN.md). Each session that
lands a version deletes its own §vN section from that doc and refines
the remaining sections in place. Historical "what shipped" stays in
this repo's `README.md` § Session Log. As of 2026-05-16 the block-in
carries v6 only (v1–v5 shipped).

When opening a scale-solver session, read [`docs/BLOCK_IN.md`](docs/BLOCK_IN.md)
§vN for the version being built. Read [`docs/NORTH_STAR.md`](docs/NORTH_STAR.md)
for the destination context.
