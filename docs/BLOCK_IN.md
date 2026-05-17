# mpa-scale-solver — v6 block-in handoff

Self-evolving trajectory handoff. v1–v5 shipped 2026-05-16: v1.0.0
(continuous flow + Banach + sidecar), v2.0.0 (JAX foundation, cut a),
v2.1.0 (Bayesian inversion via Laplace, cut b), v2.3.0 (full I1–I5
intents + composition, cut d), v2.4.0 (non-Markovian Caputo flow via
Prony, cut e), v3.0.0 (cross-substrate ops + active learning + MCP
server + LearnedField + per-intent RFC-S §5 metrics), v4.0.0
(streaming inversion + symbolic-query DSL + notebook ergonomics +
default plot hooks), v5.0.0 (continuous Banach self-test cadence +
sensitivity backprop + gradient-based inversion). Cut (c) "N-mode
generalization" cancelled 2026-05-16 — premise overturned by
framework cross-check; the 2-mode CanonicalState is the framework's
universal canonical representation, not a 2-mode-as-N=2-special-case
(see §v2.2-cancelled tombstone below).
What remains: v6.

This document is the single brief that carries from one session to the
next. It is **not** a roadmap — sequencing lives in
[`mpa-conform/docs/ROADMAP.md`](ROADMAP.md). It is the structural framing
each session reads cold.

---

## How this document evolves

Each session that lands a version exits by:

1. **Deleting its own §vN section.** The version that just shipped is no
   longer remaining work. Its detail moves to
   [`mpa-scale-solver/README.md`](https://github.com/ronviers/mpa-scale-solver/blob/main/README.md)
   § Session Log (one row per session, append-only — that is the
   historical record).
2. **Refining the remaining §vM sections** based on what the just-
   shipped session learned. New constraints surface, scopes tighten,
   dependencies clarify. The refinements are inline edits, not append
   notes — the document stays current, not annotated.
3. **Updating §Cross-cutting discipline** if a new program-wide
   constraint emerged (rare; most refinements are local).

When v6 ships, this document deletes entirely. The README session log
+ the per-version git tags are the residue.

**Discipline:** the block-in shrinks. If a session feels the urge to
*add* a section, ask whether the addition is a new version (rare — the
trajectory is fixed) or a refinement of an existing section (almost
always — refine in place).

---

## Trajectory at a glance

| Version | Theme | Depends on |
|---|---|---|
| v1–v5 | Foundation, JAX, Bayesian, intents, Caputo, cross-substrate, active learning, MCP, LearnedField, streaming + DSL + notebook ergonomics, continuous self-test + sensitivity backprop + gradient inversion — all shipped 2026-05-16 (see README §Session Log) | — (shipped) |
| ~~v2.2~~ | ~~N-mode generalization (cut c)~~ — cancelled 2026-05-16; tombstone retained below | — |
| **v6** | One-shot native port (Rust + WASM). Zero new features. Per-seed reproducibility against the v5 Python. Sessions 1-7 landed 2026-05-16 (math, types, raw forward path, gradient dispatcher, intent algebra, validation+provenance+wrapped); sessions 8-9 remaining (posterior, bindings). | v5 |

Sequencing is the user's call per ROADMAP. v6 is the only remaining
version; the API surface and capability set are frozen as of v5.

---

## Cross-cutting discipline (applies to every session below)

**Seven-operation API stays stable.** v1's
`apply_translation`, `forward_sweep_invert`, `tau_obs_sweep`,
`regime_at`, `gamut_classify`, `intent_map`, `validate_driver_profile`
+ `flow` — the surface is right. What grows is *capability depth* and
*backends*. Eighth operation requires a foundational-questions entry.

**v0/v1 fixture regression passes every session.** Every release
re-runs the v0 + v1 fixture suites unchanged. Behavior changes require
a separate release with a fresh captured fixture, called out in the
commit. `tests/test_fixtures.py`, `tests/test_banach_camera.py`,
`tests/test_camera_migration.py`, `tests/test_seed_corpus.py`,
`tests/test_sidecar.py`, `tests/test_validation.py`,
`tests/test_provenance.py` — all green or the session is blocked.

**Wrapped variants are the consumer surface.** mpa-conform calls
`*_wrapped` functions returning `OperationOutput[T]`. New capabilities
land on the wrapped path; raw v0 sigs preserve back-compat. Validation
+ provenance schemas already in v1's `ValidationReport` / `Provenance`
dataclasses — extend their fields, don't fork new types.

**Python is the pseudo-code spec for v6.** Every Python design choice
should map cleanly to native: frozen dataclasses, free functions,
value semantics in/out, no callbacks, no hidden state, tagged dispatch
via `.shape` fields (not duck-typing). v6's port reads v5's Python as
its reference; bit-identity is the per-seed reproducibility target.

**No anti-goals.** Trajectory integration → mpa-solver. Bundle
orchestration → mpa-conform. Display → mpa-auditor. RFC text →
mpa-atlas. If a session reaches for one of these, stop and rebalance.
See [`../CLAUDE.md`](../CLAUDE.md)
and [`NORTH_STAR.md` §Anti-goals](NORTH_STAR.md).

**Consumer touch-up belongs in the consumer.** mpa-conform's
walk_library + inversion currently use v0/v1 surface. If a version
unlocks a richer call (e.g., v2's Bayesian inversion enables posterior
recording in `fit_provenance`), that conform-side rewire is a separate
follow-on session, not part of the scale-solver release.

---

## §v2.2-cancelled — N-mode generalization (cut c)

Cancelled 2026-05-16 by user after framework cross-check. Kept as a
tombstone (short, not refined further) so a future session does not
re-propose the same shape under the "block-in shrinks" discipline.

**Why cancelled.** The 2-mode `(chit, gamma_AB, k_frust)`
CanonicalState IS the framework's universal canonical representation
by deliberate design — not a 2-mode-as-N=2-special-case shape. N≥3
substrates decompose into four pattern-selection tests, not a wider
state. `k_frust: bool` is already the framework's existing N≥3
frustration-test marker on the existing dataclass; `regime_at` /
`intent_map` already propagate it.

**Pointers (mpa-atlas):** `cdv1_compressed.md` §"Universal two-mode
kernel" (line 125); §10 "Four-channel pattern selection
architecture" (line 493, quote: *"any N≥3 Character kernel routes
through these four tests"*); line 150 (`k_frust` as the N≥3 marker).
`cdv1_receipts.md` §9 / §16 / §17 / §18 / §21 (every N≥3 result —
May–Leonard, Schnakenberg cycles, non-reciprocal kernels, chimera
states — composes the two-mode kernel pairwise; no wider state
appears).

**Re-opening conditions.** A concrete N≥3 measurement need that the
four-channel tests don't already cover, surfaced by a substrate.
Re-opening would land as a new BLOCK_IN section under a new cut
identifier, framed as the four-channel tests as operations consuming
a `ModePairGraph` of 2-mode states — NOT as widening CanonicalState.
A foundational-questions entry in
`H:/mpa-scale-solver/docs/foundational-questions.md` precedes any
such re-opening per CLAUDE.md's no-eighth-operation rule.

---

## §v6 — Native port (Rust + WASM). Zero new features.

**Language pick (2026-05-16):** **Rust**, browser is the load-bearing
consumer per user direction. Rust → WASM via `wasm-bindgen` /
`wasm-pack`; same Rust source produces pyo3-backed Python bindings so
mpa-conform's `import mpa_scale_solver` keeps working unchanged.

**Session log (in §v6 — deletes when v6 ships).**
- *Session 1 (2026-05-16):* Toolchain bootstrapped (rustup + stable
  1.95 + wasm32-unknown-unknown). `rust/` scaffolded (single crate;
  workspace split deferred until bindings shake out). `src/math.rs`
  ported from `mpa_scale_solver/jax_core.py` — all 12 primitives:
  tangent_flow_substrate / banach_state / tangent_flow_canonical /
  lookup_squared_distance / tangent_flow_canonical_inverse /
  tangent_flow_inversion_residual / Laplace 2x2 covariance pair +
  log evidence / caputo_flow / mlp_forward / learned_field_substrate.
  `cargo build --release` (native rlib) and
  `cargo build --release --target wasm32-unknown-unknown` both clean;
  `cargo check --tests --all-targets` clean. Analytic sanity tests in
  `rust/tests/math.rs` (round-trip identity, β=1 Caputo=Banach
  exponential, 2x2 inverse identity, log-ratio clamp at degenerate
  tau) — **17/17 pass** after MSVC VC Tools workload landed via
  GUI Modify. Doctests disabled (`doctest = false` in Cargo.toml)
  since the /// blocks carry math notation, not runnable Rust.
- *Session 2 (2026-05-16):* Bit-identity test fixtures vs Python
  landed. Emitter at `rust/tests/fixtures/emit_jax_core_reference.py`
  walks all 12 `jax_core` primitives over a small input sweep (48
  cases, 22 KB JSON), reading the Python output and writing
  `rust/tests/fixtures/jax_core_reference.json`. The fixture is
  committed; the emitter regenerates byte-identically when `jax_core`
  hasn't changed. Rust integration test at `rust/tests/bit_identity.rs`
  loads the JSON via `serde_json` (added as the first dev-dependency)
  and asserts each Rust primitive in `src/math.rs` reproduces the
  Python output within a per-primitive ULP budget: **LIBM = 4 ULPs**
  for primitives composing a small number of libm calls
  (tangent_flow_*, banach_state, lookup_squared_distance, laplace
  covariances), **LIBM_WIDE = 16 ULPs** for primitives composing
  many libm calls or sums whose JAX-pairwise vs Rust-sequential
  reduction order can differ (caputo_flow, mlp_forward,
  learned_field_substrate, laplace_log_evidence). A coverage-guard
  test asserts the fixture lists all 12 primitives so the emitter
  can't silently drop one on regeneration. **13/13 bit-identity
  tests pass; 17/17 analytic tests still pass (30/30 total).**
  Fixture-design note for next session: do NOT generate `target =
  python_forward(candidate)` and then ask Rust to compute
  `residual = (rust_forward(candidate) - target)^2` — the libm
  cancellation makes Rust's residual ~1e-32 instead of Python's
  exact 0, which is not a porting bug but does fail any ULP
  tolerance. Specify all `(candidate, target)` pairs explicitly
  so both implementations evaluate the same numerical inputs.
- *Session 4 (2026-05-16):* Port operations.py's **raw forward path**
  plus its three small dependencies. Four new Rust modules:
  `rust/src/gfdr_model.rs` (5 pure functions ported from
  `mpa_scale_solver/gfdr_model.py` — `vertex_regime`, `alpha_s`,
  `plateau_height`, `generate_locus`, `interp_locus`, `locus_residual`;
  `LocusPoint` + `EmpiricalRow` structs for the typed JS-port shapes),
  `rust/src/sidecar.rs` (`round_key` via `(x*10^n).round_ties_even()/10^n`
  matching Python's banker's rounding for the bulk of inputs;
  `lookup_inverse` / `lookup_forward` as `BTreeMap<SidecarKey, _>::get`;
  cross-language wire-format parity still deferred per the BLOCK_IN
  rule), `rust/src/flow.rs` (`flow` dispatching banach_exponential /
  generic / Caputo via `serde_json::Value`-typed refinement), and
  `rust/src/operations.rs` carrying `TranslationFieldIndex` +
  `apply_translation` + the three field-shape helpers +
  `forward_sweep_invert_grid` (grid path only; method=auto/gradient
  deferred to session 5) + `tau_obs_sweep_grid` + `regime_at` +
  `regime_display_band` + `gamut_classify` (returns typed
  `GamutClassification` / `GamutDiagnosis`, not a dict). `score_fn` /
  `forward_map` callable kwargs surface as `Option<&dyn Fn(...)>` — no
  type-parameter gymnastics at the call site. `default_substrate_score`
  iterates intersected keys in sorted (BTreeMap) order so the float-sum
  is deterministic across Rust runs; Python's hash-randomized set
  iteration produces ULP-level divergence covered by `LIBM_WIDE` in the
  bit-identity tests. Bit-identity infrastructure extended: 8 new
  fixture entries in `emit_jax_core_reference.py` (`gfdr_alpha_s`,
  `gfdr_plateau_height`, `gfdr_vertex_regime`, `gfdr_generate_locus`,
  `gfdr_interp_locus`, `gfdr_locus_residual`, `sidecar_round_key`,
  `flow`); 8 corresponding `#[test]` functions in `bit_identity.rs`;
  coverage-guard list bumped from 12 to 20 primitives. Two design
  lessons from session 2 paid off again: (1) `sidecar_round_key`
  fixture excludes `.x5`-decimal inputs whose binary representation
  shifts the value off the exact halfway (Python `round`'s dtoa path
  and Rust's `round_ties_even` disagree there — documented in
  `sidecar.rs` as a cross-language wire-format caveat); (2)
  `gfdr_locus_residual` empirical rows are SYNTHETIC invented values,
  NOT generated from `gfdr_model.generate_locus` — otherwise the
  candidate=truth self-residual collapses to exact 0 in Python and
  ~5e-33 in Rust (cross-impl libm cancellation), failing any ULP
  budget. **75/75 Rust tests pass** (19 src unit + 21 bit-identity +
  17 math + 18 types_smoke); Python 392/392 still green; `cargo build
  --release --target wasm32-unknown-unknown` still clean.
- *Session 3 (2026-05-16):* Port `types.py` → `rust/src/types.rs`.
  5 enums (`Direction`, `Gt`, `RegimeLabel`, `DisplayBand`,
  `DispatchPath`) plus `Activation` re-exported from `math`.
  17 structs mirroring the Python `@dataclass(frozen=True)` shapes
  (`CanonicalState`, `SubstrateState`, `CanonicalPoint`,
  `OperatingPoint`, `TranslationRule`, `LookupTableField`,
  `ScalingRule`, `TangentFlowField`, `LearnedField`, `GamutSpec`,
  `RegimeReading`, `Provenance`, `ValidationReport`,
  `OperationOutput<T>`, `Posterior`, `SidecarKey`,
  `InverseLookupSidecar`). **Naming divergence from Python:**
  Python's `TranslationField` is the lookup-table struct; Rust uses
  `LookupTableField` for the struct and `TranslationField` for the
  tagged enum `{LookupTable, TangentFlow, Learned}` (Python's
  `AnyTranslationField`), with serde `tag = "shape"` matching the
  Python `.shape` field discriminator. `dict[str, Any]` payloads
  (`axes` / `extras` / `refinement` / `ambiguity_regions`) →
  `BTreeMap<String, serde_json::Value>` (ordered for byte-stable
  serialization). `dict[str, float]` (`observables`) →
  `BTreeMap<String, f64>`. `SidecarKey` is a `[u64; 3]` newtype
  wrapping `f64::to_bits` of the rounded floats, with a custom
  `Serialize`/`Deserialize` to a `':'`-joined string so it works as
  a JSON map key (JSON requires string keys); chosen as the thin
  minimal default — `sidecar.rs` is free to pick a Python-parity
  wire format if cross-language sidecar I/O lands. `_repr_html_` /
  `__repr__` overrides are Python display-only (Jupyter / REPL)
  and do not port. `serde` + `serde_json` promoted from dev-deps
  to runtime deps. New smoke test at `rust/tests/types_smoke.rs`
  (18 tests): constructs every public type, round-trips via
  `serde_json`, asserts equality — catches derive misconfiguration
  and validates the `TranslationField` enum's `shape` tag.
  **48/48 Rust tests pass** (13 bit-identity + 17 math + 18 types
  smoke); Python 392/392 still green; `cargo build --release
  --target wasm32-unknown-unknown` still clean. Cross-language
  JSON parity (Python writes → Rust reads, field-by-field equality)
  remains deferred — it lands when the first module with actual
  JSON I/O ports (`sidecar.py` or `mcp_server.py`); `types.py`
  itself has no JSON producers in the Python.
- *Session 5 (2026-05-16):* `forward_sweep_invert` **gradient
  inversion dispatcher** completes the `operations.py` method-dispatch
  surface (Python's `method="auto" / "grid" / "gradient"`). New
  `rust/src/optim.rs` (`minimize_smooth_2d`): hand-rolled 2D damped-
  Newton solver with numerical finite-difference gradient + Hessian
  + backtracking line search; Hessian inversion via `math::inv_2x2`
  from session 1. **Deliberate divergence from the BLOCK_IN-noted
  `argmin` candidate** (sanctioned at session time): the problem is
  2D, this BLOCK_IN explicitly carves out non-byte-identity vs scipy
  (the optimizer just needs to converge to the same MAP within
  ~0.005 per axis from the grid-argmin warm start), and ~80 LOC of
  hand-rolled code beats pulling `argmin + argmin-math` for the
  compile-time / WASM-size / API-surface costs on a 2D problem with
  a smooth near-quadratic cost. Newton converges in 2-3 outer
  iterations on the identity-MLP test. `operations.rs` adds
  `Method::{Auto, Grid, Gradient}` enum + `InversionResult` struct
  (residuals optional; closed-form skips grid) + `forward_sweep_invert`
  dispatcher with the Python `method`-kwarg routing (`Auto` →
  closed-form/L-BFGS-equivalent/grid per field shape; `Gradient`
  errors on `LookupTable` via new `OperationError::GradientOnLookupTable`).
  Closed-form path wraps `math::tangent_flow_canonical_inverse`
  (session-1 bit-identity tested); L-BFGS-equivalent path warm-starts
  from the grid argmin. **Type-alias API fix (forward-relevant):**
  `ScoreFn` / `ForwardMap` aliases become documentation-only;
  public signatures inline `&dyn Fn(...)` directly. Reason:
  `type Alias = dyn Fn(...)` defaults to `dyn Fn(...) + 'static`,
  locking callers to `'static` closures (and breaking the camera-test
  `forward_map` capturing a call counter). Inlining lets the trait-
  object lifetime default to the enclosing reference's per Rust
  reference §"Default trait object lifetimes". Apply this inlining
  pattern to any future `&dyn Trait` parameter on the Rust port.
  **87/87 Rust tests pass** (31 src unit including 3 new optim +
  9 new gradient-inversion + 21 bit-identity + 17 math + 18
  types_smoke); Python 392/392 still green; `cargo build --release
  --target wasm32-unknown-unknown` still clean. No new bit-identity
  fixtures — session 5 composes existing math primitives.
- *Session 6 (2026-05-16):* Intent algebra port — `intent_map` +
  `intent_compose` + five `_intent_iN` handlers + helpers. New
  `types.rs` additions: `IntentId` enum (I1–I5), `CapacityClass` enum
  (`Deep`/`Shallow`), `SacrificeRecord` struct + `IntentDiagnostics`
  tagged enum implementing the BLOCK_IN-prep sketch exactly: three
  truly-common fields (`invariant_preserved`, `delta_chit`,
  `delta_gamma_AB`) on the outer struct, intent-specific fields typed
  per-variant, `#[serde(flatten)] + #[serde(tag = "intent")]` so the
  JSON wire format is a flat dict matching Python's `sac`-output shape.
  `intent` and `preserved_invariant` are derived methods (not stored)
  per the BLOCK_IN-prep rationale — they're statically determined by
  the variant. Two new `operations.rs` errors: `IntentComposeEmpty` +
  `I2InComposition` mirroring Python's two ValueErrors. `intent_map`
  takes `IntentId` directly (not a string) — Python's "unknown intent"
  runtime error becomes a compile-time impossibility. Helpers
  (`sign_i`, `capacity_class`, `clamp_to_gamut`,
  `regime_chit_interval`, `nearest_in_gamut_chit_for_regime`,
  `sign_preserving_clamp`) port 1:1 from Python's `operations.py`
  except the regime intervals are typed (match on `RegimeLabel`) and
  the `_REGIME_CHIT_INTERVALS` dict becomes a function. The I3 deep→
  shallow recovery branch (try-the-same-side-endpoint logic) ports
  verbatim. **26 new src unit tests** mirror `tests/test_intents.py`'s
  TestI1RegimePreserving / I2 / I3 / I4 / I5 / TestComposition classes
  (TestValidation defers to session 7 alongside wrapped variants —
  same deferral logic as session 5's wrapped-test skip). One
  serde-flatten round-trip smoke catches the
  `#[serde(flatten)] + tag="intent"` combination. **113/113 Rust tests
  pass** (57 src unit — up from 31, +26 intent — + 21 bit-identity + 17
  math + 18 types_smoke); Python 392/392 still green; `cargo build
  --release --target wasm32-unknown-unknown` still clean. No new
  bit-identity fixtures — session 6 is pure arithmetic, no math
  primitives added (predicted at BLOCK_IN-prep time; confirmed). One
  schema-parity fixture added: `sacrifice_record` (13 cases covering
  all 5 intents + edge cases) in `jax_core_reference.json`; new test
  `sacrifice_record_python_to_rust_json_parity` in `bit_identity.rs`
  deserializes each Python-emitted `sac` dict into Rust `SacrificeRecord`
  and asserts field-by-field agreement (common fields, derived
  `.intent()` / `.preserved_invariant()`, per-variant diagnostics).
  **Asymmetric parity is now the documented design**: Python's stored
  `preserved_invariant` STRING is silently dropped on Python→Rust
  read (serde default behavior); Rust's `.preserved_invariant()`
  reconstructs it byte-for-byte from the variant. The test asserts
  the reconstructed string equals Python's emitted string, so
  symmetric round-trip is a one-line custom Serialize away should
  a future consumer ever need it (no consumer needs it as of session
  6 — Python is the producer in the wrapped-variant path). Final
  test totals: **114/114 Rust tests pass** (57 src unit + 22
  bit-identity — was 21, +1 sacrifice parity + 17 math + 18
  types_smoke); Python 392/392 still green; WASM build still clean.
- *Session 7 (2026-05-16):* **Validation + provenance + the eight
  `*_wrapped` variants** + raw `validate_driver_profile`. Two new
  modules: `rust/src/provenance.rs` (`make_provenance` +
  `provenance_hash` + `SOLVER_VERSION` const = `"5.0.0"`, tracking
  Python's `__version__`; cross-language hash parity is the
  load-bearing contract here — the Cargo crate version is
  deliberately decoupled), and `rust/src/validation.rs` (all 14
  functions from `validation.py`: 3 checkers, 6 report builders,
  per-intent RFC-S §5 metric + aggregator, bitfield encoder, plus
  the typed `DriverProfileSummary` struct returned by
  `validate_driver_profile`). `operations.rs` extended with the
  raw `validate_driver_profile` + the 8 `*_wrapped` operations
  exactly mirroring Python's `operations.py` lines 1316–1568;
  sidecar dispatch is opt-in via `Option<&InverseLookupSidecar>`
  on the three ops that have it (apply_translation,
  forward_sweep_invert, tau_obs_sweep), matching Python. Per-intent
  metric output is `BTreeMap<String, serde_json::Value>` (not a
  parallel typed enum) — Python's `dict[str, Any]` is the consumer
  contract and inventing a typed wrapper would force callers to
  translate at the JSON boundary. **`timestamp_ns` discipline:**
  Python uses `time.monotonic_ns()` (the BLOCK_IN session-7-prep
  description said `time.time_ns()` — minor doc inaccuracy, doesn't
  affect the port since timestamps are excluded from
  `provenance_hash`); Rust uses a `OnceLock<Instant>` process-start
  epoch and computes `epoch.elapsed().as_nanos()` — monotonic,
  process-local, never bit-identity-compared. **Two new
  bit-identity fixtures:** (1) `provenance_hash` (12 cases covering
  every `DispatchPath` variant + null vs non-null `table_version`)
  — stores the **raw 4-byte blake2b digest as hex** and Rust
  compares digest bytes directly via the new public
  `provenance::provenance_digest_bytes`. The initial commit stored
  the float view + compared bits, which forced enabling
  `serde_json/float_roundtrip` to fight a 1-ULP JSON drift; that
  workaround was retired the same session in favor of the cleaner
  digest-hex contract — the float-in-JSON form was the wrong shape
  for a rational hash. See the new "fixture discipline" Open/watch
  entry. (2) `operation_output_regime_at` (4 cases) — the
  representative wrapped-variant wire parity for
  `OperationOutput<RegimeReading>`; timestamps + note strings
  excluded by documented asymmetric design (timestamps
  non-deterministic; notes have a Python/Rust float-format
  divergence — `f"{0.0}"` is `"0.0"` in Python,
  `format!("{}", 0.0)` is `"0"` in Rust). `round_trip_residual`
  comparison uses `LIBM_WIDE` ULP tolerance (preemptive — the slot
  is `None` for regime_at but session 8's posterior parity will
  populate it). New dep: `blake2 = "0.10"` (default-features off)
  for `provenance_hash`'s `blake2b-32` digest. The thin-discipline decision on
  `make_provenance` was explicit args (no kwargs / no builder) to
  match the existing `operations.rs` style — Rust callers pass
  `make_provenance("regime_at", DispatchPath::DirectCompute, None, vec![])`
  verbosely. **167/167 Rust tests pass** (108 src unit — up from
  57, +30 validation + 8 provenance + 13 wrapped-variant tests —
  + 24 bit-identity — was 22, +1 provenance_hash + 1
  operation_output_regime_at + 17 math + 18 types_smoke); Python
  392/392 still green; `cargo build --release --target
  wasm32-unknown-unknown` still clean. The session-2 fixture lesson
  did not fire — both new fixtures specify all inputs and outputs
  explicitly (provenance_hash takes operation/dispatch/table
  literals; operation_output_regime_at takes canonical literals).
  The session-6 schema-parity pattern extended cleanly to the
  wrapped-variant wire format; the "asymmetric parity is documented
  design" framing covers both timestamps and the note-string
  float-format divergence.

**Goal.** One-shot consolidation. Match the proven v5 Python under the
per-seed reproducibility discipline. Free to parallelize aggressively
(Rayon / WASM web-workers / GPU); Python's single-thread constraints
do NOT carry forward. WASM bindings are first-class (mpa-auditor
consumes).

**Capabilities to port (exactly the v5 Python surface).**

- ~~Math primitives (`jax_core.py`)~~ — landed session 1 at
  `rust/src/math.rs`.
- ~~Seven operations + `flow` + wrapped variants.~~ Raw ops + `flow`
  landed sessions 4-6; wrapped variants + `validate_driver_profile`
  + `validation.rs` + `provenance.rs` landed session 7. Surface
  complete at the operations + validation + provenance layer.
- All translation-field shapes (lookup_table, tangent_flow, learned).
- Banach substrate + sidecar + InverseLookupSidecar dispatch.
- **Continuous self-test cadence** — `self_test.py`'s
  `SelfTestCadence` + `BanachDriftReport` + `run_banach_self_test`.
  v5 ran the cadence synchronously per tick (microsecond-scale
  pure-Python ops); v6 can run it on a separate OS thread / pinned
  core trivially and have it truly out-of-band per the BLOCK_IN
  framing. The streaming-side hook (cadence + callback on
  `forward_sweep_invert_stream`) is the integration surface to
  reproduce.
- **Sensitivity backprop** (via the native autodiff library — `enzyme`
  for Rust, `autodiff` / hand-written for C++; pick at session time).
  v5 modules to reproduce: `sensitivity.py`'s
  `trajectory_substrate_diff`, `trajectory_substrate_jacobian`,
  `field_parameter_sensitivity`, `inversion_sensitivity`, and
  the one-liner `driver_profile_loss_grad`. All compose
  `jax_core` / `jax_ops` primitives through the audit traversal.
- ~~**Gradient-based inversion**~~ — landed session 5 at
  `rust/src/operations.rs` (`forward_sweep_invert` + `Method` enum)
  and `rust/src/optim.rs` (`minimize_smooth_2d`). Closed-form
  tangent_flow path wraps `math::tangent_flow_canonical_inverse`
  (session-1 bit-identity tested). Learned-field path uses a hand-
  rolled 2D damped-Newton solver (not `argmin`); see the session 5
  log bullet for the deliberate-divergence rationale and the
  Open/watch L-BFGS entry below for the cross-language convergence
  defer.
- Streaming + symbolic query + cross-substrate ops + active learning +
  Bayesian + Caputo.
- ~~Full intents (I1–I5) + composition.~~ Landed session 6 at
  `rust/src/operations.rs` (`intent_map`, `intent_compose`, five
  `intent_iN` handlers) and `rust/src/types.rs` (`IntentId`,
  `CapacityClass`, `SacrificeRecord`, `IntentDiagnostics`). Wrapped
  variants (`intent_map_wrapped`, `intent_compose_wrapped`) carry to
  session 7 per the operations.py-split note in Open/watch.
- MCP server (port via the native MCP SDK once one exists; or keep
  Python `mcp_server.py` as a thin wrapper invoking native through
  pybind11/pyo3 if no native MCP SDK is mature at v6 session time).

**Math source.** `mpa_scale_solver/jax_core.py` is the canonical
math the port reads. Every primitive in jax_core has a 1:1 native
counterpart; the Python-level operations.py / flow.py / banach.py
are wrapper-shape only. Read jax_core first when porting; read
operations.py for surface / dispatch behavior. The v5 additions
that became part of the math surface: nothing new in `jax_core`
itself (v5 only composed existing primitives in `sensitivity.py`);
the `_invert_learned_bfgs` driver in `operations.py` — the one
function whose Python form was expected to leak into solver
behavior — landed in session 5 as a hand-rolled 2D damped-Newton
solver in `rust/src/optim.rs` (`minimize_smooth_2d`) rather than
`argmin`. The convergence-vs-scipy MAP-match remains the cross-
language acceptance bar; see Open/watch.

**Zero additions.** If v6 needs a capability that isn't in v5, that
capability lands in v5 first via a v5.x release.

**Acceptance.**

- Native: every v5 Python fixture passes byte-identical for
  deterministic ops; within IEEE-754 platform tolerance for
  stochastic ops.
- Per-seed reproducibility: same top-level seed → same per-realization
  outputs across Python, native single-threaded, native multi-threaded,
  WASM (if produced).
- Performance: deterministic ops ≥10× faster than v5 Python single-
  thread; stochastic ensembles ≥50× faster with parallelism. v5's
  closed-form tangent-flow inversion is already O(1) per call; the
  native port's win there is mainly compile-time overhead removal
  + tighter inner-loop dispatch (the BLOCK_IN's "≥10× faster than
  grid" target was met inside v5; v6 doubles down via SIMD on the
  multi-frame `tau_obs_sweep` and `trajectory_substrate_jacobian`
  pathways).
- Python bindings (pybind11 / pyo3) — mpa-conform's
  `import mpa_scale_solver` works unchanged. The whole point: the
  consumer surface doesn't notice. The `method` kwarg semantics
  and `SelfTestCadence` / `BanachDriftReport` shapes are part of
  this contract.
- This document deletes entirely. The
  [`mpa-scale-solver/README.md`](https://github.com/ronviers/mpa-scale-solver/blob/main/README.md)
  § Session Log carries the history; per-version git tags carry the
  releases.

**Dependencies.** v5 shipped.

**Open / watch.**

- ~~Language pick: Rust vs C++.~~ Rust, settled session 1.
- ~~WASM bindings: produce if browser-side consumer materializes.~~
  Settled session 1: WASM is load-bearing (browser is the target).
- ~~MSVC linker for `cargo test`.~~ Resolved session 1 via GUI
  Modify (memory: `feedback_msvc_workload_gui_install.md`).
- ~~Bit-identity test fixtures vs Python (math.rs).~~ Resolved
  session 2 — fixture + Rust integration test green; per-primitive
  ULP budgets documented in `rust/tests/bit_identity.rs`. The v6
  acceptance "byte-identical for deterministic ops" check is
  satisfied at the math.rs layer.
- ~~Port types.py → types.rs.~~ Resolved session 3 — 17 structs +
  5 enums + `TranslationField` tagged enum + `SidecarKey` newtype.
  Smoke test (`rust/tests/types_smoke.rs`) covers serde round-trip
  on every public type and the `shape`-tag discriminator on
  `TranslationField`.
- **Cross-language JSON parity for shape-bearing types.** The
  types.rs serde round-trip is Rust-internal. Cross-language
  parity (a Python-emitted JSON for a `LearnedField` /
  `OperationOutput` / `InverseLookupSidecar` instance deserializes
  byte-identically into Rust) is unproven for the remaining types
  and lands with whichever module ports first that actually
  serializes the relevant type to JSON — `sidecar.py` for
  `InverseLookupSidecar`, `mcp_server.py` for `OperationOutput<T>`
  over the tool responses. The `SidecarKey` ':'-joined-bits wire
  format and the `gamma_AB` schema-field name preservation are the
  two design decisions that will need to either match the future
  Python producer or trigger a Python-side renormalize before
  merging.
  - ~~`SacrificeRecord` parity~~ — landed session 6 via the
    `sacrifice_record` fixture in `jax_core_reference.json` + the
    `sacrifice_record_python_to_rust_json_parity` test in
    `bit_identity.rs`. Asymmetric-by-design: Python→Rust works
    (serde drops the stored `preserved_invariant` key); Rust→Python
    would need a custom `Serialize` (one-helper, not currently
    needed since Python is the producer). See the session-6 log
    bullet for full rationale.
  - ~~`OperationOutput<RegimeReading>` parity~~ — landed session 7
    via the `operation_output_regime_at` fixture in
    `jax_core_reference.json` + the
    `operation_output_regime_at_python_to_rust_parity` test in
    `bit_identity.rs`. Asymmetric-by-design: timestamps + note
    strings are excluded — timestamps are non-deterministic; notes
    have a Python/Rust float-formatting divergence (`f"{0.0}"` is
    `"0.0"` in Python, `format!("{}", 0.0)` is `"0"` in Rust).
    Structured fields (`value.regime`, `value.k_frust`, the three
    `ValidationReport` flags, and `Provenance.{solver_version,
    operation, dispatch_path, table_version}`) are bit/string-exact.
    The pattern extends to each remaining wrapped op as
    `cross_substrate.py` / `mcp_server.py` / etc. port; the
    representative coverage at session 7 proves the wire format
    works end-to-end. Full per-op coverage is a future
    bit-identity-fixture extension, not a re-port.
  - ~~`provenance_hash` bit-identity~~ — landed session 7 via the
    `provenance_hash` fixture (12 cases covering every `DispatchPath`
    variant + null vs non-null `table_version`) and the
    `provenance_hash_python_to_rust_parity` test in `bit_identity.rs`.
    The fixture stores the **raw 4-byte blake2b digest as hex** (not
    the rational float view) and Rust compares digest bytes directly
    via the new public `provenance::provenance_digest_bytes`. The
    earlier session-7 commit briefly enabled `serde_json/float_roundtrip`
    to launder a 1-ULP JSON drift; that workaround was retired in the
    same session — the float-in-JSON form was the wrong contract for
    a rational hash. See the **fixture discipline** note below.
- **Fixture discipline — no bit-exact floats in JSON.** Surfaced
  during the session-7 retro on `float_roundtrip`. JSON stores floats
  as decimal strings; round-tripping IEEE-754 doubles is brittle
  under naive parsers. Two rules:
  1. Computed floats (residuals, drives, gamut bounds, MAP points,
     posterior moments) use `assert_close` with `LIBM`/`LIBM_WIDE`
     ULP budgets. The budget absorbs both wire-format drift and the
     libm reduction-order difference.
  2. Rational / digest values (provenance_hash, sidecar keys, integer
     counts) never go through `f64` in the fixture. Store the
     underlying bits as a hex string and compare bytes directly.
  Adding `serde_json/float_roundtrip` is the wrong direction — it
  papers over a fixture-design mistake at ~50 KB wasm cost. Future
  cross-language parity tests (session 8 `Posterior`, eventual
  `InverseLookupSidecar` parity) MUST follow these rules. The full
  rule lives in the `bit_identity.rs` module docstring.
- **`operations.py` deferred-session split.** Session 4 landed the
  raw forward path (`apply_translation` + three dispatch helpers,
  `forward_sweep_invert_grid`, `tau_obs_sweep_grid`, `regime_at`,
  `regime_display_band`, `gamut_classify`) plus the three small
  deps (`gfdr_model`, `sidecar`, `flow`). Session 5 landed the
  gradient-inversion dispatcher. The remaining `operations.py`
  surface ports across three subsequent sessions, each a coherent
  slice that ships its own bit-identity fixtures and lands its
  own commit + tag:
  - ~~*Session 5 — gradient inversion.*~~ Landed 2026-05-16 as
    v6.2.0 — see the §v6 session log bullet. `Method::{Auto, Grid,
    Gradient}` + `InversionResult`; closed-form tangent_flow path +
    hand-rolled 2D damped-Newton L-BFGS-equivalent (not `argmin` —
    see the divergence rationale in the session-5 log bullet).
  - ~~*Session 6 — intent algebra.*~~ Landed 2026-05-16 as v6.3.0 —
    see the §v6 session log bullet. `IntentId` + `CapacityClass` +
    `SacrificeRecord` + `IntentDiagnostics` in `types.rs`;
    `intent_map` + `intent_compose` + the five `_intent_iN` handlers
    + helpers in `operations.rs`; two new errors
    (`IntentComposeEmpty`, `I2InComposition`). 26 new tests; 113/113
    Rust total. No new bit-identity fixtures (pure arithmetic — the
    pre-session prediction held).
  - ~~*Session 7 — validation + provenance + wrapped variants.*~~
    Landed 2026-05-16 as v6.4.0 — see the §v6 session log bullet.
    `validation.py` + `provenance.py` → `validation.rs` +
    `provenance.rs`; the eight `*_wrapped` operations
    (`apply_translation_wrapped`, `forward_sweep_invert_wrapped`,
    `tau_obs_sweep_wrapped`, `regime_at_wrapped`,
    `gamut_classify_wrapped`, `intent_map_wrapped`,
    `intent_compose_wrapped`, `validate_driver_profile_wrapped`)
    plus raw `validate_driver_profile`. Two parity-fixture
    additions (provenance_hash via digest-hex storage,
    operation_output_regime_at via structured-fields + ULP-tolerance
    on residuals — see the per-fixture entries above). Single new
    dep: `blake2 = "0.10"` (default-features off).
    `SOLVER_VERSION = "5.0.0"` const tracks Python's `__version__`
    (deliberate decoupling from the Cargo crate version).
  - *Session 8 — posterior.* `forward_sweep_invert_posterior` +
    `_wrapped`. Depends on the Laplace primitives already in
    `math.rs`; the wrapper builds a `Posterior` from a posterior
    covariance + MAP point. The session-7-proven wrapped-variant
    pattern (raw + `*_wrapped` stamping `OperationOutput<T>` +
    `make_provenance` + reusing `report_for_forward_sweep_invert`
    for the MAP point) is the template; Python reuses that report
    builder verbatim (operations.py line 1662), so no new
    `report_for_posterior` is needed at the v5 surface.
    **Session-7 audit found one missing primitive:**
    `jax_ops.tangent_flow_forward_jacobian` does not have a Rust
    counterpart — Python uses `jax.jacfwd` over
    `tangent_flow_substrate`; Rust must port the analytical 2x2
    form. The Jacobian is diagonal:
    `[[1, 0], [0, (tau_obs/tau_obs_ref)^delta_gamma]]` for the
    non-degenerate path and `[[1, 0], [0, 1]]` (identity) at the
    degenerate `tau_obs <= 0` branch — ~10 LOC in `math.rs`. All
    other primitives needed by `tangent_flow_posterior`
    (`tangent_flow_canonical_inverse`, `laplace_covariance_from_jacobian`,
    `slogdet_2x2`) and `lookup_table_posterior`
    (`forward_sweep_invert` with `return_residuals=true`, libm
    `exp`, sort) are already in place. **Fixture-discipline rule
    applies to the new `operation_output_posterior` parity test:**
    `Posterior.mean` (`CanonicalState`), `covariance` (`[[f64; 2]; 2]`),
    `noise_variance`, `log_evidence` — all *computed* floats →
    `LIBM_WIDE` ULP tolerance, never bit-exact JSON storage.
    Sole non-trivial dispatch question deferred to session-8-time:
    `lookup_table_posterior`'s `k == 1` degenerate path (delta
    posterior with noise-floor covariance) needs explicit handling.
    One new bit-identity fixture (`operation_output_posterior`
    mirroring `operation_output_regime_at`) extends the
    wrapped-variant wire-parity coverage.
  - *Session 9 — bindings.* `pyo3` + `wasm-bindgen`. This is the
    one that makes v6 actually shippable — mpa-conform's
    `import mpa_scale_solver` keeps working unchanged; mpa-auditor
    gains a browser-side native solver. Per-op `__call__` shims and
    typed PyO3 conversions for the eight wrapped variants land here
    (validation + provenance ride along automatically — they're
    already `Serialize + Deserialize`). **Session-7 prerequisites
    audit:** pyo3 build prerequisites verified on this Windows
    machine — `C:\Program Files\Python312\libs\python312.lib` +
    `python312.dll` + `Include\Python.h` all present, so the
    `PYO3_PYTHON` auto-detection path will work without GUI
    installer intervention. `wasm32-unknown-unknown` target was
    bootstrapped at session 1; only `wasm-pack` / `wasm-bindgen-cli`
    (`cargo install wasm-pack`) needs to land at session-9 time.
    **Two open design questions to settle at session-9 time:**
    (1) `OperationOutput<T>` is generic — pyo3 cannot expose
    parameterized Rust types directly. Choose between per-`T`
    concrete wrappers (`PyOperationOutputSubstrateState`,
    `PyOperationOutputCanonicalState`, ...) or a dict-shaped Python
    view (`{value: ..., validation: {...}, provenance: {...}}` via
    serde_json). The dict-shape is thinner and matches Python's
    existing `dataclasses.asdict` consumer surface — leaning
    toward it. (2) `InverseLookupSidecar` wire format is currently
    undefined: this crate consumes whatever mpa-conform's curator
    path emits, but the producer side has not landed in mpa-conform
    yet. The Rust `SidecarKey`'s `':'`-joined-bits string form is a
    placeholder that will need to either match the future Python
    emitter or trigger a Python-side renormalize. **Not blocking
    for session 9** (no sidecar JSON I/O ports at session 9), but
    flag for the mpa-conform curator session.
  The session-2 fixture lesson (specify `(candidate, target)`
  pairs explicitly; never generate `target` from one impl and
  test the other) was load-bearing again in session 4: see the
  `gfdr_locus_residual` and `sidecar_round_key` comments in
  `emit_jax_core_reference.py`. Session 5 added no new bit-identity
  fixtures (composes existing primitives) so the lesson didn't fire,
  but cross-language convergence against
  `test_learned_field.py::TestForwardSweepInvertLearned` defers to
  whichever session ports `cross_substrate.py` / `active_learning.py`
  (they instantiate learned fields with materialized weights and
  exercise the MAP). Expect at least one new instance of the
  lesson per remaining session.
- ~~L-BFGS implementation choice for the learned-field inversion
  path.~~ Resolved session 5 — hand-rolled 2D damped-Newton solver
  in `rust/src/optim.rs` (`minimize_smooth_2d`) substitutes scipy's
  L-BFGS-B. Convergence tolerance: `FD_STEP = 1e-6`,
  `GRAD_TOL = 1e-10`, `F_TOL = 1e-15`, `MAX_ITER = 50`,
  `LINE_SEARCH_HALVINGS = 30`. Cross-language MAP-match verification
  against Python's `test_learned_field.py::TestForwardSweepInvertLearned`
  recovery set deferred to whichever session ports `cross_substrate.py`
  / `active_learning.py` (per the operations.py-split note above).
- Self-test cadence threading: v5 synchronous; v6 can run on a
  separate native thread per stream. The `BanachDriftReport` shape
  is plain-struct and thread-safe to pass; the cadence's call
  counter needs atomic increment if shared across producer threads.

---

## Reading order when a session opens this document

1. [`SUITE_BLOCK_IN.md`](../../mpa-central/SUITE_BLOCK_IN.md) — the
   three-layer split. Don't drift across layers.
2. [`NORTH_STAR.md`](NORTH_STAR.md) —
   the destination. Every version session points here.
3. This document — §v6 is the only remaining version.
4. [`../CLAUDE.md`](../CLAUDE.md) —
   sibling-kernel discipline + per-call discipline. The "Python is
   the pseudo-code spec for v6" rule is now load-bearing: read v5's
   `jax_core.py`, `sensitivity.py`, `self_test.py`, and the
   `forward_sweep_invert` dispatch in `operations.py` as the port's
   reference.
5. The relevant RFC-S sections + v9_receipts + cdv1_receipts entries.
6. Prior version's release notes in
   [`../README.md`](../README.md)
   § Session Log — the historical record this document doesn't carry.

When v6 ships, this document deletes entirely. The README session log
+ the per-version git tags are the residue.
