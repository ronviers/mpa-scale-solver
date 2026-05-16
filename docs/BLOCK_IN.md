# mpa-scale-solver — v2→v6 block-in handoff

Self-evolving trajectory handoff. v1.0.0 shipped 2026-05-16 (consumed
end-to-end by mpa-conform v0.2). v2.0.0 shipped 2026-05-16 (BLOCK_IN
§v2 cut (a) — JAX foundation + differentiability). v2.1.0 shipped
2026-05-16 (cut (b) — Bayesian inversion via Laplace approximation).
What remains: ~~v2.2~~ (premise overturned by framework cross-check —
see §v2.2), v2.3 (I1–I4 intents), v2.4 (Caputo flow), → v3 → v4 →
v5 → v6.

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
| v1 | Continuous flow + Banach + sidecar + per-call validation | — (shipped) |
| v2.0 | JAX foundation + differentiability (BLOCK_IN cut a) | v1 (shipped) |
| v2.1 | Bayesian inversion (Laplace around MAP) (cut b) | v2.0 (shipped) |
| ~~v2.2~~ | ~~N-mode generalization~~ — **premise overturned**, see §v2.2 | framework-side decision required |
| **v2.3** | Full I1–I4 intents + composition algebra (cut d) | v2.0 |
| **v2.4** | Non-Markovian Caputo flow (β_mem < 1) via Prony (cut e) | v2.0 |
| **v3** | Cross-substrate operations + active learning + MCP server + learned translation-field form | v2.* (active learning prefers v2.1's posteriors) |
| **v4** | Streaming / online operation + symbolic query interface + notebook ergonomics | v3 (or v2.* if v3 deferred) |
| **v5** | Continuous Banach self-test cadence + sensitivity backprop + gradient-based inversion replacing grid where invertible | v2.0 (v3/v4 optional) |
| **v6** | One-shot native port (Rust or C++). Zero new features. Per-seed reproducibility against the v5 Python. | v5 |

Sequencing is the user's call per ROADMAP. The dependency column is
the *minimum* — v5 can ship as soon as v2.0 lands if v2.2–v2.4 and
v3/v4 are deferred (sensitivity_backprop / gradient-based inversion
only need the JAX foundation, not the higher v2.x slices).

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

## §v2.2–v2.4 — remaining v2 cuts (N-mode, I1–I4, Caputo)

v2.0 (BLOCK_IN cut a) shipped the JAX foundation. v2.1 (cut b)
shipped Bayesian inversion via Laplace approximation. Cuts (c)–(e)
remain as independent v2.x slices, each one its own session. They
build on the v2.0/v2.1 surface rather than reinventing it.

**Foundation that landed at v2.0 (do not re-do):**

- `mpa_scale_solver.jax_core` — pure JAX math primitives, float64
  enabled, JIT-able and differentiable: `tangent_flow_substrate`,
  `tangent_flow_canonical_inverse`, `tangent_flow_canonical`,
  `banach_state`, `lookup_squared_distance`,
  `tangent_flow_inversion_residual`,
  `laplace_covariance_from_jacobian`, `laplace_covariance_from_hessian`,
  `laplace_log_evidence`. This is the single math source the v6 native
  port reads as reference.
- `mpa_scale_solver.jax_ops` — consumer surface returning JAX arrays:
  `tangent_flow_substrate_diff`, `flow_diff`,
  `tangent_flow_forward_jacobian`, `banach_state_diff`,
  `forward_sweep_invert_diff` (exact closed-form inverse on
  tangent-flow), `tangent_flow_posterior`, `lookup_table_posterior`.
  Composition under `jax.grad` / `jax.jacobian` / `jax.hessian` works
  directly on CanonicalState-typed callbacks.
- `mpa_scale_solver.jax_pytree` — CanonicalState as a JAX PyTree
  (leaves: `(chit, gamma_AB)`; aux: `k_frust`). Registered on
  import; idempotent.
- v0/v1 unwrapped sigs and `*_wrapped` variants unchanged. Fixture
  byte-identity contract preserved.
- JAX is now a hard dep (`jax>=0.4`, `jaxlib>=0.4`).

**Foundation that landed at v2.1 (do not re-do):**

- `Posterior` dataclass in `types.py` — mean (CanonicalState),
  covariance (2x2 tuple), noise_variance, log_evidence, modes, notes.
- `operations.forward_sweep_invert_posterior` /
  `forward_sweep_invert_posterior_wrapped` — Bayesian inversion.
  Tangent-flow: closed-form fast path. Lookup-table: weighted-moment
  fit over top-k candidates. Separate function rather than
  `posterior=True` kwarg on the existing wrapped variant — cleaner
  return-type contract.

### §v2.2 — N-mode (cut c) — **PREMISE OVERTURNED; reframing required**

**The v2.2 BLOCK_IN sketch (vector `chit` + matrix `gamma_AB`) is
misaligned with the framework. Do not implement it as written. See
the framework-side finding below before proceeding.**

**Framework finding (2026-05-16 cross-repo check):** the
"two-mode" CanonicalState `(chit, gamma_AB, k_frust)` is **not** a
2-mode-as-N=2-special-case; it IS the framework's universal canonical
representation. Authoritative pointers:

- `mpa-atlas/framework/cdv1_compressed.md` §"Universal two-mode
  kernel" (line 125) — declares the two-mode kernel as the canonical
  Character kernel, with the `D[ρ_A, ρ_B; γ_AB]` closure family
  parameterized by the same `γ_AB`.
- `mpa-atlas/framework/cdv1_compressed.md` §10 "Four-channel pattern
  selection architecture" (line 493) — N≥3 substrates do NOT widen
  the canonical state; they decompose into four closed tests
  (frustration / spectral-sync SBN / non-reciprocity / active-matter
  overlay). Quote: *"any N≥3 Character kernel routes through these
  four tests."*
- `mpa-atlas/framework/cdv1_compressed.md` line 150 — `k_frust` is
  explicitly the N≥3 frustration-test outcome bool on the existing
  CanonicalState. We already store it; we already propagate it
  through `regime_at` / `intent_map`.
- `mpa-atlas/framework/cdv1_receipts.md` §9 / §16 / §17 / §18 / §21
  — every N≥3 result (May–Leonard 3-species, Schnakenberg cycle
  currents, non-reciprocal kernels, chimera states) composes the
  two-mode kernel pairwise; no wider state ever appears.

**What this means for v2.2.** "Generalize CanonicalState to N>2"
isn't a missing feature — it would contradict the framework's
design. The 2-mode CanonicalState is the universal canonical form by
the same logic that makes the seven-operation API the canonical
surface: thinking of it as "the N=2 case" is the category error.

**Two legitimate reframings (pick one; both want a foundational-
questions entry first per CLAUDE.md no-eighth-operation rule):**

1. **Cancel v2.2.** Conclude that the framework's universal
   two-mode CanonicalState is already what v2.2 should produce. Mark
   cut (c) as not-applicable; renumber the trajectory table; carry
   on to v2.3 / v2.4. **This is the recommended default unless a
   substrate has surfaced a concrete N≥3 measurement need that the
   four-channel tests don't already cover.**

2. **Reframe v2.2 as the four-channel pattern-selection tests.** Add
   `frustration_test`, `spectral_sync_test`, `non_reciprocity_test`,
   `active_matter_overlay` as operations that consume **populations
   of 2-mode CanonicalStates** (a labelled mode-pair graph with a
   coupling matrix `γ_ij`) and return verdicts on the four channels.
   These are new operations beyond the seven — each requires a
   foundational-questions entry per CLAUDE.md. The wider type is
   `ModePairGraph = list[tuple[ModeId, ModeId, CanonicalState]]`,
   not a generalized CanonicalState; the four operations sit
   alongside the existing seven, not inside them.

**Before any code:** spec the N≥3 reading shape with mpa-atlas /
mpa-central. The driver-profile schema today is open-typed enough
that no schema bump is forced by either reframing, but
`mpa-central/SUITE_BLOCK_IN.md` decides where the four-channel
operations live (scale-solver vs. mpa-conform vs. a new sibling
focused on multi-mode pattern selection).

**Dependencies.** A foundational-questions entry in
`H:/mpa-scale-solver/docs/foundational-questions.md` (does not exist
yet — would be the first; the cross-repo pattern is mpa-auditor /
mpa-conform's `docs/foundational-questions.md`). Framework-side
ratification before implementation.

**Acceptance.** Deferred until reframing is settled.

### §v2.3 — full I1–I4 intents + composition algebra (cut d)

**Goal.** `intent_map` accepts any of `{I1 regime_preserving,
I2 drive_faithful, I3 capacity_preserving, I4 persistence_preserving,
I5 signature_preserving}`. Composition algebra between adjacent
intents per RFC-S §3.

**Capabilities to land.**

- Four new intent implementations in `operations.intent_map`
  (the dispatch arm already raises `NotImplementedError` for I1–I4;
  fill in the implementations).
- Each intent's invariance check fires in `validation.report_for_intent_map`.
- Composition algebra: applying I_i then I_j composes per RFC-S §3.
  Where they conflict (e.g. I1+I3 over-constrained), report a
  sacrifice record naming the broken invariant.

**Acceptance.**

- v0 + v1 + v2.* fixtures pass unchanged.
- New: `test_intents.py` — each intent's invariance check fires;
  composition algebra holds (no conflicts on independent intents;
  documented sacrifices on conflicting pairs).
- README + CLAUDE.md updated; this §v2.3 deleted.
- Tagged `v2.3.0`.

### §v2.4 — non-Markovian Caputo flow (cut e)

**Goal.** `flow(canonical, nu, field)` gains `beta_mem < 1` support
via Prony sum-of-exponentials fit to the Mittag-Leffler kernel
(mpa-solver's pattern, parallel-able). v1's Markovian path
(`beta_mem = 1`) unchanged.

**Capabilities to land.**

- Extend `ScalingRule.refinement` to accept `beta_mem: float` and
  `prony_terms: list[tuple[float, float]]` (amplitude, decay-rate
  pairs).
- New `jax_core.caputo_flow` primitive computing the Prony
  sum-of-exponentials approximation. Differentiable in all
  parameters.
- `flow()` and `flow_diff()` dispatch on `refinement.get("beta_mem", 1.0)`:
  1.0 → existing Markovian path; <1.0 → Caputo path.

**Acceptance.**

- v0 + v1 + v2.* fixtures pass unchanged.
- New: `test_caputo_flow.py` — β_mem=1 byte-identical to v1's
  Markovian; β_mem=0.5 matches Prony-reference within 1e-3 over a
  representative ν grid.
- README + CLAUDE.md updated; this §v2.4 deleted; §v3 refined to
  reflect any cross-substrate implications.
- Tagged `v2.4.0`.

**Open / watch.** The Prony terms can either ship in the driver
profile (curator-produced) or be fit on-the-fly from a measured
memory kernel. For v2.4, accept pre-fit Prony terms only — fitting
is mpa-conform's curator-path job (a separate follow-on session).

---

## §v3 — Cross-substrate operations + active learning + MCP server + learned translation-field form

**Goal.** First-class cross-substrate operations (gamut overlap,
canonical-state distance, universality-class agreement). Active
learning that suggests where curators should measure next. Expose the
solver as an MCP server. Add a learned-NN translation-field form
alongside lookup_table and tangent_flow.

**Capabilities to land.**

- **Cross-substrate ops** — new operations (each on the wrapped
  surface): `gamut_overlap(profile_a, profile_b)`,
  `canonical_distance(state_a, state_b, metric)`,
  `universality_agreement(profile_a, profile_b)`. The framework's
  primary cross-substrate test (s→r migration per cdv1 §gFDR
  signatures) becomes a direct call. These count against the
  "seven-operation API stays stable" rule — they are *cross-substrate
  compositions*, not new fundamental ops. Document as such.
- **Active learning** — `suggest_measurements(profile, n=5)` returns
  candidate operating points where the driver profile is weak
  (high-uncertainty regions in canonical space, gamut edges with low
  classification confidence). Curator operators consume these when
  planning library expansions. Builds on v2.1's `Posterior` (uses
  log-evidence + covariance trace as the per-point uncertainty
  surface).
- **MCP server** — the seven operations exposed as MCP tools.
  Stateless, JSON I/O. Read-only over driver profiles (no write
  surface). Tested against the MCP reference client.
- **Learned translation-field form** — `LearnedField` joins
  `LookupTableField` and `TangentFlowField` as a third shape.
  `apply_translation` dispatches on `field.shape == "learned"`.
  Implementation: small JAX MLP using the v2.0 `jax_core` /
  `jax_ops` foundation (the differentiable-forward-map surface
  is already in place). Weights ship in the driver profile. Training
  is curator-side (separate concern); solver only evaluates.

**Acceptance.**

- v0 + v1 + v2 fixtures pass unchanged.
- New: `test_cross_substrate.py`, `test_active_learning.py`,
  `test_mcp_server.py`, `test_learned_field.py`.
- README + CLAUDE.md updated; §v3 deleted from this block-in;
  §v4/v5 refined for any cross-substrate API decisions that affect
  them.
- Tagged `v3.0.0`.

**Dependencies.** v2.0 shipped (the minimum); active learning needs
v2.1's `Posterior`. MCP server needs no upstream — schedule freely
within v3.

**Open / watch.**

- MCP server lifecycle: long-running process vs stdio-per-call.
  Default to stdio (matches the broader MCP convention); long-running
  only if a consumer asks.
- The learned-field shape will pressure the driver-profile schema in
  mpa-atlas. Coordinate the schema bump with the same session, or
  ship learned_field as forward-compat optional and bump driver-
  profile separately.

---

## §v4 — Streaming / online operation + symbolic query interface + notebook ergonomics

**Goal.** Solver consumes substrate observations as a stream and
emits canonical states as a stream — for real-time experiments and
interactive analysis. Symbolic query interface (Mathematica-style).
Notebook / REPL ergonomics across the board.

**Capabilities to land.**

- **Streaming API** — `forward_sweep_invert_stream(observations: Iterator[SubstrateState], field, tau_obs) -> Iterator[InversionResult]`.
  State-local (per-frame); no cross-frame leakage. Supports stdin /
  WebSocket / polling sources via thin adapters in `mpa_scale_solver.streams`.
- **Symbolic query** — `query("what tau_obs makes substrate <id>
  cross the c→s threshold?")` parses a small DSL and translates to
  operation chains. Returns piecewise expressions where they exist +
  numerical evaluations.
- **Notebook ergonomics** — rich `__repr__` for every dataclass;
  default `_repr_html_` for Jupyter; default plot hooks (matplotlib
  + plotly) per north-star §Visualization-first; lazy evaluation
  where it helps.

**Acceptance.**

- v0 + v1 + v2 + v3 fixtures pass unchanged.
- New: `test_streaming.py`, `test_symbolic_query.py`,
  `test_notebook_repr.py`.
- README + CLAUDE.md updated; §v4 deleted; §v5 refined if any
  streaming-side reproducibility constraints surfaced.
- Tagged `v4.0.0`.

**Dependencies.** v3 shipped (or v2 if v3 deferred — the streaming
shape doesn't require cross-substrate ops). Active-learning streaming
benefits from v3's `suggest_measurements`.

**Open / watch.**

- Symbolic query is a feature-rich DSL. Scope risk: keep the v4
  surface small (5 query patterns max) and defer richer DSL to a
  later session if the demand surfaces.

---

## §v5 — Continuous Banach self-test + sensitivity backprop + gradient-based inversion

**Goal.** Last functional version before native port. Continuous
self-test cadence against Banach. Sensitivity backprop through the
RG-flow trajectory enables driver-profile hyperparameter
optimization. Gradient-based inversion fully replaces grid search in
monotonic regions (grid stays for ambiguity).

**Capabilities to land.**

- **Continuous self-test** — every k-th operation call (configurable,
  default k=100) runs a side-test on the Banach substrate. Drift
  reported with full diagnostic state. ECC for compute. Self-tests
  are async / out-of-band where backend permits; never block the
  primary call.
- **Sensitivity backprop** — full chain rule through
  `apply_translation → forward_sweep_invert → tau_obs_sweep` (the
  audit traversal). Builds on v2.0's `jax_core` /
  `jax_ops.tangent_flow_forward_jacobian`; v5 composes the per-op
  Jacobians into the full trajectory chain rule. Driver-profile
  hyperparameter optimization becomes a one-liner.
- **Gradient-based inversion** — `forward_sweep_invert` defaults to
  gradient-based (L-BFGS / Newton) in monotonic regions; falls back
  to v2's grid for ambiguity. Grid remains available via
  `method="grid"` for back-compat and ambiguity reporting. Tangent-
  flow already lands closed-form via v2.0's
  `jax_ops.forward_sweep_invert_diff`; v5 generalizes to learned-
  field and lookup-table-with-smooth-surrogate cases.

**Acceptance.**

- v0 + v1 + v2 + v3 + v4 fixtures pass unchanged.
- New: `test_continuous_self_test.py`, `test_sensitivity_backprop.py`,
  `test_gradient_inversion.py`.
- Performance: gradient-based inversion ≥10× faster than grid on the
  Banach substrate; matches grid within tolerance.
- README + CLAUDE.md updated; §v5 deleted; §v6 refined with the
  exact per-seed reproducibility frontier the port must match.
- Tagged `v5.0.0`.

**Dependencies.** v2 shipped (the minimum). v3/v4 optional.

**Open / watch.**

- Self-test cadence overhead: 1/k call cost. k=100 is the default
  starting point; profile and adjust per backend.
- Gradient-based inversion in ambiguous regions: silent fall-back to
  grid vs explicit ambiguity report. Default to explicit report
  (matches the "Bayesian posterior over point estimates" v2 stance).

---

## §v6 — Native port (Rust or C++). Zero new features.

**Goal.** One-shot consolidation. Language pick at session time:
**Rust** if we want modern + memory-safe; **C++** if we want
mpa-solver toolchain parity. Match the proven v5 Python under the
per-seed reproducibility discipline. Free to parallelize aggressively
(Rayon / OpenMP / GPU); Python's single-thread constraints do NOT
carry forward. WASM bindings produced as part of the same step if
browser-side execution becomes load-bearing.

**Capabilities to port (exactly the v5 Python surface).**

- Seven operations + `flow` + wrapped variants.
- All translation-field shapes (lookup_table, tangent_flow, learned).
- Banach substrate + sidecar + InverseLookupSidecar dispatch.
- Continuous self-test cadence.
- Sensitivity backprop (via the native autodiff library — `enzyme`
  for Rust, `autodiff` / hand-written for C++; pick at session time).
- Streaming + symbolic query + cross-substrate ops + Bayesian +
  N-mode + Caputo + full intents.

**Math source.** `mpa_scale_solver/jax_core.py` is the canonical
math the port reads. Every primitive in jax_core has a 1:1 native
counterpart; the Python-level operations.py / flow.py / banach.py
are wrapper-shape only. Read jax_core first when porting; read
operations.py for surface / dispatch behavior.

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
  thread; stochastic ensembles ≥50× faster with parallelism.
- Python bindings (pybind11 / pyo3) — mpa-conform's
  `import mpa_scale_solver` works unchanged. The whole point: the
  consumer surface doesn't notice.
- This document deletes entirely. The
  [`mpa-scale-solver/README.md`](https://github.com/ronviers/mpa-scale-solver/blob/main/README.md)
  § Session Log carries the history; per-version git tags carry the
  releases.

**Dependencies.** v5 shipped.

**Open / watch.**

- Language pick: Rust vs C++. Match mpa-solver (C++) for toolchain
  reuse, OR commit to Rust for modernness + memory safety. User call
  at session time, informed by what v5's Python ergonomics actually
  needed.
- WASM bindings: produce in this session if any browser-side consumer
  has materialized by then. Skip otherwise — the inverse-lookup-table
  sidecar pattern already sidesteps browser-side scale-solver
  execution.

---

## Reading order when a session opens this document

1. [`SUITE_BLOCK_IN.md`](../../mpa-central/SUITE_BLOCK_IN.md) — the
   three-layer split. Don't drift across layers.
2. [`NORTH_STAR.md`](NORTH_STAR.md) —
   the destination. Every version session points here.
3. This document — what remains.
4. The §vN section for the version being shipped.
5. [`../CLAUDE.md`](../CLAUDE.md) —
   sibling-kernel discipline + per-call discipline.
6. The relevant RFC-S sections + v9_receipts + cdv1_receipts entries.
7. Prior version's release notes in
   [`../README.md`](../README.md)
   § Session Log — the historical record this document doesn't carry.

Ship one version per session. Refine the block-in inline as you go.
Delete §vN when it lands.
