# mpa-scale-solver — v4→v6 block-in handoff

Self-evolving trajectory handoff. v1, v2, and v3 shipped 2026-05-16:
v1.0.0 (continuous flow + Banach + sidecar), v2.0.0 (JAX foundation,
cut a), v2.1.0 (Bayesian inversion via Laplace, cut b), v2.3.0 (full
I1–I5 intents + composition, cut d), v2.4.0 (non-Markovian Caputo flow
via Prony, cut e), v3.0.0 (cross-substrate ops + active learning + MCP
server + LearnedField + per-intent RFC-S §5 metrics). Cut (c) "N-mode
generalization" cancelled 2026-05-16 — premise overturned by framework
cross-check; the 2-mode CanonicalState is the framework's universal
canonical representation, not a 2-mode-as-N=2-special-case (see
§v2.2-cancelled tombstone below).
What remains: v4 → v5 → v6.

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
| v1–v3 | Foundation, JAX, Bayesian, intents, Caputo, cross-substrate, active learning, MCP, LearnedField — all shipped 2026-05-16 (see README §Session Log) | — (shipped) |
| ~~v2.2~~ | ~~N-mode generalization (cut c)~~ — cancelled 2026-05-16; tombstone retained below | — |
| **v4** | Streaming / online operation + symbolic query interface + notebook ergonomics | v3 (or v2.* if v3 deferred) |
| **v5** | Continuous Banach self-test cadence + sensitivity backprop + gradient-based inversion replacing grid where invertible | v2.0 (v3/v4 optional) |
| **v6** | One-shot native port (Rust or C++). Zero new features. Per-seed reproducibility against the v5 Python. | v5 |

Sequencing is the user's call per ROADMAP. The dependency column is
the *minimum* — v5 can ship as soon as v2.0 lands if v3 / v4 are
deferred (sensitivity_backprop / gradient-based inversion only need
the JAX foundation that landed in v2.0).

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

## §v4 — Streaming / online operation + symbolic query interface + notebook ergonomics

**Goal.** Solver consumes substrate observations as a stream and
emits canonical states as a stream — for real-time experiments and
interactive analysis. Symbolic query interface (Mathematica-style).
Notebook / REPL ergonomics across the board.

**Capabilities to land.**

- **Streaming API** — `forward_sweep_invert_stream(observations: Iterator[SubstrateState], field, tau_obs) -> Iterator[InversionResult]`.
  State-local (per-frame); no cross-frame leakage. Supports stdin /
  WebSocket / polling sources via thin adapters in `mpa_scale_solver.streams`.
  The v3 MCP transport is stdio-per-call; v4's streaming is the
  intra-call analogue (a single tool invocation pulling from an
  iterator). v4 may expose a streaming MCP tool variant if a
  consumer asks; defer until then.
- **Symbolic query** — `query("what tau_obs makes substrate <id>
  cross the c→s threshold?")` parses a small DSL and translates to
  operation chains. Returns piecewise expressions where they exist +
  numerical evaluations. v3's cross-substrate ops (`gamut_overlap`,
  `canonical_distance`, `universality_agreement`) are likely natural
  query targets; structure the DSL so they compose.
- **Notebook ergonomics** — rich `__repr__` for every dataclass
  (including v3's `LearnedField` and `MeasurementCandidate`);
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
benefits from v3's `suggest_measurements`; the symbolic-query DSL can
target v3's cross-substrate ops directly.

**Open / watch.**

- Symbolic query is a feature-rich DSL. Scope risk: keep the v4
  surface small (5 query patterns max) and defer richer DSL to a
  later session if the demand surfaces.
- Streaming MCP tool variant: hold until a consumer asks. The v3
  MCP server's call-per-request shape already covers the common
  agentic workflow.

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
  field (v3 `LearnedField` shape — `learned_field_substrate_diff` is
  already differentiable, so v5 only needs the inversion driver) and
  lookup-table-with-smooth-surrogate cases.

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
- Active-learning composability: v3's `suggest_measurements` returns
  a composite score. v5's gradient-based inversion can pick MCMC
  starting points from these candidates if they earn weight.

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
- Streaming + symbolic query + cross-substrate ops + active learning +
  Bayesian + Caputo + full intents (I1–I5) + composition.
- MCP server (port via the native MCP SDK once one exists; or keep
  Python `mcp_server.py` as a thin wrapper invoking native through
  pybind11/pyo3 if no native MCP SDK is mature at v6 session time).

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
