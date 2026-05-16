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
| **v6** | One-shot native port (Rust or C++). Zero new features. Per-seed reproducibility against the v5 Python. | v5 |

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
- **Gradient-based inversion** — `forward_sweep_invert`'s `method`
  kwarg dispatch: `"auto"` (default) routes tangent_flow to
  closed-form (`jax_core.tangent_flow_canonical_inverse`), learned
  to L-BFGS, lookup_table to grid. The v5 Python uses scipy's
  L-BFGS-B with `jax.grad` for the learned path; the v6 port
  swaps to its native optimizer + autodiff (Rust: `argmin` + `enzyme`;
  C++: `dlib` / hand-rolled + `autodiff`). The closed-form tangent_flow
  path is pure arithmetic — direct port.
- Streaming + symbolic query + cross-substrate ops + active learning +
  Bayesian + Caputo + full intents (I1–I5) + composition.
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
the `_invert_learned_bfgs` driver in `operations.py` is the one
function whose Python form leaks into solver behavior — it uses
scipy's BFGS, and the v6 port's choice of native optimizer must
converge to the same `(chit, gamma_AB)` MAP within tolerance.

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

- Language pick: Rust vs C++. Match mpa-solver (C++) for toolchain
  reuse, OR commit to Rust for modernness + memory safety. User call
  at session time, informed by what v5's Python ergonomics actually
  needed.
- WASM bindings: produce in this session if any browser-side consumer
  has materialized by then. Skip otherwise — the inverse-lookup-table
  sidecar pattern already sidesteps browser-side scale-solver
  execution.
- L-BFGS implementation choice for the learned-field inversion path:
  v5 uses scipy's L-BFGS-B (well-tested, default tolerances). v6
  picks a native optimizer (Rust `argmin`, C++ `dlib` or
  hand-rolled). Convergence tolerance choice affects bit-identity
  budget on the learned-field MAP point; document the chosen
  tolerance and verify against the v5 Python on the
  `test_learned_field.py::TestForwardSweepInvertLearned` recovery
  set.
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
