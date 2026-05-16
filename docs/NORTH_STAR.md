# mpa-scale-solver — north star

The destination. What the τ_obs-projection / canonical-frame kernel
becomes at maturity. Sequencing lives in `mpa-conform/docs/ROADMAP.md`
and in per-session handoffs; this file declares where we are heading.

The shape of the solver is sibling to `mpa-solver` (forward physics) per
[`H:/mpa-central/SUITE_BLOCK_IN.md`](../../mpa-central/SUITE_BLOCK_IN.md).
mpa-scale-solver's named family of operations stays at seven across all
versions — the surface is right. What grows is the *implementation depth*
behind each operation, the *backends* the solver runs on, and the
*ecosystem* it participates in.

---

## Math foundations

### Continuous flow as production

`C^ν = exp(ν · ln C)` as the primary representation across both proven
regimes:

- **Markovian** (`β_mem = 1`, where the Banach substrate sits): grounded
  by v9 receipts §RG closure.
- **Non-Markovian Caputo** (`β_mem < 1`): grounded by the fractional-RG
  generalization with substrate-class Hurst-class verification residual
  (v9 receipts §RG closure substrate-scope note).

Integer-N is a sampling helper, not the primary form.

### N-mode generalization

Arbitrary mode count. Non-reciprocal coupling, dynamic bath, Caputo
memory all compose cleanly with N-mode. The two-mode v0 generalizes
without API surface change — only the underlying kernel dispatch grows.

### Full I1–I5 intent operations

All five RFC-S intents (Regime-preserving, Drive-faithful,
Capacity-preserving, Persistence-preserving, Signature-preserving)
implemented as the single intent-parameterized "scale uniformly along
the gamut" rule. Composition algebra between adjacent intents grounded
in RFC-S §3.

### Translation field forms

All three forms on the same API, dispatching internally:

- **Lookup table** (v0 / v2 schema): production today.
- **Tangent-flow** (RFC-S Appendix B item 1): carries derivatives, not
  just values. The Banach substrate's γ-scaling rule is the canonical
  leading-order tangent-flow auto-remap.
- **Learned** (NN, GP, other parametric ML): driver profiles can ship
  a learned representation; solver evaluates via the same API.

### Cross-substrate operations

The solver takes two driver profiles and computes their
gamut-overlap, canonical-state distance, universality-class agreement.
First-class operations for cross-substrate universality testing —
the framework's primary cross-substrate test (s→r migration per cdv1
§gFDR signatures) becomes a direct solver call.

### Topological invariant tracking

`k_frust` is the canonical example. The solver tracks topological
invariants across every operation and verifies they are preserved.
v9 §Scale-relativity invariance becomes a runtime check, not an
implementation assumption.

### Asymptotic-Closure verification

Per v9 §Asymptotic closure (candidate): no framework-prediction
observable attains exact 0 or 1 at non-asymptotic points. The solver
verifies its outputs comply per call. Suspicious outputs are flagged
in the provenance trail. The Banach substrate, sitting at the
asymptotic limits by construction, is the unique exception — and the
solver knows it.

---

## Compute substrate

### Python + JAX as production through v5; native consolidation at v6

One source of truth, sequenced:

- **Python** (v0–v5): the canonical implementation while the API and
  capabilities are still proving themselves. Single source, no
  backend-sync cost. mpa-conform consumes via direct
  `import mpa_scale_solver`.
- **JAX** (v2+): adopted where it earns its weight — differentiability,
  vectorization, GPU. JAX is a Python library; the solver stays one
  codebase.
- **Native port** (v6): one-shot consolidation step. Language pick at
  that session (Rust if we want modern; C++ if we want mpa-solver
  parity). Matches the proven v5 Python under the **per-seed
  reproducibility** discipline (see Reliability section below) — same
  top-level seed produces same aggregate outputs across worker counts,
  platforms, and backends; deterministic ops byte-identical,
  stochastic ops within IEEE-754 platform tolerance. Native is free to
  parallelize aggressively (OpenMP / Rayon / GPU); Python's
  single-thread limitations do NOT carry forward. Zero new features —
  pure performance + safety + portability upgrade.
- **WASM** (deferred or skipped): browser-side scale-solver execution
  is sidestepped by the inverse-lookup-table sidecar (precompute dense
  tables in mpa-conform's curator path; browser does pure
  lookup+interpolation; no scale-solver execution in browser). If
  browser-side execution ever becomes load-bearing, the v6 native port
  produces WASM bindings as part of the same step.

The seven-operation API stays unchanged across v0–v6. Only v6 changes
the implementation language without changing the capability set.

### Fully differentiable end-to-end

JAX-native composition through the entire pipeline:

- `apply_translation` is a differentiable function of canonical state.
- `forward_sweep_invert` replaces brute-force grid search with
  gradient-based optimization (L-BFGS, Newton) for monotonic regions;
  falls back to grid search where the forward map is ambiguous.
- Sensitivity analysis (∂canonical / ∂substrate) comes free from autograd.
- Backprop through the RG-flow trajectory enables driver-profile
  hyperparameter optimization.

### Bayesian primitives

Inversion returns *posteriors*, not point estimates. Given a substrate
observation, `forward_sweep_invert` returns a distribution over
canonical states with uncertainty propagated through the forward map.
Multi-modal posteriors (ambiguity regions) are first-class outputs, not
silent failures.

### GPU acceleration

Where ensemble compute or MCMC matters (Bayesian inversion chains,
N-mode ensemble propagation, cross-substrate sweeps). CPU is sufficient
for single-frame operations.

### Streaming / online operation

The solver consumes substrate observations as a stream and emits
canonical states as a stream. For real-time experiments and
interactive analysis. Not batch-only.

---

## API surface

### Seven canonical operations (unchanged)

The v0 API is right and stays:

- `apply_translation`
- `forward_sweep_invert`
- `tau_obs_sweep`
- `regime_at`
- `gamut_classify`
- `intent_map`
- `validate_driver_profile`

What changes across versions: the *capability depth* behind each. v1's
`forward_sweep_invert` is brute-force grid; v2's is gradient-based +
Bayesian. The signature stays the same.

### MCP server interface

The solver exposes itself as a Model Context Protocol server. LLM-driven
workflows (mpa-conform's researcher path, agentic analysis tools,
external research models) call the seven operations as tools. Same
function surface, different transport.

### Symbolic query interface

Consumers write canonical-space queries declaratively:

```
"what τ_obs makes substrate <profile-id> cross the c→s threshold?"
"find γ_AB values where this substrate shows k_frust at τ_obs=10"
"compare gamut overlap between profile-A and profile-B"
```

Solver translates to operation chains, returns symbolic answers
(piecewise expressions, fixed points) plus numerical evaluations.

### Notebook + REPL friendly

Rich `__repr__` for every dataclass. Default plot hooks (matplotlib +
plotly). Lazy evaluation where it helps. Jupyter / IPython /
marimo-friendly. Mathematica-style symbolic-numerical exploration.

---

## Reliability

### Per-call self-validation

Every operation produces an output AND a validation report:

- **Asymptotic-Closure compliance**: did the output land on a
  non-asymptotic exact 0 or 1?
- **k_frust invariance**: did the topological invariant change when it
  shouldn't have?
- **Round-trip residual**: forward-then-back recovery within tolerance.
- **Internal consistency**: does the output match other channels'
  predictions?

Validation cost is amortized via shared structure with the primary
compute. Failures are flagged, not raised — consumers decide whether
to trust borderline outputs.

### Continuous self-test against Banach substrate

Every k-th solver call (configurable, default k=100) runs a
side-test on the Banach substrate to verify the implementation hasn't
drifted. ECC for compute. Drift is reported with full diagnostic
state.

### Full provenance / audit trail

Every output carries a provenance record:

- Which operations ran
- Which lookup tables hit (with table version + cache key)
- Which calls fell through table-first to compute-fallback
- Residuals, validation reports, version stamps
- Driver-profile IDs and versions used

Consumers (mpa-conform's bundle assembly, the auditor's display layer)
read provenance directly into AuditDelta and the bundle.

### Per-seed reproducibility (parallel-friendly)

Reproducibility is **per-seed deterministic + order-independent
aggregation**, not serial-execution-identical. The pattern is
`mpa-solver`'s OpenMP discipline and `mpa-central/library/grind_library.py`'s
`ProcessPoolExecutor` pattern: each realization `k` uses an RNG seeded
deterministically from the top-level seed (e.g., `seed + k` or
`r * seed_step`), workers are independent (no shared mutable state),
and reductions (mean, SEM, sum) are commutative so aggregate outputs
do not depend on completion order.

This gives:

- **Same top-level seed → same per-realization outputs** in Python and
  every native backend.
- **Same aggregate output** regardless of worker count, completion
  order, or platform (within IEEE-754 platform tolerance for stochastic
  ops; byte-identical for deterministic ops).
- **Parallel execution unconstrained** — native backends use OpenMP /
  Rayon / tokio / GPU freely; Python uses `ProcessPoolExecutor` or
  `joblib`; both produce the same aggregate outputs as serial runs.

Deterministic operations (lookup dispatch, tangent-flow scaling,
continuous-flow on the Banach substrate, validation, provenance) are
trivially byte-identical across backends — no RNG to coordinate.
Stochastic operations (Bayesian inversion at v2; ensemble-based active
learning at v3) use the per-seed pattern and match within tolerance.

---

## Ecosystem integration

### Composition with mpa-solver

The two kernels do not call each other — both are consumed by
mpa-conform. But their outputs compose:

```
substrate observable
  → mpa-solver.fit_invariants → canonical observables (α_s, P_s, ...)
  → mpa-scale-solver.forward_sweep_invert → canonical state (chit, γ_AB)
  → mpa-scale-solver.tau_obs_sweep → canonical trajectory
  → mpa-conform.audit → AuditDelta
```

Each consumer handles its own composition; the solvers stay pure.

### Active learning

The solver identifies where a driver profile is weak (high-uncertainty
regions in canonical space, gamut edges with low gamut classification
confidence) and suggests measurements to fill the gap. Curator
operators consume these suggestions when planning library expansions.

### Inverse-lookup-table sidecar

Driver-profile artifacts ride with curator-precomputed inverse-lookup
tables. Solver dispatches table-first / compute-fallback:

- **Table-first**: nearest-neighbor / interpolated lookup. Sub-millisecond
  response. Banach substrate hits 100% of the time.
- **Compute-fallback**: full `forward_sweep_invert` or differentiable
  inversion. For ambiguity regions, table misses, or cache invalidation.

Solver itself stays pure (no caching logic in the operations); the
sidecar is consumed by the dispatch layer.

---

## User-facing

### Real-time τ_obs scrubbing

A user-facing slider over τ_obs produces a canonical trajectory in
sub-millisecond latency. Achieved via the inverse-lookup-table sidecar
plus interpolation between table grid points. The c→s→r migration
animates continuously as the camera moves.

### 3D / VR phase-portrait visualization

The canonical-space trajectory rendered in 3D, with the RG flow as a
vector field at each point. Optional VR target for immersive
substrate-space navigation. Default backend: three.js for browser; VR
extension when consumers ask.

### Visualization-first

Every operation has a default visualization:

- `apply_translation`: substrate-state plot vs canonical-state plot.
- `forward_sweep_invert`: residual landscape with the recovered point.
- `tau_obs_sweep`: trajectory in canonical space, regime-banded.
- `regime_at` / `gamut_classify`: the canonical-state point on the
  regime/gamut map.

Notebook integration shows the viz inline; programmatic use returns
the figure object.

### Mathematica-style exploration

Symbolic + numerical interplay. The user writes a query, gets back
both a closed-form expression (where one exists) and a numerical
evaluation. Operations compose: the output of one query feeds the
next without leaving symbolic space.

---

## Anti-goals

What the solver explicitly does NOT become, no matter how much compute
2026 has:

- **A physics solver.** Trajectory integration, observable extraction,
  closure dispatch — that's mpa-solver. The scale solver projects what
  physics produces; it does not produce physics.
- **An orchestration layer.** Bundle assembly, declaration trails,
  signing, MCP brokering for non-solver tools — that's mpa-conform.
- **A display layer.** Renderers, layout management, audit-engine
  display — that's mpa-auditor.
- **A spec author.** RFC text, schemas, framework prose — that's
  mpa-atlas. Spec questions surfaced by the solver route through
  mpa-conform's foundational-questions pipeline, never edited here.
- **A bespoke math source.** All math the solver implements has prior
  art with friendly attribution (per v9_receipts + cdv1_receipts
  bespoke→composition pass). The solver applies established
  mathematics in a unifying register; it does not invent new mathematics.

If any of these creep in — stop and rebalance. The framework's
sharpness comes from each component owning exactly one job.

---

## Reading order

When a session starts work toward this north star:

1. [`SUITE_BLOCK_IN.md`](../../mpa-central/SUITE_BLOCK_IN.md) — three-
   layer split.
2. This document — destination.
3. The session's specific handoff — immediate scope.
4. The relevant RFC-S sections + v9_receipts + cdv1_receipts entries —
   spec authority.
5. mpa-solver's CLAUDE.md — the sibling-kernel discipline.

Build toward this in small, deliberate sessions. Each session ships
one capability or one backend; the seven-operation API stays stable;
the trajectory accumulates.
