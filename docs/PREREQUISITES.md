# Prerequisites

## mpa-solver Python bindings: `fit_invariants`

The mpa-solver C++/WASM kernel ships
`fit_invariants(locus) → {X_c, X_r, alpha_s, P_s, N_f, regime}`. The
Python binding in `mpa-conform/conformer/compute/observables.py` has
`correlator`, `response_direct`, and `gfdr_locus` — but not yet
`fit_invariants`.

**Why this matters here.** Once a real driver profile is being consumed
end-to-end (curator-path output → mpa-solver observables →
mpa-scale-solver projection → mpa-conform bundle), the scale-solver's
integration test against real substrates needs `fit_invariants` to extract
the canonical observables from the substrate-native locus. The synthetic
camera test in this repo does not need `fit_invariants` because the
synthetic carries its own analytical truth.

**Action.** The port is a one-shot session in `mpa-solver` (or
`mpa-conform`'s observable bindings). Until that lands, integration tests
against real driver profiles run in scale-solver as substrate-cell-closure
checks (`tests/test_seed_corpus.py`), not as observable-fitting checks.

## Out of scope for this repo

- `fit_invariants` lives in `mpa-solver`, not here. Do not port it into
  this repo.
- Bundle orchestration that consumes both `mpa-solver` observables and
  `mpa-scale-solver` projections is `mpa-conform`'s job.

## Schema version pin

This repo consumes `mpa-atlas/schema/driver-profile.v2.0.json`. The
schema's `direction = "forward"` and `shape = "lookup_table"` are
**Literal type pins**, not enums with branches — do not write dispatch
logic on them. Future v3 features (tangent-flow form, backward map) will
ship a new schema and a new types module, paralleled rather than
extended.
