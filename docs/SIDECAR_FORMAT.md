# Inverse-lookup-table sidecar — wire-format spec

**Authoritative.** This document defines the cross-language wire format
of `InverseLookupSidecar`. Producers (mpa-conform's curator path, the
`BanachSubstrate.build_sidecar` reference producer) MUST emit JSON
conforming to this spec. Consumers (`mpa_scale_solver` Python via
`decode_sidecar_from_json`, the Rust port via
`serde_json::from_str::<InverseLookupSidecar>`) read it byte-losslessly.

Wire version: **1.0**.

---

## What a sidecar is

A curator-precomputed table that lets `forward_sweep_invert`
short-circuit brute-force grid search when the `(substrate, tau_obs)`
pair is in the table. Opt-in everywhere; not passing one falls through
to the v0 brute-force path with no behavioral change.

Dispatch contract (Python and Rust agree):

- **TABLE_HIT** — the `(substrate, tau_obs)` key was in the inverse
  table; the recorded canonical is returned. Sub-millisecond.
- **COMPUTE_FALLBACK** — a sidecar was provided but the key missed;
  the brute-force grid search ran. `table_version` is still populated
  on the provenance so downstream consumers know which sidecar was
  consulted.
- **DIRECT_COMPUTE** — no sidecar was provided; brute-force only.

---

## Top-level JSON shape

```jsonc
{
  "wire_version": "1.0",                  // required string; this spec
  "version": "1.0.0",                     // producer's version stamp
  "driver_profile_id": "<id>",            // string; producer-defined
  "driver_profile_version": "<version>",  // string; producer-defined
  "rounding_decimals": 6,                 // int; default 6
  "tau_obs_grid": [<f64>, ...],
  "substrate_grid": [<SubstrateState>, ...],
  "canonical_grid": [<CanonicalState>, ...],
  "forward_lookup": { "<key>": <SubstrateState>, ... },
  "inverse_lookup": { "<key>": <CanonicalState>, ... },
  "ambiguity_regions": [{<arbitrary>}, ...]   // optional; default []
}
```

Field semantics:

| Field | Contract |
|---|---|
| `wire_version` | This spec's version. Consumers MUST refuse unknown major versions; minor-version mismatches indicate additive changes that older consumers MAY tolerate (they default the missing optional fields per the serde defaults). |
| `version` | The producer's own stamp on this particular sidecar artifact. Rides into `Provenance.table_version` on every TABLE_HIT / COMPUTE_FALLBACK dispatch so consumers can tell which sidecar was consulted. |
| `driver_profile_id` / `driver_profile_version` | Identify the driver profile the sidecar was built for. Solver does not validate; consumers MAY cross-check against the field they're inverting against. |
| `rounding_decimals` | The decimal precision the producer rounded keys at (see §Key encoding). Consumers MUST round their query keys at the same precision. Default 6. |
| `tau_obs_grid` | The τ_obs frames the producer swept over. Used for fast-path frame iteration; not load-bearing for lookup hits. |
| `substrate_grid` / `canonical_grid` | Parallel arrays: index `i` is the substrate the producer recorded at `tau_obs_grid[i]` for `canonical_grid[i]`. Same constraint — informational, not used for lookup. |
| `forward_lookup` | `(canonical, tau_obs) → substrate`. Keys per §Key encoding; values per §SubstrateState shape. |
| `inverse_lookup` | `(substrate, tau_obs) → canonical`. Keys derived from the substrate's `observables["substrate_chit"]` and `observables["substrate_gamma_AB"]`; values per §CanonicalState shape. |
| `ambiguity_regions` | Arbitrary JSON objects, one per multi-valued inverse zone. Curator-defined per-region metadata; consumers MAY use these to fall through to compute even on a hit. Default empty. |

---

## Key encoding

Lookup-dict keys are strings of the form

```
<chit_bits>:<gamma_AB_bits>:<tau_obs_bits>
```

where each `*_bits` is the **decimal string** of `f64::to_bits(x)` —
the IEEE-754 binary representation as an unsigned 64-bit integer.
For example, the rounded triple `(0.5, -0.3, 1.0)` encodes as

```
4602678819172646912:13821247184028831027:4607182418800017408
```

The decimal string of the raw bits is chosen over hex because it
matches the existing Rust `SidecarKey::serialize` shape exactly. The
representation is lossless: `bits_to_float(float_to_bits(x)) == x` for
all finite f64.

### Rounding algorithm

Producers MUST round the float key before encoding. The algorithm is

```
rounded(x, n) = IEEE-754 roundTiesToEven(x * 10^n) / 10^n
```

Concretely:

- **Python**: `float(np.rint(x * 10**n)) / 10**n` —
  `mpa_scale_solver.sidecar.round_key`.
- **Rust**: `(x * 10f64.powi(n)).round_ties_even() / 10f64.powi(n)` —
  `mpa_scale_solver::sidecar::round_decimal`.

These produce bit-identical f64 for every finite `x`. **Do not use
Python's `round(x, n)` builtin** — it uses dtoa-based decimal rounding
that diverges from `roundTiesToEven` on `.x5`-decimal binary halfway
cases. The session-7 retro on `serde_json/float_roundtrip` and the
sidecar v1 → v1.0 spec-formalisation discovered this; the
Python-side fix landed in `sidecar.round_key` at the same time.

Non-finite inputs (NaN / ±∞) pass through unchanged. NaN keys are
allowed in principle but pathological in practice; the curator SHOULD
NOT emit them.

---

## SubstrateState shape

```json
{
  "tau_obs": <f64>,
  "label": "<string>" | null,
  "axes": { "<key>": <any JSON value>, ... },
  "observables": { "<key>": <f64>, ... }
}
```

`tau_obs` is the τ_obs the substrate observation was taken at.
`label` is an optional human-readable tag. `axes` carries arbitrary
JSON-typed substrate axes (the producer's choice). `observables` is
a flat string→f64 map.

The inverse-lookup keying convention requires
`observables["substrate_chit"]` and `observables["substrate_gamma_AB"]`
to be present on substrates the consumer plans to look up. Substrates
lacking those keys are a guaranteed lookup miss (consumer falls
through to compute).

---

## CanonicalState shape

```json
{
  "chit": <f64>,
  "gamma_AB": <f64>,
  "k_frust": <bool>
}
```

The frozen canonical-state dataclass. Field names match the Python
struct verbatim (including the unconventional `gamma_AB` mixed-case
name — this is load-bearing for cross-language parity).

---

## Wire-version evolution

- **Additive changes** (new optional top-level fields, new optional
  `SubstrateState`/`CanonicalState` fields with serde defaults) MAY
  land at a minor version bump (1.0 → 1.1). Older consumers tolerate
  them by defaulting the missing fields.
- **Breaking changes** (renamed fields, changed key encoding, changed
  rounding algorithm) MUST bump the major version (1.0 → 2.0). Older
  consumers MUST refuse the artifact rather than guess.

The major version is `wire_version.split(".")[0]`.

---

## Test of conformance

Cross-language parity is exercised by
`rust/tests/bit_identity.rs::sidecar_python_to_rust_parity`:

1. Python builds a small `BanachSubstrate` sidecar at a fixed
   `tau_obs` grid.
2. Python encodes via `encode_sidecar_to_json` and emits to JSON
   (the fixture entry `sidecar_wire_format` in
   `jax_core_reference.json`).
3. Rust deserializes via `serde_json::from_str::<InverseLookupSidecar>`.
4. For each `(canonical, tau)` in the grid: Rust's
   `lookup_forward` returns the expected substrate (bit-identical
   field values), and Rust's `lookup_inverse` returns the expected
   canonical.

A producer claiming to emit this format MUST pass an analogous
cross-language round-trip test against the Rust consumer.

---

## Producer responsibility

Sidecar production is **mpa-conform's curator-path job**, not this
solver's. The curator builds the table by sweeping the driver profile's
gamut at the chosen `tau_obs_grid` and recording every (canonical,
substrate) pair plus any ambiguity regions discovered during the sweep.

`mpa_scale_solver.banach.BanachSubstrate.build_sidecar` is the v1
reference producer (used by the camera test). It is intentionally
trivial — it consults the analytical Banach truth directly.
Real-substrate producers in mpa-conform are non-trivial because the
curator must actually run the full forward sweep.

## Reproducibility

Sidecars are deterministic functions of the driver profile + tau_obs
grid + rounding precision + rounding algorithm. Same inputs →
byte-identical sidecar contents → byte-identical solver outputs on
table hits. Per-seed reproducibility (parallel-friendly) applies to
the producer side; the consumer side is pure lookup with no RNG.

## When NOT to use a sidecar

- Adaptive or streaming inversion where the `(substrate, tau_obs)`
  distribution is not known in advance.
- Driver profiles still under development — the sidecar would
  invalidate on every profile edit.
- Profiles small enough that brute-force is already sub-millisecond.

The sidecar is opt-in everywhere; not passing one falls through to
the v0 brute-force path with no behavioral change.
