# Inverse-lookup-table sidecar

The v1 fast-path for `forward_sweep_invert`. The sidecar is a curator-
precomputed table that lets the solver short-circuit brute-force grid
search when the (substrate, tau_obs) pair is in the table.

## Type

```python
@dataclass(frozen=True)
class InverseLookupSidecar:
    version: str
    driver_profile_id: str
    driver_profile_version: str
    tau_obs_grid: tuple[float, ...]
    substrate_grid: tuple[SubstrateState, ...]
    canonical_grid: tuple[CanonicalState, ...]
    forward_lookup: dict[tuple[float, float, float], SubstrateState]
    inverse_lookup: dict[tuple[float, float, float], CanonicalState]
    ambiguity_regions: tuple[dict[str, Any], ...] = ()
```

## Key contract

Keys are 3-tuples `(chit, gamma_AB, tau_obs)` rounded to a fixed decimal
precision (default 6). Producers and consumers must agree on the
precision — `BanachSubstrate.build_sidecar` uses `DEFAULT_ROUNDING_DECIMALS`
from `sidecar.py`.

```python
from mpa_scale_solver.sidecar import round_key, DEFAULT_ROUNDING_DECIMALS
key = round_key((chit, gamma_AB, tau_obs))     # for forward lookup
```

For the inverse table, the key is derived from the substrate's
`observables["substrate_chit"]` and `observables["substrate_gamma_AB"]`
— the canonical curator-emitted observable names.

## Dispatch

```python
from mpa_scale_solver import forward_sweep_invert_wrapped

out = forward_sweep_invert_wrapped(
    substrate, field, tau_obs, candidate_grid,
    sidecar=sidecar,            # opt-in
)

out.provenance.dispatch_path    # TABLE_HIT | COMPUTE_FALLBACK | DIRECT_COMPUTE
out.provenance.table_version    # sidecar.version (None if no sidecar)
```

- **TABLE_HIT**: the (substrate, tau_obs) key was in the inverse table;
  the recorded canonical is returned. Sub-millisecond.
- **COMPUTE_FALLBACK**: a sidecar was provided but the key missed; the
  brute-force grid search ran. `table_version` is still populated so
  downstream consumers know which sidecar was consulted.
- **DIRECT_COMPUTE**: no sidecar was provided; brute-force only.

## Producer responsibility

Sidecar production is **mpa-conform's curator-path job**, not this
solver's. The curator builds the table by sweeping the driver profile's
gamut at the chosen `tau_obs_grid` and recording every (canonical,
substrate) pair plus any ambiguity regions discovered during the sweep.

v1 ships a single producer here: `BanachSubstrate.build_sidecar(grid)`,
which produces the Banach calibration sidecar used by the camera test.
This producer is intentionally trivial — it consults the analytical
truth directly. Real-substrate producers in mpa-conform are non-trivial
because the curator must actually run the full forward sweep.

## Ambiguity regions

`ambiguity_regions` records canonical-space zones where the forward map
is many-to-one (multiple canonicals project to the same substrate cell).
Consumers can choose to fall through to compute even on a table hit when
the hit lands in an ambiguity region, by checking the region records
before trusting the lookup. v1's Banach sidecar reports no ambiguity
regions (the identity translation is bijective).

## Reproducibility

Sidecars are deterministic functions of the driver profile + tau_obs
grid + rounding precision. Same inputs → byte-identical sidecar
contents → byte-identical solver outputs on table hits. Per-seed
reproducibility (parallel-friendly) applies to the producer side; the
consumer side is pure lookup with no RNG.

## When NOT to use a sidecar

- Adaptive or streaming inversion where the `(substrate, tau_obs)`
  distribution is not known in advance.
- Driver profiles still under development — the sidecar would invalidate
  on every profile edit.
- Profiles small enough that brute-force is already sub-millisecond.

The sidecar is opt-in everywhere; not passing one falls through to the
v0 brute-force path with no behavioral change.
