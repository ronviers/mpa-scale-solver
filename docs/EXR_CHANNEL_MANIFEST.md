# EXR channel manifest

**Authoritative spec — EXR Channel Contract v1.**

This document is the producer-side authority for the file-import boundary
between the compute layer (`mpa-solver` + `mpa-scale-solver`, assembled by
`mpa-conform`) and the rendering / viewer layer (`mpa-conform`'s
`particle_renderer.py`, the future `mpa-auditor` WebGL port). It declares
what channels appear in each per-frame EXR, who produces each channel,
and what value range / encoding each carries.

Consumers bind these channels to render properties via
`mpa-conform/conformer/shot/RENDERING_DISCIPLINE.md` §"What this rules IN"
and the executable spec `channel_to_emitter_params()` in
`mpa-conform/conformer/shot/particle_renderer.py`.

**Version-bump rule.** When this contract changes (channel added, encoding
changed, semantic shift), the version stamp here bumps in **lockstep**
with the consumer doc and the consumer function's docstring, in the same
change. Silent drift between producer and consumer is the failure mode
this versioning prevents.

Per handoff §D.3. The per-camera-frame EXR is `mpa-conform`'s assembly,
not this repo's emission. This document names what `mpa-scale-solver`
contributes and what comes from upstream / downstream.

## Channel ownership

| Channel | dtype | Source | Per-frame or trajectory? |
|---|---|---|---|
| RGB, A | uint8 → float32 in [0, 1] | matplotlib render | per-frame |
| chit | float32 | **mpa-scale-solver** (canonical state at this τ_obs) | per-frame |
| gamma_AB | float32 | **mpa-scale-solver** | per-frame |
| regime_label | float32 (encoded enum) | **mpa-scale-solver** (`regime_at`, 5-bucket) | per-frame |
| in_gamut | float32 (0 or 1) | **mpa-scale-solver** (`gamut_classify`) | per-frame |
| provenance_hash | float32 | **mpa-scale-solver** (`provenance_hash` on wrapped-op output; v1) | per-frame |
| validation_flags | float32 (bitfield) | **mpa-scale-solver** (`validation_flags_bitfield`; v1) | per-frame |
| X_c | float32 | mpa-solver (`fit_invariants`) | per-frame |
| X_r | float32 | mpa-solver | per-frame |
| alpha_s | float32 | mpa-solver | per-frame |
| P_s | float32 | mpa-solver | per-frame |
| N_f | float32 | mpa-solver | per-frame |
| beta_mem | float32 | mpa-solver (v2 of fit_invariants) | per-frame |
| Q | float32 | mpa-solver (cycles-of-headroom) | per-frame |
| I_pred | float32 | mpa-solver (predictive information) | per-frame |
| C_mu | float32 | mpa-solver (statistical complexity) | per-frame |
| window_mean | float32 | curator (raw substrate-side window-average) | per-frame |
| sem_chit, sem_X_c, sem_alpha_s, ... | float32 | curator (multi-realization SEM) | per-frame |
| trajectory_chit | float32 array | mpa-conform (composition across all frames) | trajectory |
| trajectory_regime | float32 array | mpa-conform | trajectory |
| trajectory_alpha_s | float32 array | mpa-conform | trajectory |

Trajectory channels pack into the EXR via a 1D image part (EXR supports
multipart files). One channel per per-frame observable; length = number
of frames in the sweep.

## What this repo emits

The six scale-solver channels — **chit**, **gamma_AB**, **regime_label**,
**in_gamut**, **provenance_hash**, **validation_flags** — come out of
five operations / helpers:

| Channel | Operation / helper | Field |
|---|---|---|
| chit | `forward_sweep_invert` | `recovered.chit` |
| gamma_AB | `forward_sweep_invert` | `recovered.gamma_AB` |
| regime_label | `regime_at` | `RegimeReading.regime` (enum, encoded as 0..4) |
| in_gamut | `gamut_classify` | `{"in_gamut": bool}` → 1.0 / 0.0 |
| provenance_hash | `provenance_hash(prov)` on `*_wrapped` output | float in [0, 1) — fingerprints (version, op, dispatch_path, table_version) |
| validation_flags | `validation_flags_bitfield(report)` | bit 0 = asymptotic_closure_compliant; bit 1 = k_frust_invariant; bit 2 = round_trip_residual present |

`regime_at` returns the **five-bucket** label. The encoding convention
for the EXR float32 channel:

```
0.0 = deep_c
1.0 = c_near_s
2.0 = s_critical
3.0 = r_near_s
4.0 = deep_r
```

Renderers that prefer the three-bucket display banding call
`regime_display_band(regime)` to collapse.

## What this repo does NOT emit

- No EXR files. The encoding layer is `mpa-conform`'s.
- No matplotlib rendering at production time. The camera test
  (`tests/test_camera_migration.py`) renders a PNG for visual inspection;
  that PNG is a test artifact, not a production output.
- No multi-realization SEM. Multi-realization is the curator path; this
  repo runs against a single substrate observation per frame.

## Units, ranges

- **chit**: dimensionless (cdv1 §chit unit). Range substrate-class-conditional;
  the seed corpus convention is roughly [-3, +3].
- **gamma_AB**: dimensionless cross-coupling (cdv1 §Universal two-mode
  kernel). Sign: < 0 cooperative, > 0 competitive.
- **regime_label**: enum, float32-encoded as above. Lossless under
  round-trip; downstream consumers must read the encoding from this
  manifest (or call `regime_display_band` on the deserialized label).
- **in_gamut**: 1.0 / 0.0. Frames where the canonical state is outside
  the substrate's declared gamut carry 0.0 here; consumers should consult
  `intent_map` output in adjacent channels before interpreting.
- **provenance_hash**: float in `[0, 1)`. Stable across runs; differs by
  operation + dispatch_path + table_version. Consumers compare two
  frames' hashes to detect dispatch-path drift without unpacking the
  full provenance payload.
- **validation_flags**: small integer cast to float32 (0..7). Bit 0:
  asymptotic_closure_compliant; bit 1: k_frust_invariant; bit 2:
  round_trip_residual present (operation actually computed a round
  trip).

## Custom channels — experimental slots

Six reserved slots let experimental framework data ride the EXR pipeline
without bumping the contract version each iteration:

| Channel | dtype | Convention |
|---|---|---|
| `custom_0` .. `custom_3` | float32 | per-frame scalar; producer names the meaning in its session handoff |
| `custom_lut_0` | float32 array, length ≤ 256 | 1D look-up table; e.g., regime_label → color override, chit → altitude mapping |
| `custom_lut_1` | float32 array, length ≤ 256 | second LUT slot |

Custom slots are **not versioned** with the contract. A producer can
populate `custom_0` with any framework-derived quantity (a tweak, an
experiment, a graduation candidate) without bumping the contract version.
Consumers treat unused / unpopulated slots as zeros.

**Discipline carries.** Even a custom-slot channel must derive from
framework data — `mpa-conform/conformer/shot/RENDERING_DISCIPLINE.md`
applies identically here. "Make it look prettier" populates that don't
trace to a framework quantity are decoration and forbidden.

**Graduation.** When a custom slot proves load-bearing across multiple
sessions, it graduates to a committed channel: bump the contract version,
add the channel to the ownership table above, retire its custom slot,
and bind it explicitly in the consumer mapping.

**Custom slot tracking.** When a slot is in active experimental use,
add a one-line entry below naming the producer, the framework quantity
it carries, and the session that flagged it. Remove the entry when the
slot is retired (graduated or abandoned).

*Currently in use:* none.
