# Wishlist — ambient projection-layer ideas

Parking lot for scale-solver extensions not yet in
`ORDER_OF_OPERATIONS.md` or `NORTH_STAR.md`. Dated entries, one
paragraph each, newest-first within a theme. When a session lands an
item, **delete the entry** — this file decays as work moves through it.

**Scope.** Scale-solver concerns: per-frame inversion, $\tau_{\text{obs}}$
sweeps, canonical-space projection, validation/provenance,
gamut/posterior emission. Anything upstream of that (data ingestion,
substrate-class adapters) belongs in `mpa-conform`'s curator path.
Anything downstream (EXR ↔ shot rendering, DJV review) belongs in
`mpa-conform`'s shot pipeline or `mpa-auditor`.

Items currently driven by the character-renderer consumer (the
unfolding cdv1 dissipation visualisation). When the character
renderer is the only thing asking, say so in the entry.

---

## $\tau_{\text{obs}}$ semantics

### $\tau_{\text{obs}}$ aggregation window as separate parameter *(flagged 2026-05-17)*

Currently $\tau_{\text{obs}}$ is the frame coordinate — a single
observer timescale per inversion call. Over-sampled physics
substrates (QEC syndrome streams at ~1 μs cycle, ~10⁵ cycles per
shot) need an **aggregation window** as a separate degree of freedom:
the canonical-space inversion at frame $f$ uses substrate observations
within $[\tau_{\text{obs}}(f) - W/2,\ \tau_{\text{obs}}(f) + W/2]$. Window
$W$ becomes a render parameter the character renderer scrubs against.
Without it, every frame is a delta function on substrate time and the
RG-flow trajectory is forced to render at the substrate's native
resolution.

### Canonical-space interpolation mode for under-sampled frames *(flagged 2026-05-17)*

Under-sampled substrates (behavioural at ~1 trial/s, ecological at
season cadence) leave inter-observation frames without a native
inversion. Wish: an interpolation mode that interpolates in
**canonical space** (between recovered $(\chi, \gamma_{AB})$ points)
rather than substrate space. Canonical-space interpolation respects
the smooth manifold cdv1 actually predicts; substrate-space
interpolation invents drift between known points that may not respect
the kernel. Inversion residual on interpolated frames is meaningless
by construction — needs its own flag (see below).

---

## Validation / provenance

### Data-quality bits in `validation_flags` *(flagged 2026-05-17)*

`validation_flags` is a 3-bit field today (asymptotic-closure / k_frust
preserved / round-trip computed). Wish: extend to a per-frame quality
quartet — `native` (inversion on a real observation window),
`aggregated` (averaging across N native samples; record N), `interpolated`
(canonical-space fill between observations), `off_gamut` (already a
separate channel; reflected here for compactness). Character renderer
reads these to drive sharpness/confidence-haze per frame, per
rendering discipline ("every visual property maps to framework data").

---

## Multi-level / tower

### Per-(frame × level) canonical states *(flagged 2026-05-17)*

cdv1 §Heat-tax tower routes dissipation upward through levels with
parameter-feeding ($L_{n+1}$ receives $\alpha_\sigma\langle\sigma_n\rangle + \alpha_\Sigma\langle\Sigma_n\rangle$). For a
multi-level character render, the scale-solver's per-frame envelope
needs to extend to **per-(frame × level)**: a stack of canonical
states per frame, one per tower level. EXR channel manifest already
admits multi-layer; the inversion API does not yet vectorise across
levels. Wishes one of: (a) a `level` axis on `forward_sweep_invert_stream`,
or (b) a `tower_invert` operation that takes a vector of substrate
states (one per level) and returns the stack.

---

## Comparison lens

### Substrate parallax channels *(flagged 2026-05-17)*

Per the comparison-lens design (see memory + sibling-repo
`mpa-conform`), each substrate carries native channel-richness that
the 2-channel canonical display strips. Wish: emit substrate-parallax
channels alongside canonical channels — per-window viewpoints that
preserve the channel-richness the substrate observation natively
carried (e.g., for surface-code QEC, the full syndrome-graph
topology; for behavioural, the trial-by-trial response distribution).
Render-side decides what to show; scale-solver decides what to
preserve. Schema needs to be open enough that adding a new substrate
class doesn't force a re-spec.

---

## Contract evolution

### Custom-slot tracking + version-bump convention *(flagged 2026-05-17)*

`docs/EXR_CHANNEL_MANIFEST.md` v1 added six experimental custom slots
(`custom_0`..`custom_3` scalar; `custom_lut_0`, `custom_lut_1` 1D
look-up tables) plus a graduation path. Wish: keep the "Currently in
use" appendix table accurate as slots fill and empty, and apply the
version-bump-in-lockstep rule when a slot graduates to a committed
channel. The rule applies to the producer manifest, the consumer
`mpa-conform/conformer/shot/RENDERING_DISCIPLINE.md`, and the
executable spec `channel_to_emitter_params()` — all in the same change.

### Tower-level (multi-level) channel block *(flagged 2026-05-17)*

When the multi-level / tower projection wishlist item lands (per-frame
× per-level canonical states for the heat-tax composite stack), the
contract grows from per-frame to per-(frame × level). Channel layout
will need either a per-level suffix convention (`chit_0`, `chit_1`,
...) or an EXR multi-part part-per-level layout. Decide at landing
time per single-move discipline, not now. Bumps the contract version.

## Reference-substrate dispatch

### Banach reference plate alongside real substrate in same frame *(flagged 2026-05-17)*

Banach substrate is the analytical reference; `SelfTestCadence`
already validates against it on every k-th frame. Wish: a dispatch
mode that emits the Banach analytical canonical state **in the same
frame** as the real substrate's inverted canonical state — paired
channels for in-shot ground truth. Character renderer can then show
the real substrate and the Banach reference side by side (or
overlaid) without a separate render pass. This is the in-frame
analog of the daily's reference-plate convention.
