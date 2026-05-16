# Tangent-flow translation field

The v1 second translation-field shape (RFC-S Appendix B item 1 canonical
leading-order auto-remap). Sibling of `TranslationField` (lookup_table);
both ride the same `apply_translation` callsite, dispatched on
`field.shape`.

## Why this shape exists

Lookup-table fields ship a list of canonical → substrate rules sampled
on a grid. They handle arbitrary substrate-specific structure but they
do not expose the underlying generator: continuous flow, derivatives,
analytical inversion all sit outside the schema.

Tangent-flow fields carry the leading-order generator directly. The
`ScalingRule` is a closed-form auto-remap that maps the canonical state
through the substrate's reference operating point under `tau_obs`
scaling. The Banach substrate's `gamma`-scaling rule (v9 §Scale-
relativity) is the canonical leading-order form; substrate-conditional
refinements ride `ScalingRule.refinement`.

## Schema

```python
@dataclass(frozen=True)
class TangentFlowField:
    direction: Literal["forward"]
    shape: Literal["tangent_flow"]
    rule_at_origin: TranslationRule   # canonical reference point + substrate identity
    scaling: ScalingRule              # leading-order tangent-flow auto-remap
    description: Optional[str] = None

@dataclass(frozen=True)
class ScalingRule:
    tau_obs_ref: float                # reference camera depth
    delta_gamma: float = 0.0          # gamma_AB scales as (tau/tau_ref) ** delta_gamma
    delta_chit: float = 0.0           # chit drifts by delta_chit * ln(tau/tau_ref)
    refinement: Optional[dict] = None # substrate-conditional overrides
```

## apply_translation semantics

```python
substrate_chit  = canonical.chit  + delta_chit  * ln(tau_obs / tau_obs_ref)
substrate_gamma = canonical.gamma * (tau_obs / tau_obs_ref) ** delta_gamma
```

The returned `SubstrateState`:

- `tau_obs` = the call-site argument
- `label` = `rule_at_origin.operating_point.label`
- `axes` = `rule_at_origin.operating_point.axes` plus the call-site
  `tau_obs`
- `observables` = `{"substrate_chit": scaled_chit, "substrate_gamma_AB": scaled_gamma}`

For the Banach default (`delta_chit = delta_gamma = 0`), the substrate
observables equal the canonical values — identity translation.

## Banach canonical form

`BanachSubstrate.translation_field()` returns:

```python
TangentFlowField(
    direction="forward",
    shape="tangent_flow",
    rule_at_origin=TranslationRule(
        operating_point=OperatingPoint(label="banach_origin", gt=...),
        xdot_choice="identity",
        canonical=CanonicalPoint(chit=chit_0, gamma_AB=gamma_AB_0, k_frust=False, method="banach_canonical"),
    ),
    scaling=ScalingRule(
        tau_obs_ref=1.0,
        delta_gamma=0.0,
        delta_chit=0.0,
        refinement={
            "flow_kind": "banach_exponential",
            "lambda_chit": 1.0,
            "lambda_gamma": 1.0,
        },
    ),
)
```

The `refinement` dict's `banach_exponential` `flow_kind` is read by
`flow()` (not by `apply_translation`) — translation is identity at the
substrate level; the RG flow happens in canonical space and is computed
by `flow()` via the closed-form exp decay.

## Real-substrate refinement

Real substrates ship a `ScalingRule` with non-zero `delta_chit` or
`delta_gamma` capturing their substrate-conditional drift. The
`refinement` dict carries higher-order or substrate-specific terms that
`apply_translation` (at v1) does not interpret — they ride for downstream
consumers and v2 differentiable backends.

## When to choose tangent-flow over lookup-table

| Use case | Shape |
|---|---|
| Curator-precomputed dense sampling, no closed form available | `lookup_table` |
| Closed-form leading-order rule available; want differentiability later | `tangent_flow` |
| Banach calibration reference | `tangent_flow` (the Banach instance) |
| Substrate with known scaling-law structure (power-law tails, log drift) | `tangent_flow` |

Mixed-shape profiles (lookup_table + tangent_flow per substrate-class)
are not v1; the field shape is single-valued per driver profile. The
schema bump (v2.0 → v3.0) for mixed-shape support is out of scope here.
