//! The seven scale-solver operations — port of
//! `mpa_scale_solver/operations.py`.
//!
//! Scope landed:
//!   * Sessions 1-3: math primitives + types.
//!   * Session 4 — raw forward path:
//!     - `apply_translation` + the three field-shape dispatch helpers.
//!     - `forward_sweep_invert_grid` (Python `method="grid"`).
//!     - `tau_obs_sweep_grid` (forced grid path).
//!     - `regime_at` + `regime_display_band`.
//!     - `gamut_classify`.
//!   * Session 5 — gradient inversion:
//!     - `forward_sweep_invert` dispatcher with `Method::{Auto, Grid,
//!       Gradient}` (Python `method` kwarg).
//!     - Closed-form tangent-flow inverse routes through
//!       `math::tangent_flow_canonical_inverse` (session-1 bit-identity
//!       tested).
//!     - LearnedField L-BFGS substituted by a hand-rolled 2D damped-
//!       Newton solver in `optim.rs` (BLOCK_IN §v6 explicitly carves out
//!       non-byte-identity vs scipy's L-BFGS-B; see optim.rs for
//!       rationale on the deviation from the BLOCK_IN-noted `argmin`
//!       candidate).
//!   * Session 6 — intent algebra (RFC-S §3 cut d):
//!     - `intent_map(state, tau_obs, gamut, IntentId)` and the five
//!       private `intent_iN` handlers + helpers.
//!     - `intent_compose(state, tau_obs, gamut, &[IntentId])` enforcing
//!       the I2-doesn't-compose rule (Python's `ValueError` ports as
//!       `OperationError::I2InComposition`).
//!     - `SacrificeRecord` + `IntentDiagnostics` (flat-dict JSON shape
//!       matching Python's `sac` output via
//!       `#[serde(flatten)] + #[serde(tag = "intent")]`).
//!
//! Deferred to subsequent sessions (named in BLOCK_IN §v6):
//!   * `validate_driver_profile` + the `*_wrapped` variants (session 7)
//!   * `forward_sweep_invert_posterior` (session 8)
//!
//! Python's `score_fn` / `forward_map` callable parameters are surfaced
//! as `Option<&dyn Fn(...)>` so callers can pass `None` without juggling
//! type parameters. Dynamic dispatch overhead is negligible compared to
//! the per-candidate forward map.

#![allow(non_snake_case)]

use std::collections::BTreeMap;

use serde_json::Value;

use crate::math::{
    Activation, MlpLayer, laplace_covariance_from_jacobian, learned_field_substrate,
    lookup_squared_distance, slogdet_2x2, tangent_flow_canonical_inverse,
    tangent_flow_forward_jacobian, tangent_flow_substrate,
};
use crate::optim::minimize_smooth_2d;
use crate::provenance::make_provenance;
use crate::sidecar::{DEFAULT_ROUNDING_DECIMALS, lookup_forward, lookup_inverse};
use crate::types::{
    CanonicalState, CapacityClass, DispatchPath, DisplayBand, GamutSpec, InverseLookupSidecar,
    IntentDiagnostics, IntentId, LearnedField, LookupTableField, OperationOutput, Posterior,
    RegimeLabel, RegimeReading, SacrificeRecord, SubstrateState, TangentFlowField,
    TranslationField, TranslationRule,
};
use crate::validation::{
    self as _validation, DriverProfileSummary, per_intent_cell_metric, aggregate_per_intent_metrics,
};

use crate::gfdr_model::vertex_regime;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors `apply_translation` / `forward_sweep_invert` raise in lieu of
/// Python's `ValueError`.
#[derive(Debug, Clone, PartialEq)]
pub enum OperationError {
    /// Lookup translation field has zero rules.
    EmptyTranslationField,
    /// Lookup translation field's nearest rule lies beyond the
    /// declared domain threshold. Carries the distance and threshold
    /// so callers can build a diagnostic.
    OutsideDomain {
        distance: f64,
        threshold: f64,
    },
    /// Per-frame target-list length doesn't match the `tau_obs_grid`.
    TargetGridLengthMismatch {
        targets: usize,
        frames: usize,
    },
    /// Grid had zero candidate rows — `forward_sweep_invert` requires
    /// at least one.
    EmptyCanonicalGrid,
    /// `Method::Gradient` requested for a `LookupTableField`. Mirrors
    /// Python's `ValueError("method='gradient' requires a differentiable
    /// field ...")`. Lookup tables have no differentiable surface — use
    /// `Method::Grid` or `Method::Auto` (which routes lookup_table to
    /// grid).
    GradientOnLookupTable,
    /// `intent_compose` called with an empty intent list. Mirrors
    /// Python's `ValueError("intent_compose requires at least one intent")`.
    IntentComposeEmpty,
    /// `intent_compose` called with I2 alongside other intents. Per
    /// RFC-S §3: I2 (drive-faithful) does not compose with adjusting
    /// intents. Mirrors Python's `ValueError("I2 ... does not compose ...")`.
    I2InComposition,
    /// `forward_sweep_invert_posterior` dispatched on a `LookupTableField`
    /// without the `canonical_grid` argument the brute-force inversion
    /// needs. Mirrors Python's `ValueError("forward_sweep_invert_posterior
    /// requires canonical_grid for lookup_table fields ...")`.
    PosteriorRequiresCanonicalGrid,
    /// `forward_sweep_invert_posterior` dispatched on a translation-field
    /// shape that has no posterior implementation. Currently `LearnedField`
    /// has no Laplace surface (the posterior would need MAP + Jacobian +
    /// covariance through the MLP; deferred to v6.x if a consumer needs
    /// it). Mirrors Python's `TypeError("unsupported translation field
    /// type: LearnedField")`.
    PosteriorUnsupportedFieldShape,
}

impl std::fmt::Display for OperationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyTranslationField => {
                write!(f, "translation field has no rules")
            }
            Self::OutsideDomain {
                distance,
                threshold,
            } => write!(
                f,
                "canonical state outside translation field domain: \
                 nearest rule distance {distance} > threshold {threshold}"
            ),
            Self::TargetGridLengthMismatch { targets, frames } => write!(
                f,
                "per-frame target list length {targets} != tau_obs_grid length {frames}"
            ),
            Self::EmptyCanonicalGrid => {
                write!(f, "canonical_grid must contain at least one candidate")
            }
            Self::GradientOnLookupTable => write!(
                f,
                "method='gradient' requires a differentiable field \
                 (TangentFlowField or LearnedField); got LookupTableField \
                 (use method='grid' for lookup_table)."
            ),
            Self::IntentComposeEmpty => {
                write!(f, "intent_compose requires at least one intent")
            }
            Self::I2InComposition => write!(
                f,
                "I2 (drive-faithful) does not compose with adjusting intents \
                 (RFC-S §3). Call intent_map(IntentId::I2) directly."
            ),
            Self::PosteriorRequiresCanonicalGrid => write!(
                f,
                "forward_sweep_invert_posterior requires canonical_grid for \
                 lookup_table fields (the search grid the brute-force \
                 inversion uses)."
            ),
            Self::PosteriorUnsupportedFieldShape => write!(
                f,
                "forward_sweep_invert_posterior: no posterior implementation \
                 for this translation field shape (LearnedField has no Laplace \
                 surface in v5; tangent_flow + lookup_table are supported)."
            ),
        }
    }
}

impl std::error::Error for OperationError {}

/// Mirrors Python's `DEFAULT_DOMAIN_DISTANCE_THRESHOLD = 1e9` — tables
/// specify their own threshold; this is "effectively off".
pub const DEFAULT_DOMAIN_DISTANCE_THRESHOLD: f64 = 1e9;

// NOTE: these are kept as documentation aliases only and are not used in
// public signatures. A bare `dyn Trait` alias defaults to `dyn Trait +
// 'static`, which would lock `forward_map` / `score_fn` arguments to
// `'static` closures and forbid capturing local state. The public
// functions inline `&dyn Fn(...)` so the trait-object lifetime defaults
// to the enclosing reference's lifetime (per Rust reference §"Default
// trait object lifetimes"), permitting non-`'static` closures.
#[allow(dead_code)]
pub type ScoreFn = dyn Fn(&SubstrateState, &SubstrateState) -> f64;
#[allow(dead_code)]
pub type ForwardMap = dyn Fn(&CanonicalState, f64) -> SubstrateState;

// ---------------------------------------------------------------------------
// TranslationFieldIndex — pre-cached view for fast lookup
// ---------------------------------------------------------------------------

/// Pre-cached view of a `LookupTableField`'s rule canonicals. Callers
/// that apply the same field many times (the camera test,
/// `forward_sweep_invert` at repeated `tau_obs` frames, `tau_obs_sweep`)
/// build one index and reuse it.
pub struct TranslationFieldIndex<'a> {
    rules: &'a [TranslationRule],
    chits: Vec<f64>,
    gammas: Vec<f64>,
    taus: Vec<f64>,
    has_tau: Vec<bool>,
}

impl<'a> TranslationFieldIndex<'a> {
    pub fn new(field: &'a LookupTableField) -> Self {
        let n = field.rule.len();
        let mut chits = Vec::with_capacity(n);
        let mut gammas = Vec::with_capacity(n);
        let mut taus = Vec::with_capacity(n);
        let mut has_tau = Vec::with_capacity(n);
        for rule in &field.rule {
            chits.push(rule.canonical.chit);
            gammas.push(rule.canonical.gamma_AB);
            match rule
                .operating_point
                .axes
                .get("tau_obs")
                .and_then(Value::as_f64)
            {
                Some(t) => {
                    taus.push(t);
                    has_tau.push(true);
                }
                None => {
                    taus.push(f64::NAN);
                    has_tau.push(false);
                }
            }
        }
        Self {
            rules: &field.rule,
            chits,
            gammas,
            taus,
            has_tau,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    pub fn len(&self) -> usize {
        self.rules.len()
    }

    /// Return `(rule_index, squared_distance)` for the nearest rule.
    /// Wraps `math::lookup_squared_distance` + an argmin pass (first-min
    /// matches `np.argmin`).
    pub fn nearest(&self, chit: f64, gamma_AB: f64, tau_obs: f64, tau_obs_weight: f64) -> (usize, f64) {
        let d2 = lookup_squared_distance(
            chit,
            gamma_AB,
            &self.chits,
            &self.gammas,
            &self.taus,
            &self.has_tau,
            tau_obs,
            tau_obs_weight,
        );
        let mut best_idx = 0;
        let mut best = d2[0];
        for (i, v) in d2.iter().enumerate().skip(1) {
            if *v < best {
                best = *v;
                best_idx = i;
            }
        }
        (best_idx, best)
    }
}

// ---------------------------------------------------------------------------
// Op 1: apply_translation
// ---------------------------------------------------------------------------

/// Forward map: canonical state → substrate-native at `tau_obs`.
///
/// Dispatches on `field` variant:
/// - `LookupTable`: nearest-rule lookup (L2 + optional log-tau term).
/// - `TangentFlow`: closed-form scaling rule.
/// - `Learned`: MLP forward map (`math::learned_field_substrate`).
pub fn apply_translation(
    canonical: &CanonicalState,
    field: &TranslationField,
    tau_obs: f64,
    domain_distance_threshold: f64,
    tau_obs_weight: f64,
) -> Result<SubstrateState, OperationError> {
    match field {
        TranslationField::TangentFlow(tf) => Ok(apply_tangent_flow(canonical, tf, tau_obs)),
        TranslationField::Learned(lf) => Ok(apply_learned(canonical, lf, tau_obs)),
        TranslationField::LookupTable(lt) => {
            let index = TranslationFieldIndex::new(lt);
            apply_lookup_with_index(canonical, &index, tau_obs, domain_distance_threshold, tau_obs_weight)
        }
    }
}

/// Variant of `apply_translation` that takes a pre-built index — avoids
/// rebuilding per call when the same lookup field is applied many times
/// (the `tau_obs_sweep` + grid-search hot loop).
pub fn apply_translation_indexed(
    canonical: &CanonicalState,
    index: &TranslationFieldIndex<'_>,
    tau_obs: f64,
    domain_distance_threshold: f64,
    tau_obs_weight: f64,
) -> Result<SubstrateState, OperationError> {
    apply_lookup_with_index(canonical, index, tau_obs, domain_distance_threshold, tau_obs_weight)
}

fn apply_lookup_with_index(
    canonical: &CanonicalState,
    index: &TranslationFieldIndex<'_>,
    tau_obs: f64,
    domain_distance_threshold: f64,
    tau_obs_weight: f64,
) -> Result<SubstrateState, OperationError> {
    if index.is_empty() {
        return Err(OperationError::EmptyTranslationField);
    }
    let (idx, d2) = index.nearest(canonical.chit, canonical.gamma_AB, tau_obs, tau_obs_weight);
    if d2 > domain_distance_threshold * domain_distance_threshold {
        return Err(OperationError::OutsideDomain {
            distance: d2.sqrt(),
            threshold: domain_distance_threshold,
        });
    }
    let rule = &index.rules[idx];
    let mut observables = BTreeMap::new();
    observables.insert("canonical_chit".to_string(), rule.canonical.chit);
    observables.insert("canonical_gamma_AB".to_string(), rule.canonical.gamma_AB);
    Ok(SubstrateState {
        tau_obs,
        label: Some(rule.operating_point.label.clone()),
        axes: rule.operating_point.axes.clone(),
        observables,
    })
}

fn apply_learned(canonical: &CanonicalState, field: &LearnedField, tau_obs: f64) -> SubstrateState {
    let (s_chit, s_gamma) = learned_field_substrate(
        canonical.chit,
        canonical.gamma_AB,
        tau_obs,
        field.tau_obs_ref,
        field.weights.as_slice(),
        field.activation,
    );
    let mut axes = field.rule_at_origin.operating_point.axes.clone();
    axes.insert("tau_obs".to_string(), Value::from(tau_obs));
    let mut observables = BTreeMap::new();
    observables.insert("substrate_chit".to_string(), s_chit);
    observables.insert("substrate_gamma_AB".to_string(), s_gamma);
    SubstrateState {
        tau_obs,
        label: Some(field.rule_at_origin.operating_point.label.clone()),
        axes,
        observables,
    }
}

fn apply_tangent_flow(
    canonical: &CanonicalState,
    field: &TangentFlowField,
    tau_obs: f64,
) -> SubstrateState {
    // Routes through `math::tangent_flow_substrate` so the libm-order
    // matches the bit-identity-tested core primitive.
    let (scaled_chit, scaled_gamma) = tangent_flow_substrate(
        canonical.chit,
        canonical.gamma_AB,
        field.scaling.delta_chit,
        field.scaling.delta_gamma,
        tau_obs,
        field.scaling.tau_obs_ref,
    );
    let mut axes = field.rule_at_origin.operating_point.axes.clone();
    axes.insert("tau_obs".to_string(), Value::from(tau_obs));
    let mut observables = BTreeMap::new();
    observables.insert("substrate_chit".to_string(), scaled_chit);
    observables.insert("substrate_gamma_AB".to_string(), scaled_gamma);
    SubstrateState {
        tau_obs,
        label: Some(field.rule_at_origin.operating_point.label.clone()),
        axes,
        observables,
    }
}

// Compile-time check that math.rs activation / layer types are the ones
// embedded in LearnedField (catches accidental re-typing during port).
const _: fn() = || {
    fn assert_activation(_: Activation) {}
    fn assert_layer(_: &MlpLayer) {}
    assert_activation(Activation::Tanh);
    let _ = assert_layer as fn(&MlpLayer);
};

// ---------------------------------------------------------------------------
// Op 2: forward_sweep_invert (grid path only — gradient methods deferred)
// ---------------------------------------------------------------------------

/// Result of one `forward_sweep_invert_grid` call.
#[derive(Debug, Clone, PartialEq)]
pub struct GridInversionResult {
    pub best_state: CanonicalState,
    /// `sqrt(min_residual)` — same scale as Python's `best_residual`.
    pub best_residual: f64,
    /// Per-candidate squared residuals in the same order as
    /// `canonical_grid`. Always populated (the grid scan is the work).
    pub residuals: Vec<f64>,
    pub best_index: usize,
}

/// Default substrate-vs-substrate scoring: L2 over shared numeric keys
/// in `observables ∪ axes` (axes overrides observables on conflict).
///
/// Key iteration is sorted to make the float-sum order deterministic;
/// Python's `set & set` is hash-randomized so Python-Rust agreement on
/// this score is tolerance-bound rather than bit-exact. The
/// `LIBM_WIDE = 16 ULPs` budget in `bit_identity.rs` absorbs the
/// reduction-order difference for the typical 2–4 shared keys per call.
pub fn default_substrate_score(predicted: &SubstrateState, target: &SubstrateState) -> f64 {
    let p = numeric_merged(predicted);
    let t = numeric_merged(target);
    let mut score = 0.0_f64;
    for (k, pv) in &p {
        if let Some(tv) = t.get(k) {
            let d = pv - tv;
            score += d * d;
        }
    }
    score
}

fn numeric_merged(s: &SubstrateState) -> BTreeMap<&str, f64> {
    // BTreeMap iterates in key order → deterministic.
    let mut out: BTreeMap<&str, f64> = BTreeMap::new();
    for (k, v) in &s.observables {
        out.insert(k.as_str(), *v);
    }
    for (k, v) in &s.axes {
        // Mirror Python's `isinstance(v, (int, float)) and not bool`:
        // `Value::as_f64` returns `None` for `Bool` / `String` / `Null` /
        // `Array` / `Object`, and `Some` for both integer and float
        // numbers.
        if let Some(n) = v.as_f64() {
            out.insert(k.as_str(), n);
        }
    }
    out
}

/// Substrate observation → canonical state at `tau_obs` via grid search.
///
/// This is the grid path only — Python's `method = "grid"`. The closed-
/// form / L-BFGS dispatch (Python's `method = "auto" / "gradient"`)
/// lands in session 5.
///
/// `canonical_grid` is `[[chit, gamma_AB], ...]`. `score_fn` and
/// `forward_map` are Python's `score_fn` / `forward_map` kwargs;
/// passing `None` for `forward_map` builds the default closure from
/// `field` via `apply_translation`. Passing `None` for `score_fn`
/// uses `default_substrate_score`.
pub fn forward_sweep_invert_grid(
    target: &SubstrateState,
    field: &TranslationField,
    tau_obs: f64,
    canonical_grid: &[[f64; 2]],
    score_fn: Option<&dyn Fn(&SubstrateState, &SubstrateState) -> f64>,
    forward_map: Option<&dyn Fn(&CanonicalState, f64) -> SubstrateState>,
) -> Result<GridInversionResult, OperationError> {
    if canonical_grid.is_empty() {
        return Err(OperationError::EmptyCanonicalGrid);
    }
    let default_score = default_substrate_score;
    let score: &dyn Fn(&SubstrateState, &SubstrateState) -> f64 =
        score_fn.unwrap_or(&default_score);

    // Default forward-map dispatch is inlined. For lookup-table fields we
    // pre-build the index once and reuse it across all candidates — the
    // same hot-loop optimization the Python `_grid_forward_map` closure
    // captures.
    let index_storage = if forward_map.is_none() {
        match field {
            TranslationField::LookupTable(lt) => Some(TranslationFieldIndex::new(lt)),
            _ => None,
        }
    } else {
        None
    };

    let n = canonical_grid.len();
    let mut residuals = Vec::with_capacity(n);
    for row in canonical_grid {
        let candidate = CanonicalState {
            chit: row[0],
            gamma_AB: row[1],
            k_frust: false,
        };
        let predicted = if let Some(fm) = forward_map {
            fm(&candidate, tau_obs)
        } else if let Some(idx) = index_storage.as_ref() {
            apply_translation_indexed(&candidate, idx, tau_obs, DEFAULT_DOMAIN_DISTANCE_THRESHOLD, 1.0)?
        } else {
            apply_translation(&candidate, field, tau_obs, DEFAULT_DOMAIN_DISTANCE_THRESHOLD, 1.0)?
        };
        residuals.push(score(&predicted, target));
    }

    // First-min — matches `np.argmin` for tied values.
    let mut best_index = 0usize;
    let mut best = residuals[0];
    for (i, v) in residuals.iter().enumerate().skip(1) {
        if *v < best {
            best = *v;
            best_index = i;
        }
    }
    let best_state = CanonicalState {
        chit: canonical_grid[best_index][0],
        gamma_AB: canonical_grid[best_index][1],
        k_frust: false,
    };
    let best_residual = residuals[best_index].sqrt();
    Ok(GridInversionResult {
        best_state,
        best_residual,
        residuals,
        best_index,
    })
}

// ---------------------------------------------------------------------------
// Op 2 (continued): forward_sweep_invert — method dispatcher (session 5)
// ---------------------------------------------------------------------------

/// Inversion strategy. Mirror of Python's `method` string kwarg on
/// `forward_sweep_invert`.
///
///  * `Auto`: per-field-shape choice — closed-form for `TangentFlow`,
///    L-BFGS-equivalent for `Learned`, grid for `LookupTable`. This is
///    Python's default.
///  * `Grid`: brute-force grid scan on any field shape (byte-identical
///    to v0–v4 behavior).
///  * `Gradient`: closed-form/L-BFGS for differentiable shapes; returns
///    `OperationError::GradientOnLookupTable` for lookup_table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    Auto,
    Grid,
    Gradient,
}

/// Result of `forward_sweep_invert`. Slimmer than `GridInversionResult`
/// because the gradient paths don't materialize per-candidate residuals
/// unless `return_residuals=true` is requested.
#[derive(Debug, Clone, PartialEq)]
pub struct InversionResult {
    pub best_state: CanonicalState,
    /// `sqrt(residual_squared)` at `best_state` — same scale as Python's
    /// `best_residual`. For the closed-form path this is evaluated via
    /// the field's forward map at the recovered point.
    pub best_residual: f64,
    /// Per-candidate squared residuals when the grid was scanned (Grid
    /// method, LookupTable under Auto, LearnedBfgs warm-start, or
    /// `return_residuals=true`). `None` when the closed-form path skipped
    /// the grid entirely.
    pub residuals: Option<Vec<f64>>,
    /// Grid index of the best candidate when `residuals.is_some()`.
    pub best_index: Option<usize>,
}

/// Dispatching variant of `forward_sweep_invert_grid` — Python's
/// `mpa_scale_solver.operations.forward_sweep_invert(..., method=...)`.
///
/// Routing follows Python:
///   * `forward_map` override forces grid (the override is opaque to the
///     gradient driver).
///   * `Method::Auto` → closed-form for TangentFlow, L-BFGS-equivalent
///     for Learned, grid for LookupTable.
///   * `Method::Grid` → grid scan on any field shape.
///   * `Method::Gradient` → closed-form/L-BFGS for differentiable shapes;
///     errors for LookupTable.
///
/// `return_residuals=true` always evaluates the grid (the field IS the
/// grid scan), independent of `method` — matches Python's
/// `return_residual_field=True`.
pub fn forward_sweep_invert(
    target: &SubstrateState,
    field: &TranslationField,
    tau_obs: f64,
    canonical_grid: &[[f64; 2]],
    score_fn: Option<&dyn Fn(&SubstrateState, &SubstrateState) -> f64>,
    forward_map: Option<&dyn Fn(&CanonicalState, f64) -> SubstrateState>,
    method: Method,
    return_residuals: bool,
) -> Result<InversionResult, OperationError> {
    if canonical_grid.is_empty() {
        return Err(OperationError::EmptyCanonicalGrid);
    }

    let effective = resolve_method(method, field, forward_map.is_some())?;

    let need_grid = matches!(effective, EffectiveMethod::Grid | EffectiveMethod::LearnedBfgs)
        || return_residuals;

    let grid = if need_grid {
        Some(forward_sweep_invert_grid(
            target, field, tau_obs, canonical_grid, score_fn, forward_map,
        )?)
    } else {
        None
    };

    match effective {
        EffectiveMethod::Grid => {
            let g = grid.expect("grid populated when effective method is Grid");
            Ok(InversionResult {
                best_state: g.best_state,
                best_residual: g.best_residual,
                residuals: Some(g.residuals),
                best_index: Some(g.best_index),
            })
        }
        EffectiveMethod::TangentFlowClosedForm => {
            let tf = match field {
                TranslationField::TangentFlow(t) => t,
                _ => unreachable!("resolve_method guarantees TangentFlow here"),
            };
            let best_state = invert_tangent_flow_closed_form(target, tf, tau_obs);
            let best_residual =
                residual_at(&best_state, field, tau_obs, target, score_fn, forward_map)?;
            Ok(InversionResult {
                best_state,
                best_residual,
                residuals: grid.map(|g| g.residuals),
                best_index: None,
            })
        }
        EffectiveMethod::LearnedBfgs => {
            let lf = match field {
                TranslationField::Learned(l) => l,
                _ => unreachable!("resolve_method guarantees Learned here"),
            };
            let g = grid
                .as_ref()
                .expect("grid populated when effective method is LearnedBfgs");
            let x0 = [g.best_state.chit, g.best_state.gamma_AB];
            let best_state = invert_learned_bfgs(target, lf, tau_obs, x0);
            let best_residual =
                residual_at(&best_state, field, tau_obs, target, score_fn, forward_map)?;
            Ok(InversionResult {
                best_state,
                best_residual,
                residuals: if return_residuals {
                    Some(g.residuals.clone())
                } else {
                    None
                },
                best_index: None,
            })
        }
    }
}

enum EffectiveMethod {
    Grid,
    TangentFlowClosedForm,
    LearnedBfgs,
}

fn resolve_method(
    method: Method,
    field: &TranslationField,
    forward_map_supplied: bool,
) -> Result<EffectiveMethod, OperationError> {
    if forward_map_supplied {
        return Ok(EffectiveMethod::Grid);
    }
    Ok(match (method, field) {
        (Method::Grid, _) => EffectiveMethod::Grid,
        (Method::Auto, TranslationField::TangentFlow(_)) => EffectiveMethod::TangentFlowClosedForm,
        (Method::Auto, TranslationField::Learned(_)) => EffectiveMethod::LearnedBfgs,
        (Method::Auto, TranslationField::LookupTable(_)) => EffectiveMethod::Grid,
        (Method::Gradient, TranslationField::TangentFlow(_)) => {
            EffectiveMethod::TangentFlowClosedForm
        }
        (Method::Gradient, TranslationField::Learned(_)) => EffectiveMethod::LearnedBfgs,
        (Method::Gradient, TranslationField::LookupTable(_)) => {
            return Err(OperationError::GradientOnLookupTable);
        }
    })
}

/// Exact closed-form inverse of the tangent-flow forward map. Mirror of
/// Python's `_invert_tangent_flow_closed_form`. Routes through
/// `math::tangent_flow_canonical_inverse` (session-1 bit-identity tested).
///
/// Falls back to a per-axis identity recovery when the substrate target
/// is missing the `substrate_chit` / `substrate_gamma_AB` keys — same
/// behavior as Python (v0 score collapses to 0 across all candidates in
/// that absent-keys case, so any seed works; pick the obvious one).
fn invert_tangent_flow_closed_form(
    target: &SubstrateState,
    field: &TangentFlowField,
    tau_obs: f64,
) -> CanonicalState {
    let s_chit = target.observables.get("substrate_chit").copied();
    let s_gamma = target.observables.get("substrate_gamma_AB").copied();
    match (s_chit, s_gamma) {
        (Some(s_chit), Some(s_gamma)) => {
            let (c, g) = tangent_flow_canonical_inverse(
                s_chit,
                s_gamma,
                field.scaling.delta_chit,
                field.scaling.delta_gamma,
                tau_obs,
                field.scaling.tau_obs_ref,
            );
            CanonicalState {
                chit: c,
                gamma_AB: g,
                k_frust: false,
            }
        }
        _ => {
            let chit_label = target
                .axes
                .get("chit_label")
                .and_then(Value::as_f64)
                .unwrap_or(0.0);
            CanonicalState {
                chit: target
                    .observables
                    .get("substrate_chit")
                    .copied()
                    .unwrap_or(chit_label),
                gamma_AB: target
                    .observables
                    .get("substrate_gamma_AB")
                    .copied()
                    .unwrap_or(0.0),
                k_frust: false,
            }
        }
    }
}

/// L-BFGS-equivalent inversion for a `LearnedField`. Mirror of Python's
/// `_invert_learned_bfgs`. The Python uses `scipy.optimize.minimize(
/// method="L-BFGS-B")` with `jax.grad`-provided analytical gradients;
/// the Rust port substitutes a hand-rolled 2D damped-Newton solver
/// (`optim::minimize_smooth_2d`) with numerical finite-difference
/// gradient + Hessian. Justification in `optim.rs` module docs; BLOCK_IN
/// §v6 session-5 carves out non-byte-identity vs scipy here.
///
/// Cost is the squared substrate residual:
///     ||learned_field_substrate(c) - target||^2
/// Warm-started from the grid argmin to avoid local minima.
///
/// If the target is missing `substrate_chit` / `substrate_gamma_AB`
/// there's nothing to optimize against — return the warm-start seed.
fn invert_learned_bfgs(
    target: &SubstrateState,
    field: &LearnedField,
    tau_obs: f64,
    x0: [f64; 2],
) -> CanonicalState {
    let s_chit = target.observables.get("substrate_chit").copied();
    let s_gamma = target.observables.get("substrate_gamma_AB").copied();
    let (target_chit, target_gamma) = match (s_chit, s_gamma) {
        (Some(c), Some(g)) => (c, g),
        _ => {
            return CanonicalState {
                chit: x0[0],
                gamma_AB: x0[1],
                k_frust: false,
            };
        }
    };

    let weights = field.weights.as_slice();
    let activation = field.activation;
    let tau_ref = field.tau_obs_ref;

    let cost = |chit: f64, gamma: f64| -> f64 {
        let (s_chit, s_gamma) =
            learned_field_substrate(chit, gamma, tau_obs, tau_ref, weights, activation);
        let d_chit = s_chit - target_chit;
        let d_gamma = s_gamma - target_gamma;
        d_chit * d_chit + d_gamma * d_gamma
    };

    let x_opt = minimize_smooth_2d(cost, x0);
    CanonicalState {
        chit: x_opt[0],
        gamma_AB: x_opt[1],
        k_frust: false,
    }
}

/// `sqrt(score)` at a recovered point — same scale as Python's
/// `_residual_at`. Used by the gradient paths so the returned
/// `best_residual` matches the v0/v1/v2/v3/v4 convention.
fn residual_at(
    state: &CanonicalState,
    field: &TranslationField,
    tau_obs: f64,
    target: &SubstrateState,
    score_fn: Option<&dyn Fn(&SubstrateState, &SubstrateState) -> f64>,
    forward_map: Option<&dyn Fn(&CanonicalState, f64) -> SubstrateState>,
) -> Result<f64, OperationError> {
    let predicted = if let Some(fm) = forward_map {
        fm(state, tau_obs)
    } else {
        apply_translation(
            state,
            field,
            tau_obs,
            DEFAULT_DOMAIN_DISTANCE_THRESHOLD,
            1.0,
        )?
    };
    let default_score = default_substrate_score;
    let score: &dyn Fn(&SubstrateState, &SubstrateState) -> f64 =
        score_fn.unwrap_or(&default_score);
    Ok(score(&predicted, target).sqrt())
}

// ---------------------------------------------------------------------------
// Op 3: tau_obs_sweep (per-frame fan-out)
// ---------------------------------------------------------------------------

/// Walk the RG-flow trajectory across `tau_obs`. Either a single
/// substrate observation broadcast across frames, or one per-frame
/// observation list.
///
/// Like `forward_sweep_invert_grid` this is the grid-path-only version
/// for session 4; gradient dispatch lands in session 5.
pub enum SweepTargets<'a> {
    Broadcast(&'a SubstrateState),
    PerFrame(&'a [SubstrateState]),
}

pub fn tau_obs_sweep_grid(
    targets: SweepTargets<'_>,
    field: &TranslationField,
    tau_obs_grid: &[f64],
    canonical_search_grid: &[[f64; 2]],
    score_fn: Option<&dyn Fn(&SubstrateState, &SubstrateState) -> f64>,
    forward_map: Option<&dyn Fn(&CanonicalState, f64) -> SubstrateState>,
) -> Result<Vec<CanonicalState>, OperationError> {
    let n_frames = tau_obs_grid.len();
    let resolved: Vec<&SubstrateState> = match targets {
        SweepTargets::Broadcast(s) => vec![s; n_frames],
        SweepTargets::PerFrame(list) => {
            if list.len() != n_frames {
                return Err(OperationError::TargetGridLengthMismatch {
                    targets: list.len(),
                    frames: n_frames,
                });
            }
            list.iter().collect()
        }
    };

    let mut trajectory = Vec::with_capacity(n_frames);
    for (i, &tau) in tau_obs_grid.iter().enumerate() {
        let result = forward_sweep_invert_grid(
            resolved[i],
            field,
            tau,
            canonical_search_grid,
            score_fn,
            forward_map,
        )?;
        trajectory.push(result.best_state);
    }
    Ok(trajectory)
}

// ---------------------------------------------------------------------------
// Op 4: regime_at + regime_display_band
// ---------------------------------------------------------------------------

/// Five-bucket vertex regime at this `tau_obs` frame. `tau_obs` is
/// accepted for traceability but ignored at v0 (RFC-S Appendix B item 4
/// territory for any future tau-conditional classifier).
pub fn regime_at(canonical: &CanonicalState, _tau_obs: f64) -> RegimeReading {
    RegimeReading {
        regime: vertex_regime(canonical.chit),
        k_frust: canonical.k_frust,
    }
}

/// Display-only collapse from the five-bucket regime to the three-bucket
/// display band.
pub fn regime_display_band(regime: RegimeLabel) -> DisplayBand {
    match regime {
        RegimeLabel::DeepC | RegimeLabel::CNearS => DisplayBand::C,
        RegimeLabel::SCritical => DisplayBand::S,
        RegimeLabel::RNearS | RegimeLabel::DeepR => DisplayBand::R,
    }
}

// ---------------------------------------------------------------------------
// Op 5: gamut_classify
// ---------------------------------------------------------------------------

/// One axis-level diagnosis emitted by `gamut_classify` when an axis is
/// out of range.
#[derive(Debug, Clone, PartialEq)]
pub struct GamutDiagnosis {
    pub axis: String,
    pub value: f64,
    pub range: (f64, f64),
    /// Distance to the nearer bound.
    pub distance: f64,
}

/// Result of `gamut_classify`. `diagnoses` is empty iff `in_gamut`.
#[derive(Debug, Clone, PartialEq)]
pub struct GamutClassification {
    pub in_gamut: bool,
    pub diagnoses: Vec<GamutDiagnosis>,
}

fn diag_for(axis: &str, value: f64, rng: (f64, f64)) -> GamutDiagnosis {
    let distance = (value - rng.0).abs().min((value - rng.1).abs());
    GamutDiagnosis {
        axis: axis.to_string(),
        value,
        range: rng,
        distance,
    }
}

pub fn gamut_classify(
    canonical: &CanonicalState,
    tau_obs: f64,
    gamut: &GamutSpec,
) -> GamutClassification {
    let mut diagnoses = Vec::new();
    if !(gamut.chit_range.0 <= canonical.chit && canonical.chit <= gamut.chit_range.1) {
        diagnoses.push(diag_for("chit", canonical.chit, gamut.chit_range));
    }
    if !(gamut.gamma_AB_range.0 <= canonical.gamma_AB
        && canonical.gamma_AB <= gamut.gamma_AB_range.1)
    {
        diagnoses.push(diag_for("gamma_AB", canonical.gamma_AB, gamut.gamma_AB_range));
    }
    if let Some(rng) = gamut.tau_obs_range {
        if !(rng.0 <= tau_obs && tau_obs <= rng.1) {
            diagnoses.push(diag_for("tau_obs", tau_obs, rng));
        }
    }
    GamutClassification {
        in_gamut: diagnoses.is_empty(),
        diagnoses,
    }
}

// ---------------------------------------------------------------------------
// Op 6: intent_map + intent_compose (RFC-S §3 — session 6)
// ---------------------------------------------------------------------------
//
// Five intents map an out-of-gamut canonical state to in-gamut, each
// preserving a named invariant:
//
//   I1 regime-preserving   : 5-bucket regime ∧ sign(gamma_AB) ∧ k_frust
//                            (the strongest constraint; subsumes I3/I4/I5
//                            on a single state)
//   I2 drive-faithful      : no adjustment; out-of-gamut flagged as
//                            completeness sacrifice (RFC-S §3
//                            "out-of-gamut rejected, diagnostic-listed")
//   I3 capacity-preserving : capacity class (|chit| >= 0.7 deep vs shallow)
//                            ∧ k_frust
//   I4 persistence-preserv : sign(gamma_AB) (contraction-ordering proxy at
//                            the state level)
//   I5 signature-preserving: 5-bucket regime label (v0/v1 contract)
//
// Each handler returns (mapped, SacrificeRecord). I5's v0/v1 diagnostic
// keys (regime_preserved / original_regime / mapped_regime) are
// preserved verbatim via `IntentDiagnostics::I5`; the v2.3 uniform keys
// (preserved_invariant, invariant_preserved) live on the outer record /
// `preserved_invariant()` derived method per the BLOCK_IN §v6 session-6
// sketch.

/// Map an out-of-gamut canonical state to in-gamut per the chosen intent.
/// Mirror of Python's `intent_map`.
///
/// Per RFC-S §3: scale uniformly along the gamut to fit, preserving the
/// named invariant. The state-level invariant for each intent is
/// documented in the module header above; sacrifice records carry the
/// uniform `invariant_preserved` field plus intent-specific diagnostics.
///
/// `tau_obs` is accepted for parity with Python but not consumed by any
/// of the five handlers — state-level intent mapping is a function of
/// `(state, gamut)` only at this layer.
pub fn intent_map(
    state: &CanonicalState,
    tau_obs: f64,
    gamut: &GamutSpec,
    intent: IntentId,
) -> (CanonicalState, SacrificeRecord) {
    match intent {
        IntentId::I1 => intent_i1(state, tau_obs, gamut),
        IntentId::I2 => intent_i2(state, tau_obs, gamut),
        IntentId::I3 => intent_i3(state, tau_obs, gamut),
        IntentId::I4 => intent_i4(state, tau_obs, gamut),
        IntentId::I5 => intent_i5(state, tau_obs, gamut),
    }
}

/// Apply intents sequentially per RFC-S §3 composition algebra. Mirror
/// of Python's `intent_compose`.
///
/// Per §3: "Two adjacent intents compose iff their preserved-invariant
/// sets union without conflict. I2 (drive-faithful) does not compose
/// with adjusting intents." This function enforces the I2 rule by
/// rejecting any composition containing I2 alongside other intents.
/// Beyond that, the union-without-conflict check is evidential: it
/// surfaces in each sacrifice's `invariant_preserved` flag — if a later
/// intent could not preserve its invariant on the output of an earlier
/// intent, the conflict is observable in the sacrifice trace.
///
/// Empty `intents` returns `OperationError::IntentComposeEmpty`.
pub fn intent_compose(
    state: &CanonicalState,
    tau_obs: f64,
    gamut: &GamutSpec,
    intents: &[IntentId],
) -> Result<(CanonicalState, Vec<SacrificeRecord>), OperationError> {
    if intents.is_empty() {
        return Err(OperationError::IntentComposeEmpty);
    }
    if intents.len() > 1 && intents.iter().any(|&i| i == IntentId::I2) {
        return Err(OperationError::I2InComposition);
    }
    let mut sacrifices = Vec::with_capacity(intents.len());
    let mut current = state.clone();
    for &iid in intents {
        let (next, sac) = intent_map(&current, tau_obs, gamut, iid);
        current = next;
        sacrifices.push(sac);
    }
    Ok((current, sacrifices))
}

// ---- intent helpers --------------------------------------------------------

fn sign_i(x: f64) -> i32 {
    if x > 0.0 {
        1
    } else if x < 0.0 {
        -1
    } else {
        0
    }
}

fn capacity_class(chit: f64) -> CapacityClass {
    if chit.abs() >= 0.7 {
        CapacityClass::Deep
    } else {
        CapacityClass::Shallow
    }
}

fn clamp_to_gamut(state: &CanonicalState, gamut: &GamutSpec) -> CanonicalState {
    CanonicalState {
        chit: state.chit.clamp(gamut.chit_range.0, gamut.chit_range.1),
        gamma_AB: state
            .gamma_AB
            .clamp(gamut.gamma_AB_range.0, gamut.gamma_AB_range.1),
        k_frust: state.k_frust,
    }
}

/// 5-bucket regime intervals on chit (matches `gfdr_model::vertex_regime`).
/// Open on the inside; the boundaries belong to the deeper bucket per
/// `vertex_regime`'s `chit >= 0.7` form (inclusive of 0.7 / 0.2).
fn regime_chit_interval(regime: RegimeLabel) -> (f64, f64) {
    match regime {
        RegimeLabel::DeepC => (0.7, f64::INFINITY),
        RegimeLabel::CNearS => (0.2, 0.7),
        RegimeLabel::SCritical => (-0.2, 0.2),
        RegimeLabel::RNearS => (-0.7, -0.2),
        RegimeLabel::DeepR => (f64::NEG_INFINITY, -0.7),
    }
}

/// Nearest in-gamut chit preserving `regime`, or `None` if unreachable.
/// The regime intervals are half-open on the deep side per
/// `vertex_regime`'s `chit >= threshold` form; we treat each interval
/// as closed for clipping purposes (vertex_regime's boundary inclusion
/// makes the endpoint a valid representative of the regime).
fn nearest_in_gamut_chit_for_regime(
    orig_chit: f64,
    regime: RegimeLabel,
    chit_range: (f64, f64),
) -> Option<f64> {
    let (lo_r, hi_r) = regime_chit_interval(regime);
    let (lo_g, hi_g) = chit_range;
    let lo = lo_r.max(lo_g);
    let hi = hi_r.min(hi_g);
    if lo > hi {
        return None;
    }
    if lo == hi {
        return Some(lo);
    }
    Some(orig_chit.clamp(lo, hi))
}

/// Clamp `value` to `rng` preserving sign. Returns
/// `(clamped, sign_preserved)`. Mirror of Python's
/// `_sign_preserving_clamp`.
///
/// `orig_sign == 0` returns the naive clamp (zero has no sign to
/// preserve). Otherwise prefers the nearest in-range value with the
/// same sign; falls back to the naive clamp with `sign_preserved=false`
/// when the gamut excludes the sign entirely.
fn sign_preserving_clamp(value: f64, orig_sign: i32, rng: (f64, f64)) -> (f64, bool) {
    let (lo, hi) = rng;
    if orig_sign == 0 {
        return (value.clamp(lo, hi), true);
    }
    if orig_sign > 0 {
        if hi <= 0.0 {
            return (value.clamp(lo, hi), false);
        }
        let sub_lo = lo.max(0.0);
        return (value.clamp(sub_lo, hi), true);
    }
    // orig_sign < 0
    if lo >= 0.0 {
        return (value.clamp(lo, hi), false);
    }
    let sub_hi = hi.min(0.0);
    (value.clamp(lo, sub_hi), true)
}

// ---- the five handlers -----------------------------------------------------

/// I1 regime-preserving: 5-bucket regime ∧ sign(gamma_AB) ∧ k_frust.
fn intent_i1(
    state: &CanonicalState,
    _tau_obs: f64,
    gamut: &GamutSpec,
) -> (CanonicalState, SacrificeRecord) {
    let orig_regime = vertex_regime(state.chit);
    let orig_sign = sign_i(state.gamma_AB);

    let target_chit = nearest_in_gamut_chit_for_regime(state.chit, orig_regime, gamut.chit_range);
    let regime_preserved = target_chit.is_some();
    let chit_out = target_chit
        .unwrap_or_else(|| state.chit.clamp(gamut.chit_range.0, gamut.chit_range.1));

    let (gamma_out, sign_preserved) =
        sign_preserving_clamp(state.gamma_AB, orig_sign, gamut.gamma_AB_range);

    let mapped = CanonicalState {
        chit: chit_out,
        gamma_AB: gamma_out,
        k_frust: state.k_frust,
    };
    let invariant_preserved = regime_preserved && sign_preserved;
    let mapped_regime = vertex_regime(chit_out);
    let sac = SacrificeRecord {
        invariant_preserved,
        delta_chit: chit_out - state.chit,
        delta_gamma_AB: gamma_out - state.gamma_AB,
        diagnostics: IntentDiagnostics::I1 {
            regime_preserved,
            gamma_AB_sign_preserved: sign_preserved,
            k_frust_preserved: true,
            original_regime: orig_regime,
            mapped_regime,
            original_gamma_AB_sign: orig_sign,
            mapped_gamma_AB_sign: sign_i(gamma_out),
        },
    };
    (mapped, sac)
}

/// I2 drive-faithful: no adjustment; flag completeness if out-of-gamut.
/// Per RFC-S §3: "Completeness sacrificed (out-of-gamut rejected,
/// diagnostic-listed)." The mapped state equals the original.
fn intent_i2(
    state: &CanonicalState,
    _tau_obs: f64,
    gamut: &GamutSpec,
) -> (CanonicalState, SacrificeRecord) {
    let chit_oog = !(gamut.chit_range.0 <= state.chit && state.chit <= gamut.chit_range.1);
    let gamma_oog =
        !(gamut.gamma_AB_range.0 <= state.gamma_AB && state.gamma_AB <= gamut.gamma_AB_range.1);
    let in_gamut = !(chit_oog || gamma_oog);
    let mut out_of_gamut_axes = Vec::new();
    if chit_oog {
        out_of_gamut_axes.push("chit".to_string());
    }
    if gamma_oog {
        out_of_gamut_axes.push("gamma_AB".to_string());
    }
    let sac = SacrificeRecord {
        invariant_preserved: in_gamut,
        delta_chit: 0.0,
        delta_gamma_AB: 0.0,
        diagnostics: IntentDiagnostics::I2 {
            out_of_gamut_rejected: !in_gamut,
            out_of_gamut_axes,
        },
    };
    (state.clone(), sac)
}

/// I3 capacity-preserving: capacity class (|chit| >= 0.7 deep) ∧ k_frust.
fn intent_i3(
    state: &CanonicalState,
    _tau_obs: f64,
    gamut: &GamutSpec,
) -> (CanonicalState, SacrificeRecord) {
    let orig_capacity = capacity_class(state.chit);
    let mut clamped = clamp_to_gamut(state, gamut);
    let mut mapped_capacity = capacity_class(clamped.chit);

    // If naive clamp drops a deep state to shallow, try the in-gamut
    // endpoint on the same side that retains deep.
    if orig_capacity == CapacityClass::Deep && mapped_capacity == CapacityClass::Shallow {
        if state.chit >= 0.7 && gamut.chit_range.1 >= 0.7 {
            clamped = CanonicalState {
                chit: state.chit.min(gamut.chit_range.1),
                gamma_AB: clamped.gamma_AB,
                k_frust: state.k_frust,
            };
            mapped_capacity = capacity_class(clamped.chit);
        } else if state.chit <= -0.7 && gamut.chit_range.0 <= -0.7 {
            clamped = CanonicalState {
                chit: state.chit.max(gamut.chit_range.0),
                gamma_AB: clamped.gamma_AB,
                k_frust: state.k_frust,
            };
            mapped_capacity = capacity_class(clamped.chit);
        }
    }

    let capacity_preserved = orig_capacity == mapped_capacity;
    let sac = SacrificeRecord {
        invariant_preserved: capacity_preserved,
        delta_chit: clamped.chit - state.chit,
        delta_gamma_AB: clamped.gamma_AB - state.gamma_AB,
        diagnostics: IntentDiagnostics::I3 {
            capacity_class: orig_capacity,
            mapped_capacity_class: mapped_capacity,
            k_frust: state.k_frust,
            k_frust_preserved: true,
        },
    };
    (clamped, sac)
}

/// I4 persistence-preserving: sign(gamma_AB) (contraction-ordering proxy).
fn intent_i4(
    state: &CanonicalState,
    _tau_obs: f64,
    gamut: &GamutSpec,
) -> (CanonicalState, SacrificeRecord) {
    let clamped = clamp_to_gamut(state, gamut);
    let orig_sign = sign_i(state.gamma_AB);
    let (gamma_out, sign_preserved) =
        sign_preserving_clamp(state.gamma_AB, orig_sign, gamut.gamma_AB_range);
    let mapped = CanonicalState {
        chit: clamped.chit,
        gamma_AB: gamma_out,
        k_frust: state.k_frust,
    };
    let sac = SacrificeRecord {
        invariant_preserved: sign_preserved,
        delta_chit: mapped.chit - state.chit,
        delta_gamma_AB: mapped.gamma_AB - state.gamma_AB,
        diagnostics: IntentDiagnostics::I4 {
            original_gamma_AB_sign: orig_sign,
            mapped_gamma_AB_sign: sign_i(gamma_out),
        },
    };
    (mapped, sac)
}

/// I5 signature-preserving: 5-bucket regime label.
///
/// v0/v1 contract: naive clamp on both axes; report `regime_preserved`
/// based on the 5-bucket `vertex_regime` comparison. Per RFC-S §5
/// metric, I5 is universality-class agreement; the 5-bucket regime is
/// the universality-class label at the operational layer (each regime
/// carries its own FDR-signature exponents per cdv1).
fn intent_i5(
    state: &CanonicalState,
    _tau_obs: f64,
    gamut: &GamutSpec,
) -> (CanonicalState, SacrificeRecord) {
    let original_regime = vertex_regime(state.chit);
    let chit_out = state.chit.clamp(gamut.chit_range.0, gamut.chit_range.1);
    let gamma_out = state
        .gamma_AB
        .clamp(gamut.gamma_AB_range.0, gamut.gamma_AB_range.1);
    let mapped = CanonicalState {
        chit: chit_out,
        gamma_AB: gamma_out,
        k_frust: state.k_frust,
    };
    let mapped_regime = vertex_regime(chit_out);
    let regime_preserved = original_regime == mapped_regime;
    let sac = SacrificeRecord {
        invariant_preserved: regime_preserved,
        delta_chit: chit_out - state.chit,
        delta_gamma_AB: gamma_out - state.gamma_AB,
        diagnostics: IntentDiagnostics::I5 {
            regime_preserved,
            original_regime,
            mapped_regime,
        },
    };
    (mapped, sac)
}

// ===========================================================================
// Op 7: validate_driver_profile (RFC-S §5 round-trip + per-intent metrics)
// ===========================================================================

/// One row of a reference dataset passed to `validate_driver_profile`.
/// Mirror of Python's `dict[str, Any]` with the three documented keys.
/// `expected_substrate=None` means "auto-compute via `apply_translation`",
/// matching Python's `entry.get("expected_substrate")` path.
#[derive(Debug, Clone, PartialEq)]
pub struct ReferenceDatasetEntry {
    pub canonical_state: CanonicalState,
    pub tau_obs: f64,
    pub expected_substrate: Option<SubstrateState>,
}

/// RFC-S §5 round-trip validation with per-intent metrics. Mirror of
/// Python's `validate_driver_profile` (operations.py).
///
/// For each entry of `reference_dataset`:
///   1. Forward: predict the substrate observation via `apply_translation`.
///   2. Inverse: recover the canonical via `forward_sweep_invert` (grid).
///   3. Score: forward residual (vs `expected_substrate` if provided),
///      round-trip residual (L2 in canonical space), regime agreement,
///      per-intent cell metric per RFC-S §5.
///
/// The summary's `per_intent` block is the aggregate from
/// `validation::aggregate_per_intent_metrics`. v2.3 back-compat keys
/// (`forward_residuals`, `round_trip_residuals`, `regime_agreements`,
/// `forward_mean`, `round_trip_mean`, `regime_agreement_rate`) are
/// preserved verbatim.
///
/// `gamut` (optional) supplies the I4 survival check via
/// `gamut_classify`; without it, I4's `survival` defaults to `true` per
/// cell, matching Python.
pub fn validate_driver_profile(
    field: &TranslationField,
    reference_dataset: &[ReferenceDatasetEntry],
    canonical_search_grid: &[[f64; 2]],
    intent_id: IntentId,
    gamut: Option<&GamutSpec>,
) -> Result<DriverProfileSummary, OperationError> {
    let index = match field {
        TranslationField::LookupTable(lt) => Some(TranslationFieldIndex::new(lt)),
        _ => None,
    };

    let mut forward_residuals: Vec<f64> = Vec::with_capacity(reference_dataset.len());
    let mut round_trip_residuals: Vec<f64> = Vec::with_capacity(reference_dataset.len());
    let mut regime_agreements: Vec<bool> = Vec::with_capacity(reference_dataset.len());
    let mut per_cell_metrics: Vec<std::collections::BTreeMap<String, Value>> =
        Vec::with_capacity(reference_dataset.len());

    for entry in reference_dataset {
        let canonical = &entry.canonical_state;
        let tau_obs = entry.tau_obs;

        let predicted = if let Some(idx) = index.as_ref() {
            apply_translation_indexed(
                canonical,
                idx,
                tau_obs,
                DEFAULT_DOMAIN_DISTANCE_THRESHOLD,
                1.0,
            )?
        } else {
            apply_translation(canonical, field, tau_obs, DEFAULT_DOMAIN_DISTANCE_THRESHOLD, 1.0)?
        };

        let fwd_err = if let Some(expected) = entry.expected_substrate.as_ref() {
            default_substrate_score(&predicted, expected)
        } else {
            0.0
        };
        forward_residuals.push(fwd_err.sqrt());

        // Run the grid inverter (Python `forward_sweep_invert` defaults to
        // `method="auto"`, but `validate_driver_profile` in Python calls
        // it without a method kwarg, so the default "auto" applies — which
        // for LookupTable is grid, and for TangentFlow / Learned is
        // closed-form / L-BFGS. We mirror by using `Method::Auto`).
        let inv = forward_sweep_invert(
            &predicted,
            field,
            tau_obs,
            canonical_search_grid,
            None,
            None,
            Method::Auto,
            false,
        )?;
        // Python preserves k_frust from the truth canonical through the
        // recovered state so the I1/I3 metrics that read it are aligned.
        let recovered = CanonicalState {
            chit: inv.best_state.chit,
            gamma_AB: inv.best_state.gamma_AB,
            k_frust: canonical.k_frust,
        };

        let d_chit = recovered.chit - canonical.chit;
        let d_gamma = recovered.gamma_AB - canonical.gamma_AB;
        let rt_err = (d_chit * d_chit + d_gamma * d_gamma).sqrt();
        round_trip_residuals.push(rt_err);

        let orig_r = regime_at(canonical, tau_obs).regime;
        let rec_r = regime_at(&recovered, tau_obs).regime;
        regime_agreements.push(orig_r == rec_r);

        let in_gamut = gamut.map(|g| gamut_classify(&recovered, tau_obs, g).in_gamut);
        per_cell_metrics.push(per_intent_cell_metric(
            intent_id,
            canonical,
            &recovered,
            in_gamut,
        ));
    }

    let n = reference_dataset.len() as f64;
    let forward_mean = if forward_residuals.is_empty() {
        0.0
    } else {
        forward_residuals.iter().sum::<f64>() / n
    };
    let round_trip_mean = if round_trip_residuals.is_empty() {
        0.0
    } else {
        round_trip_residuals.iter().sum::<f64>() / n
    };
    let regime_agreement_rate = if regime_agreements.is_empty() {
        0.0
    } else {
        regime_agreements.iter().filter(|b| **b).count() as f64 / n
    };

    let per_intent = aggregate_per_intent_metrics(intent_id, &per_cell_metrics);

    Ok(DriverProfileSummary {
        intent: intent_id,
        forward_residuals,
        round_trip_residuals,
        regime_agreements,
        forward_mean,
        round_trip_mean,
        regime_agreement_rate,
        per_intent,
    })
}

// ===========================================================================
// Wrapped variants — handoff §A.2 / §C.5 / §C.6
// ===========================================================================
//
// Each `*_wrapped` calls the matching raw operation, then stamps a
// ValidationReport + Provenance onto an OperationOutput<T>. Sidecar
// dispatch (handoff §C.4) is opt-in via the `sidecar` parameter and is
// meaningful for `apply_translation_wrapped`, `forward_sweep_invert_wrapped`,
// and `tau_obs_sweep_wrapped`. The remaining wrapped variants have no
// sidecar dispatch — they only attach validation + provenance.
//
// Rust signatures are verbose because Rust has no keyword arguments;
// callers pass `None` / `&[]` / defaults explicitly. This matches the
// style established by the raw operations in this module.

/// Wrapped variant of `apply_translation` (handoff §A.2 / §C.5).
pub fn apply_translation_wrapped(
    canonical: &CanonicalState,
    field: &TranslationField,
    tau_obs: f64,
    domain_distance_threshold: f64,
    tau_obs_weight: f64,
    sidecar: Option<&InverseLookupSidecar>,
) -> Result<OperationOutput<SubstrateState>, OperationError> {
    let mut dispatch = DispatchPath::DirectCompute;
    let mut table_version: Option<String> = None;
    let substrate = if let Some(sc) = sidecar {
        table_version = Some(sc.version.clone());
        match lookup_forward(sc, canonical, tau_obs, DEFAULT_ROUNDING_DECIMALS) {
            Some(hit) => {
                dispatch = DispatchPath::TableHit;
                hit
            }
            None => {
                dispatch = DispatchPath::ComputeFallback;
                apply_translation(
                    canonical,
                    field,
                    tau_obs,
                    domain_distance_threshold,
                    tau_obs_weight,
                )?
            }
        }
    } else {
        apply_translation(
            canonical,
            field,
            tau_obs,
            domain_distance_threshold,
            tau_obs_weight,
        )?
    };
    let report = _validation::report_for_apply_translation(canonical, &substrate, &[]);
    let prov = make_provenance("apply_translation", dispatch, table_version, Vec::new());
    Ok(OperationOutput {
        value: substrate,
        validation: report,
        provenance: prov,
    })
}

/// Wrapped variant of `forward_sweep_invert`.
///
/// Sidecar dispatch is table-first: an inverse-table hit returns the
/// recorded canonical with `dispatch_path = TableHit`; on miss the
/// brute-force grid search runs with `dispatch_path = ComputeFallback`.
///
/// `compute_round_trip` controls whether the wrapped variant runs a
/// forward-then-back recovery for the validation report's
/// `round_trip_residual`. Default Python is `True`; turn off in tight
/// inner loops (e.g. `tau_obs_sweep_wrapped`'s per-frame dispatch).
///
/// `method` is forwarded to `forward_sweep_invert`.
pub fn forward_sweep_invert_wrapped(
    target_substrate: &SubstrateState,
    field: &TranslationField,
    tau_obs: f64,
    canonical_grid: &[[f64; 2]],
    score_fn: Option<&dyn Fn(&SubstrateState, &SubstrateState) -> f64>,
    forward_map: Option<&dyn Fn(&CanonicalState, f64) -> SubstrateState>,
    sidecar: Option<&InverseLookupSidecar>,
    compute_round_trip: bool,
    method: Method,
) -> Result<OperationOutput<CanonicalState>, OperationError> {
    let mut dispatch = DispatchPath::DirectCompute;
    let mut table_version: Option<String> = None;
    let recovered = if let Some(sc) = sidecar {
        table_version = Some(sc.version.clone());
        match lookup_inverse(sc, target_substrate, tau_obs, DEFAULT_ROUNDING_DECIMALS) {
            Some(hit) => {
                dispatch = DispatchPath::TableHit;
                hit
            }
            None => {
                dispatch = DispatchPath::ComputeFallback;
                forward_sweep_invert(
                    target_substrate,
                    field,
                    tau_obs,
                    canonical_grid,
                    score_fn,
                    forward_map,
                    method,
                    false,
                )?
                .best_state
            }
        }
    } else {
        forward_sweep_invert(
            target_substrate,
            field,
            tau_obs,
            canonical_grid,
            score_fn,
            forward_map,
            method,
            false,
        )?
        .best_state
    };

    let rt_residual: Option<f64> = if compute_round_trip {
        // Forward-then-back via the same translation field. Python's
        // version catches ValueError and substitutes +inf; the Rust port
        // promotes any `OperationError` to the same `+inf` sentinel so
        // the wrapped variant never propagates a round-trip failure
        // (the residual carries the diagnostic).
        match apply_translation(
            &recovered,
            field,
            tau_obs,
            DEFAULT_DOMAIN_DISTANCE_THRESHOLD,
            1.0,
        ) {
            Ok(forward_back) => Some(default_substrate_score(&forward_back, target_substrate).sqrt()),
            Err(_) => Some(f64::INFINITY),
        }
    } else {
        None
    };

    let report = _validation::report_for_forward_sweep_invert(
        target_substrate,
        &recovered,
        rt_residual,
        &[],
    );
    let prov = make_provenance("forward_sweep_invert", dispatch, table_version, Vec::new());
    Ok(OperationOutput {
        value: recovered,
        validation: report,
        provenance: prov,
    })
}

/// Wrapped variant of `tau_obs_sweep` — per-frame dispatch via
/// `forward_sweep_invert_wrapped` with `compute_round_trip=false`.
///
/// The aggregate `provenance.dispatch_path` is `TableHit` only when
/// every frame hit the table; otherwise `DirectCompute` and the
/// per-frame mix is summarized in `notes`.
pub fn tau_obs_sweep_wrapped(
    targets: SweepTargets<'_>,
    field: &TranslationField,
    tau_obs_grid: &[f64],
    canonical_search_grid: &[[f64; 2]],
    score_fn: Option<&dyn Fn(&SubstrateState, &SubstrateState) -> f64>,
    forward_map: Option<&dyn Fn(&CanonicalState, f64) -> SubstrateState>,
    sidecar: Option<&InverseLookupSidecar>,
) -> Result<OperationOutput<Vec<CanonicalState>>, OperationError> {
    let n_frames = tau_obs_grid.len();
    let resolved: Vec<&SubstrateState> = match targets {
        SweepTargets::Broadcast(s) => vec![s; n_frames],
        SweepTargets::PerFrame(list) => {
            if list.len() != n_frames {
                return Err(OperationError::TargetGridLengthMismatch {
                    targets: list.len(),
                    frames: n_frames,
                });
            }
            list.iter().collect()
        }
    };

    let mut trajectory: Vec<CanonicalState> = Vec::with_capacity(n_frames);
    let mut n_table = 0usize;
    let mut n_fallback = 0usize;
    let mut n_direct = 0usize;

    for (i, &tau) in tau_obs_grid.iter().enumerate() {
        let out = forward_sweep_invert_wrapped(
            resolved[i],
            field,
            tau,
            canonical_search_grid,
            score_fn,
            forward_map,
            sidecar,
            false,
            Method::Auto,
        )?;
        trajectory.push(out.value);
        match out.provenance.dispatch_path {
            DispatchPath::TableHit => n_table += 1,
            DispatchPath::ComputeFallback => n_fallback += 1,
            DispatchPath::DirectCompute => n_direct += 1,
        }
    }

    let aggregate = if n_table == n_frames {
        DispatchPath::TableHit
    } else {
        DispatchPath::DirectCompute
    };
    let notes = vec![format!(
        "frames: table_hit={n_table}, compute_fallback={n_fallback}, direct_compute={n_direct}"
    )];
    let report = _validation::report_for_tau_obs_sweep(&trajectory);
    let prov = make_provenance(
        "tau_obs_sweep",
        aggregate,
        sidecar.map(|s| s.version.clone()),
        notes,
    );
    Ok(OperationOutput {
        value: trajectory,
        validation: report,
        provenance: prov,
    })
}

/// Wrapped variant of `regime_at`.
pub fn regime_at_wrapped(
    canonical: &CanonicalState,
    tau_obs: f64,
) -> OperationOutput<RegimeReading> {
    let reading = regime_at(canonical, tau_obs);
    let report = _validation::report_for_regime_at(canonical);
    let prov = make_provenance("regime_at", DispatchPath::DirectCompute, None, Vec::new());
    OperationOutput {
        value: reading,
        validation: report,
        provenance: prov,
    }
}

/// Wrapped variant of `gamut_classify`.
pub fn gamut_classify_wrapped(
    canonical: &CanonicalState,
    tau_obs: f64,
    gamut: &GamutSpec,
) -> OperationOutput<GamutClassification> {
    let value = gamut_classify(canonical, tau_obs, gamut);
    let report = _validation::report_for_gamut_classify(canonical);
    let prov = make_provenance(
        "gamut_classify",
        DispatchPath::DirectCompute,
        None,
        Vec::new(),
    );
    OperationOutput {
        value,
        validation: report,
        provenance: prov,
    }
}

/// Wrapped variant of `intent_map`.
pub fn intent_map_wrapped(
    state: &CanonicalState,
    tau_obs: f64,
    gamut: &GamutSpec,
    intent: IntentId,
) -> OperationOutput<(CanonicalState, SacrificeRecord)> {
    let (mapped, sacrifice) = intent_map(state, tau_obs, gamut, intent);
    let report = _validation::report_for_intent_map(state, &mapped, &sacrifice);
    let prov = make_provenance("intent_map", DispatchPath::DirectCompute, None, Vec::new());
    OperationOutput {
        value: (mapped, sacrifice),
        validation: report,
        provenance: prov,
    }
}

/// Wrapped variant of `intent_compose` (RFC-S §3 composition).
///
/// Validation aggregates the per-intent `invariant_preserved` flags:
/// `k_frust_invariant` is True only when every intent in the chain
/// preserved its invariant. Per-intent failures are listed in notes.
///
/// Propagates `intent_compose`'s `Result` — empty intents and I2 in a
/// composition surface as `OperationError`.
pub fn intent_compose_wrapped(
    state: &CanonicalState,
    tau_obs: f64,
    gamut: &GamutSpec,
    intents: &[IntentId],
) -> Result<OperationOutput<(CanonicalState, Vec<SacrificeRecord>)>, OperationError> {
    let (mapped, sacrifices) = intent_compose(state, tau_obs, gamut, intents)?;
    let report = _validation::report_for_intent_compose(state, &mapped, &sacrifices);
    let intent_repr = format!(
        "intents=({})",
        intents
            .iter()
            .map(|i| match i {
                IntentId::I1 => "I1",
                IntentId::I2 => "I2",
                IntentId::I3 => "I3",
                IntentId::I4 => "I4",
                IntentId::I5 => "I5",
            })
            .collect::<Vec<_>>()
            .join(", "),
    );
    let prov = make_provenance(
        "intent_compose",
        DispatchPath::DirectCompute,
        None,
        vec![intent_repr],
    );
    Ok(OperationOutput {
        value: (mapped, sacrifices),
        validation: report,
        provenance: prov,
    })
}

/// Wrapped variant of `validate_driver_profile`.
pub fn validate_driver_profile_wrapped(
    field: &TranslationField,
    reference_dataset: &[ReferenceDatasetEntry],
    canonical_search_grid: &[[f64; 2]],
    intent_id: IntentId,
    gamut: Option<&GamutSpec>,
) -> Result<OperationOutput<DriverProfileSummary>, OperationError> {
    let summary = validate_driver_profile(
        field,
        reference_dataset,
        canonical_search_grid,
        intent_id,
        gamut,
    )?;
    let report = _validation::report_for_validate_driver_profile(&summary);
    let prov = make_provenance(
        "validate_driver_profile",
        DispatchPath::DirectCompute,
        None,
        Vec::new(),
    );
    Ok(OperationOutput {
        value: summary,
        validation: report,
        provenance: prov,
    })
}

// ---------------------------------------------------------------------------
// Session 8 — Bayesian inversion (Laplace approximation) — BLOCK_IN cut b
// ---------------------------------------------------------------------------
//
// Mirrors `mpa_scale_solver.operations.forward_sweep_invert_posterior` +
// `_wrapped` and the underlying `jax_ops.tangent_flow_posterior` /
// `lookup_table_posterior`. Single-mode Laplace: tangent-flow gets the
// closed-form fast path (MAP exact, residual at MAP zero, covariance =
// `noise_variance * inv(J^T J)`); lookup-table gets the softmax-weighted
// top-k discrete moment estimate (no Hessian on a step-function residual).
//
// Per BLOCK_IN §v6 session 8:
//   * `k == 1` lookup-table path returns a delta posterior with
//     noise-floor covariance (one-line literal port of Python).
//   * Top-k ranking uses a stable sort `(residual, index)` so ties break
//     by index — Python's `np.argsort` is stable, Rust's `sort_by` is
//     unstable so the tiebreak is explicit.
//   * `Posterior.modes` is populated only when the MAP point differs from
//     the posterior mean (lookup-table only); the comparison uses
//     `f64::to_bits` equality to match Python's tuple equality on floats.

/// Laplace-approximation posterior for tangent-flow inversion. Mirrors
/// `jax_ops.tangent_flow_posterior`.
///
/// Fast-path: MAP is the exact closed-form inverse, residual at MAP is
/// zero, Hessian at MAP is `(1/σ²) JᵀJ` where J is the forward-map
/// Jacobian (session-8 primitive `math::tangent_flow_forward_jacobian`),
/// and the posterior covariance is `σ² (JᵀJ)⁻¹`. Log evidence collapses
/// to the noise-prior-only normalizer since the residual term vanishes.
pub fn tangent_flow_posterior(
    target: &SubstrateState,
    field: &TangentFlowField,
    tau_obs: f64,
    noise_variance: f64,
    k_frust: bool,
) -> Posterior {
    let s_chit = target
        .observables
        .get("substrate_chit")
        .copied()
        .unwrap_or(0.0);
    let s_gamma = target
        .observables
        .get("substrate_gamma_AB")
        .copied()
        .unwrap_or(0.0);
    let (map_chit, map_gamma) = tangent_flow_canonical_inverse(
        s_chit,
        s_gamma,
        field.scaling.delta_chit,
        field.scaling.delta_gamma,
        tau_obs,
        field.scaling.tau_obs_ref,
    );
    let map = CanonicalState {
        chit: map_chit,
        gamma_AB: map_gamma,
        k_frust,
    };

    let jac =
        tangent_flow_forward_jacobian(field.scaling.delta_gamma, tau_obs, field.scaling.tau_obs_ref);
    let cov = laplace_covariance_from_jacobian(&jac[..], noise_variance)
        .unwrap_or([[noise_variance, 0.0], [0.0, noise_variance]]);

    // log p(y) at zero residual:
    //   -0.5 * dim_y * log(2π σ²)
    //   + 0.5 * dim_c * log(2π)
    //   - 0.5 * log det((1/σ²) JᵀJ)
    let dim_y = 2.0;
    let dim_c = 2.0;
    let jtj: [[f64; 2]; 2] = [
        [
            jac[0][0] * jac[0][0] + jac[1][0] * jac[1][0],
            jac[0][0] * jac[0][1] + jac[1][0] * jac[1][1],
        ],
        [
            jac[0][1] * jac[0][0] + jac[1][1] * jac[1][0],
            jac[0][1] * jac[0][1] + jac[1][1] * jac[1][1],
        ],
    ];
    let precision = [
        [jtj[0][0] / noise_variance, jtj[0][1] / noise_variance],
        [jtj[1][0] / noise_variance, jtj[1][1] / noise_variance],
    ];
    let (_sign, log_det_precision) = slogdet_2x2(&precision);
    let two_pi = 2.0 * std::f64::consts::PI;
    let log_evidence = -0.5 * dim_y * (two_pi * noise_variance).ln()
        + 0.5 * dim_c * two_pi.ln()
        - 0.5 * log_det_precision;

    Posterior {
        mean: map,
        covariance: cov,
        noise_variance,
        log_evidence: Some(log_evidence),
        modes: Vec::new(),
        notes: vec!["laplace_from_closed_form_jacobian".to_string()],
    }
}

/// Weighted-moment posterior for lookup-table inversion. Mirrors
/// `jax_ops.lookup_table_posterior`.
///
/// Discrete grids have no meaningful Hessian (the residual is a step
/// function over candidates), so the Laplace formula doesn't apply
/// directly. Instead we treat the residual field as an unnormalized
/// log-posterior `log p(c | y) ∝ -0.5 R(c) / σ²` and report the moments
/// of the resulting discrete distribution, concentrated on the `top_k`
/// lowest-residual candidates.
pub fn lookup_table_posterior(
    target: &SubstrateState,
    field: &LookupTableField,
    tau_obs: f64,
    canonical_grid: &[[f64; 2]],
    noise_variance: f64,
    k_frust: bool,
    score_fn: Option<&dyn Fn(&SubstrateState, &SubstrateState) -> f64>,
    top_k: usize,
) -> Result<Posterior, OperationError> {
    // forward_sweep_invert wants the enum; the clone is bounded by the
    // schema-fixed rule list and happens once per posterior call.
    let field_enum = TranslationField::LookupTable(field.clone());
    let inv = forward_sweep_invert(
        target,
        &field_enum,
        tau_obs,
        canonical_grid,
        score_fn,
        None,
        Method::Auto,
        true,
    )?;
    let residuals = inv
        .residuals
        .expect("Method::Auto + return_residuals=true on LookupTable populates residuals");

    let n = residuals.len();
    let k = top_k.clamp(1, n);

    let map_with_kfrust = CanonicalState {
        chit: inv.best_state.chit,
        gamma_AB: inv.best_state.gamma_AB,
        k_frust,
    };

    if k == 1 {
        // Degenerate single-candidate case. Python adds a noise-floor on
        // the diagonal so the posterior remains usable as a covariance.
        return Ok(Posterior {
            mean: map_with_kfrust,
            covariance: [[noise_variance, 0.0], [0.0, noise_variance]],
            noise_variance,
            log_evidence: None,
            modes: Vec::new(),
            notes: vec!["lookup_table_grid_top_k=1_delta_with_noise_floor".to_string()],
        });
    }

    // Stable top-k by (residual, index) — ties break by original index so
    // the softmax-weighted moments are deterministic across Rust runs.
    let mut indexed: Vec<(f64, usize)> = residuals.iter().copied().enumerate()
        .map(|(i, r)| (r, i))
        .collect();
    indexed.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
    let top: Vec<(f64, [f64; 2])> = indexed
        .iter()
        .take(k)
        .map(|&(r, i)| (r, canonical_grid[i]))
        .collect();

    // log-weights = -0.5 * R / σ²; shift by max for numerical stability.
    let nv_safe = noise_variance.max(1e-300);
    let log_w: Vec<f64> = top.iter().map(|&(r, _)| -0.5 * r / nv_safe).collect();
    let log_w_max = log_w.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let raw_w: Vec<f64> = log_w.iter().map(|&lw| (lw - log_w_max).exp()).collect();
    let raw_sum: f64 = raw_w.iter().copied().sum();
    let weights: Vec<f64> = raw_w.iter().map(|w| w / raw_sum).collect();

    let mean_chit: f64 = weights
        .iter()
        .zip(top.iter())
        .map(|(&w, &(_, p))| w * p[0])
        .sum();
    let mean_gamma: f64 = weights
        .iter()
        .zip(top.iter())
        .map(|(&w, &(_, p))| w * p[1])
        .sum();

    let cov_cc: f64 = weights
        .iter()
        .zip(top.iter())
        .map(|(&w, &(_, p))| w * (p[0] - mean_chit).powi(2))
        .sum();
    let cov_cg: f64 = weights
        .iter()
        .zip(top.iter())
        .map(|(&w, &(_, p))| w * (p[0] - mean_chit) * (p[1] - mean_gamma))
        .sum();
    let cov_gg: f64 = weights
        .iter()
        .zip(top.iter())
        .map(|(&w, &(_, p))| w * (p[1] - mean_gamma).powi(2))
        .sum();
    let cov = [[cov_cc, cov_cg], [cov_cg, cov_gg]];

    let posterior_mean = CanonicalState {
        chit: mean_chit,
        gamma_AB: mean_gamma,
        k_frust,
    };

    // Python: `if (map_chit, map_gamma) != (mean_chit, mean_gamma)` — tuple
    // equality is bit-equality on f64. Mirror exactly so the emit-condition
    // discipline matches across languages.
    let modes = if map_with_kfrust.chit.to_bits() != mean_chit.to_bits()
        || map_with_kfrust.gamma_AB.to_bits() != mean_gamma.to_bits()
    {
        vec![map_with_kfrust]
    } else {
        Vec::new()
    };

    Ok(Posterior {
        mean: posterior_mean,
        covariance: cov,
        noise_variance,
        log_evidence: None,
        modes,
        notes: vec![format!("lookup_table_weighted_moments_top_k={k}")],
    })
}

/// Bayesian inversion dispatcher — Python's
/// `mpa_scale_solver.operations.forward_sweep_invert_posterior`.
///
/// Dispatches on field shape:
///   * `TangentFlow` → closed-form fast path via `tangent_flow_posterior`.
///     `canonical_grid` is ignored.
///   * `LookupTable` → weighted-moment estimate via `lookup_table_posterior`.
///     Requires `canonical_grid`.
///   * `Learned` → no Laplace surface in v5 — returns
///     `OperationError::PosteriorUnsupportedFieldShape` (Python raises
///     `TypeError`).
pub fn forward_sweep_invert_posterior(
    target: &SubstrateState,
    field: &TranslationField,
    tau_obs: f64,
    canonical_grid: Option<&[[f64; 2]]>,
    noise_variance: f64,
    k_frust: bool,
    score_fn: Option<&dyn Fn(&SubstrateState, &SubstrateState) -> f64>,
    top_k: usize,
) -> Result<Posterior, OperationError> {
    match field {
        TranslationField::TangentFlow(tf) => Ok(tangent_flow_posterior(
            target,
            tf,
            tau_obs,
            noise_variance,
            k_frust,
        )),
        TranslationField::LookupTable(lt) => {
            let grid = canonical_grid.ok_or(OperationError::PosteriorRequiresCanonicalGrid)?;
            lookup_table_posterior(
                target,
                lt,
                tau_obs,
                grid,
                noise_variance,
                k_frust,
                score_fn,
                top_k,
            )
        }
        TranslationField::Learned(_) => Err(OperationError::PosteriorUnsupportedFieldShape),
    }
}

/// Wrapped variant of `forward_sweep_invert_posterior`. Mirrors Python's
/// `forward_sweep_invert_posterior_wrapped`. Validation reuses the
/// `forward_sweep_invert` report shape on the MAP point — the canonical
/// at MAP rides through the same asymptotic-closure gate as every other
/// wrapped operation's canonical-state output. `round_trip_residual` is
/// `None` (no forward-then-back is meaningful when the value is a
/// posterior distribution rather than a point estimate).
pub fn forward_sweep_invert_posterior_wrapped(
    target: &SubstrateState,
    field: &TranslationField,
    tau_obs: f64,
    canonical_grid: Option<&[[f64; 2]]>,
    noise_variance: f64,
    k_frust: bool,
    score_fn: Option<&dyn Fn(&SubstrateState, &SubstrateState) -> f64>,
    top_k: usize,
) -> Result<OperationOutput<Posterior>, OperationError> {
    let posterior = forward_sweep_invert_posterior(
        target,
        field,
        tau_obs,
        canonical_grid,
        noise_variance,
        k_frust,
        score_fn,
        top_k,
    )?;
    let report =
        _validation::report_for_forward_sweep_invert(target, &posterior.mean, None, &[]);
    let prov = make_provenance(
        "forward_sweep_invert_posterior",
        DispatchPath::DirectCompute,
        None,
        Vec::new(),
    );
    Ok(OperationOutput {
        value: posterior,
        validation: report,
        provenance: prov,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        CanonicalPoint, Direction, Gt, OperatingPoint, ScalingRule, TranslationRule,
    };

    fn rule(label: &str, chit: f64, gamma: f64, tau_obs: Option<f64>) -> TranslationRule {
        let mut axes = BTreeMap::new();
        if let Some(t) = tau_obs {
            axes.insert("tau_obs".to_string(), Value::from(t));
        }
        TranslationRule {
            operating_point: OperatingPoint {
                label: label.to_string(),
                gt: Gt::S,
                axes,
            },
            xdot_choice: "default".to_string(),
            canonical: CanonicalPoint {
                chit,
                gamma_AB: gamma,
                k_frust: false,
                method: "test".to_string(),
                extras: BTreeMap::new(),
            },
        }
    }

    fn tangent_flow_field(delta_chit: f64, delta_gamma: f64) -> TangentFlowField {
        TangentFlowField {
            direction: Direction::Forward,
            rule_at_origin: rule("origin", 0.0, 0.0, None),
            scaling: ScalingRule {
                tau_obs_ref: 1.0,
                delta_chit,
                delta_gamma,
                refinement: None,
            },
            description: None,
        }
    }

    #[test]
    fn regime_at_dispatches_via_vertex_regime() {
        let state = CanonicalState {
            chit: 0.8,
            gamma_AB: 0.0,
            k_frust: true,
        };
        let r = regime_at(&state, 1.0);
        assert_eq!(r.regime, RegimeLabel::DeepC);
        assert!(r.k_frust);
    }

    #[test]
    fn regime_display_band_collapses_to_three() {
        assert_eq!(regime_display_band(RegimeLabel::DeepC), DisplayBand::C);
        assert_eq!(regime_display_band(RegimeLabel::CNearS), DisplayBand::C);
        assert_eq!(regime_display_band(RegimeLabel::SCritical), DisplayBand::S);
        assert_eq!(regime_display_band(RegimeLabel::RNearS), DisplayBand::R);
        assert_eq!(regime_display_band(RegimeLabel::DeepR), DisplayBand::R);
    }

    #[test]
    fn apply_translation_tangent_flow_at_ref_is_identity() {
        let field = TranslationField::TangentFlow(tangent_flow_field(0.3, 0.5));
        let state = CanonicalState {
            chit: 1.5,
            gamma_AB: 2.5,
            k_frust: false,
        };
        // tau_obs == tau_obs_ref → ratio = 1, no shift.
        let s = apply_translation(&state, &field, 1.0, DEFAULT_DOMAIN_DISTANCE_THRESHOLD, 1.0)
            .unwrap();
        assert_eq!(s.observables["substrate_chit"], 1.5);
        assert_eq!(s.observables["substrate_gamma_AB"], 2.5);
    }

    #[test]
    fn apply_translation_lookup_empty_errors() {
        let field = TranslationField::LookupTable(LookupTableField {
            direction: Direction::Forward,
            rule: vec![],
            description: None,
        });
        let state = CanonicalState {
            chit: 0.0,
            gamma_AB: 0.0,
            k_frust: false,
        };
        assert_eq!(
            apply_translation(&state, &field, 1.0, DEFAULT_DOMAIN_DISTANCE_THRESHOLD, 1.0),
            Err(OperationError::EmptyTranslationField)
        );
    }

    #[test]
    fn apply_translation_lookup_picks_nearest() {
        let field = TranslationField::LookupTable(LookupTableField {
            direction: Direction::Forward,
            rule: vec![
                rule("far", 5.0, 5.0, None),
                rule("origin", 0.0, 0.0, None),
                rule("medium", 0.5, 0.5, None),
            ],
            description: None,
        });
        let state = CanonicalState {
            chit: 0.05,
            gamma_AB: 0.05,
            k_frust: false,
        };
        let s = apply_translation(&state, &field, 1.0, DEFAULT_DOMAIN_DISTANCE_THRESHOLD, 1.0)
            .unwrap();
        assert_eq!(s.label.as_deref(), Some("origin"));
        assert_eq!(s.observables["canonical_chit"], 0.0);
    }

    #[test]
    fn forward_sweep_invert_grid_recovers_known_canonical() {
        // Identity tangent-flow (delta_chit = delta_gamma = 0): substrate ≡ canonical.
        let field = TranslationField::TangentFlow(tangent_flow_field(0.0, 0.0));
        let target_state = CanonicalState {
            chit: 0.7,
            gamma_AB: 0.3,
            k_frust: false,
        };
        let target = apply_translation(
            &target_state,
            &field,
            2.0,
            DEFAULT_DOMAIN_DISTANCE_THRESHOLD,
            1.0,
        )
        .unwrap();
        let grid: Vec<[f64; 2]> = (0..21)
            .flat_map(|i| {
                (0..21).map(move |j| {
                    [
                        (i as f64) * 0.05,
                        (j as f64) * 0.05,
                    ]
                })
            })
            .collect();
        let result = forward_sweep_invert_grid(&target, &field, 2.0, &grid, None, None).unwrap();
        // Grid resolution puts (0.7, 0.3) exactly on a candidate.
        assert!((result.best_state.chit - 0.7).abs() < 1e-12);
        assert!((result.best_state.gamma_AB - 0.3).abs() < 1e-12);
        assert!(result.best_residual < 1e-12);
    }

    #[test]
    fn forward_sweep_invert_grid_empty_grid_errors() {
        let field = TranslationField::TangentFlow(tangent_flow_field(0.0, 0.0));
        let target = SubstrateState {
            tau_obs: 1.0,
            label: None,
            axes: BTreeMap::new(),
            observables: BTreeMap::new(),
        };
        assert_eq!(
            forward_sweep_invert_grid(&target, &field, 1.0, &[], None, None),
            Err(OperationError::EmptyCanonicalGrid)
        );
    }

    #[test]
    fn gamut_classify_in_and_out() {
        let gamut = GamutSpec {
            chit_range: (-1.0, 1.0),
            gamma_AB_range: (-2.0, 2.0),
            tau_obs_range: Some((0.5, 5.0)),
            out_of_scope_residual_threshold: 0.05,
        };
        let state = CanonicalState {
            chit: 0.5,
            gamma_AB: 1.0,
            k_frust: false,
        };
        let r = gamut_classify(&state, 1.0, &gamut);
        assert!(r.in_gamut);
        assert!(r.diagnoses.is_empty());

        let out = CanonicalState {
            chit: 2.0,
            gamma_AB: 3.0,
            k_frust: false,
        };
        let r2 = gamut_classify(&out, 10.0, &gamut);
        assert!(!r2.in_gamut);
        assert_eq!(r2.diagnoses.len(), 3); // chit, gamma_AB, tau_obs
    }

    #[test]
    fn default_substrate_score_l2_over_shared_keys() {
        let mut a_obs = BTreeMap::new();
        a_obs.insert("x".to_string(), 1.0);
        a_obs.insert("y".to_string(), 2.0);
        let mut b_obs = BTreeMap::new();
        b_obs.insert("x".to_string(), 1.0);
        b_obs.insert("y".to_string(), 5.0);
        b_obs.insert("z".to_string(), 99.0); // not in a
        let a = SubstrateState {
            tau_obs: 1.0,
            label: None,
            axes: BTreeMap::new(),
            observables: a_obs,
        };
        let b = SubstrateState {
            tau_obs: 1.0,
            label: None,
            axes: BTreeMap::new(),
            observables: b_obs,
        };
        // (1-1)^2 + (2-5)^2 = 9. z not shared → 0 contribution.
        assert_eq!(default_substrate_score(&a, &b), 9.0);
    }

    #[test]
    fn default_substrate_score_filters_non_numeric_axes() {
        let mut axes_a = BTreeMap::new();
        axes_a.insert("k".to_string(), Value::from(3.0));
        axes_a.insert("label".to_string(), Value::String("alpha".to_string()));
        let mut axes_b = BTreeMap::new();
        axes_b.insert("k".to_string(), Value::from(0.0));
        axes_b.insert("label".to_string(), Value::String("beta".to_string()));
        let a = SubstrateState {
            tau_obs: 1.0,
            label: None,
            axes: axes_a,
            observables: BTreeMap::new(),
        };
        let b = SubstrateState {
            tau_obs: 1.0,
            label: None,
            axes: axes_b,
            observables: BTreeMap::new(),
        };
        // Only k contributes: (3-0)^2 = 9. label is non-numeric.
        assert_eq!(default_substrate_score(&a, &b), 9.0);
    }

    // -----------------------------------------------------------------
    // Session 5: forward_sweep_invert method dispatch (Auto / Grid /
    // Gradient). Mirrors the in-scope subset of
    // tests/test_gradient_inversion.py — BanachSubstrate-based +
    // wrapped-variant tests are skipped (banach.py / *_wrapped not
    // ported until later sessions).
    // -----------------------------------------------------------------

    fn identity_learned_field() -> LearnedField {
        // Matches Python tests/test_gradient_inversion.py::_identity_learned_field.
        // The 2-layer eps + 1/eps trick makes the MLP ~identity for small
        // chit/gamma: tanh(eps*x) / eps ≈ x. log_ratio at tau==tau_ref is
        // 0, dropped by the second layer's all-zero third column.
        let mut origin_axes = BTreeMap::new();
        origin_axes.insert("tau_obs".to_string(), Value::from(1.0));
        let origin = TranslationRule {
            operating_point: OperatingPoint {
                label: "origin".to_string(),
                gt: Gt::C,
                axes: origin_axes,
            },
            xdot_choice: "default".to_string(),
            canonical: CanonicalPoint {
                chit: 0.0,
                gamma_AB: 0.0,
                k_frust: false,
                method: "learned".to_string(),
                extras: BTreeMap::new(),
            },
        };
        let eps = 0.01_f64;
        let w1 = vec![
            vec![eps, 0.0, 0.0],
            vec![0.0, eps, 0.0],
            vec![0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0],
        ];
        let b1 = vec![0.0; 4];
        let w2 = vec![
            vec![1.0 / eps, 0.0, 0.0, 0.0],
            vec![0.0, 1.0 / eps, 0.0, 0.0],
        ];
        let b2 = vec![0.0, 0.0];
        LearnedField {
            direction: Direction::Forward,
            rule_at_origin: origin,
            weights: vec![MlpLayer { w: w1, b: b1 }, MlpLayer { w: w2, b: b2 }],
            architecture: vec![3, 4, 2],
            activation: Activation::Tanh,
            tau_obs_ref: 1.0,
            description: None,
        }
    }

    fn three_cell_lookup_field() -> TranslationField {
        // chit ∈ {-1, 0, +1}, gamma_AB = 0 — the classic 3-cell partition.
        let mut rules = Vec::new();
        for chit in [-1.0_f64, 0.0, 1.0] {
            let mut axes = BTreeMap::new();
            axes.insert("chit_label".to_string(), Value::from(chit));
            rules.push(TranslationRule {
                operating_point: OperatingPoint {
                    label: format!("chit={chit:+}"),
                    gt: if chit < 0.0 {
                        Gt::R
                    } else if chit > 0.0 {
                        Gt::C
                    } else {
                        Gt::S
                    },
                    axes,
                },
                xdot_choice: "x".to_string(),
                canonical: CanonicalPoint {
                    chit,
                    gamma_AB: 0.0,
                    k_frust: false,
                    method: "test".to_string(),
                    extras: BTreeMap::new(),
                },
            });
        }
        TranslationField::LookupTable(LookupTableField {
            direction: Direction::Forward,
            rule: rules,
            description: None,
        })
    }

    #[test]
    fn gradient_method_rejects_lookup_table() {
        let field = three_cell_lookup_field();
        let target = SubstrateState {
            tau_obs: 1.0,
            label: None,
            axes: BTreeMap::new(),
            observables: BTreeMap::new(),
        };
        let grid = [[0.0_f64, 0.0]];
        let err = forward_sweep_invert(
            &target, &field, 1.0, &grid, None, None, Method::Gradient, false,
        )
        .unwrap_err();
        assert_eq!(err, OperationError::GradientOnLookupTable);
    }

    #[test]
    fn auto_on_lookup_table_matches_grid() {
        // Mirror of TestMethodKwarg::test_method_grid_byte_identical_to_v4_on_lookup_table.
        let field = three_cell_lookup_field();
        let mut target_axes = BTreeMap::new();
        target_axes.insert("chit_label".to_string(), Value::from(1.0));
        let target = SubstrateState {
            tau_obs: 1.0,
            label: None,
            axes: target_axes,
            observables: BTreeMap::new(),
        };
        let grid = [[-1.0_f64, 0.0], [0.0, 0.0], [1.0, 0.0]];
        let auto =
            forward_sweep_invert(&target, &field, 1.0, &grid, None, None, Method::Auto, false)
                .unwrap();
        let grid_only =
            forward_sweep_invert(&target, &field, 1.0, &grid, None, None, Method::Grid, false)
                .unwrap();
        assert_eq!(auto.best_state.chit, grid_only.best_state.chit);
        assert_eq!(auto.best_state.gamma_AB, grid_only.best_state.gamma_AB);
    }

    #[test]
    fn tangent_flow_auto_exact_recovery() {
        // Mirror of TestTangentFlowClosedForm::test_exact_recovery_under_auto.
        // Closed-form inverse is exact at float64 across a range of tau_obs.
        let field = TranslationField::TangentFlow(tangent_flow_field(0.7, -0.4));
        let truth = CanonicalState {
            chit: 0.8,
            gamma_AB: -0.25,
            k_frust: false,
        };
        for &tau in &[0.5_f64, 1.0, 2.0, 5.0] {
            let target = apply_translation(
                &truth,
                &field,
                tau,
                DEFAULT_DOMAIN_DISTANCE_THRESHOLD,
                1.0,
            )
            .unwrap();
            // Coarse grid — irrelevant; closed-form ignores it.
            let coarse_grid = [[0.0_f64, 0.0], [0.5, -0.5]];
            let result = forward_sweep_invert(
                &target,
                &field,
                tau,
                &coarse_grid,
                None,
                None,
                Method::Auto,
                false,
            )
            .unwrap();
            assert!(
                (result.best_state.chit - truth.chit).abs() < 1e-12,
                "tau={tau}, chit={}",
                result.best_state.chit
            );
            assert!(
                (result.best_state.gamma_AB - truth.gamma_AB).abs() < 1e-12,
                "tau={tau}, gamma_AB={}",
                result.best_state.gamma_AB
            );
            assert!(result.best_residual < 1e-10);
            // Closed-form path skipped the grid (no return_residuals flag).
            assert!(result.residuals.is_none());
            assert!(result.best_index.is_none());
        }
    }

    #[test]
    fn tangent_flow_grid_remains_grid_resolution() {
        // Mirror of TestTangentFlowClosedForm::test_method_grid_remains_grid_resolution.
        let field = TranslationField::TangentFlow(tangent_flow_field(0.7, -0.4));
        let truth = CanonicalState {
            chit: 0.8,
            gamma_AB: -0.25,
            k_frust: false,
        };
        let target = apply_translation(
            &truth,
            &field,
            2.0,
            DEFAULT_DOMAIN_DISTANCE_THRESHOLD,
            1.0,
        )
        .unwrap();
        // 5×5 grid, step 0.25 in chit; truth 0.8 lands between cells.
        let coarse_grid: Vec<[f64; 2]> = (0..5)
            .flat_map(|i| {
                (0..5).map(move |j| {
                    [
                        (i as f64) * 0.25,
                        -0.5 + (j as f64) * 0.125,
                    ]
                })
            })
            .collect();
        let grid_result = forward_sweep_invert(
            &target,
            &field,
            2.0,
            &coarse_grid,
            None,
            None,
            Method::Grid,
            false,
        )
        .unwrap();
        // Grid snap — NOT exact.
        assert!((grid_result.best_state.chit - truth.chit).abs() > 1e-6);

        let auto_result = forward_sweep_invert(
            &target,
            &field,
            2.0,
            &coarse_grid,
            None,
            None,
            Method::Auto,
            false,
        )
        .unwrap();
        // Auto on the same coarse grid → exact via closed-form.
        assert!((auto_result.best_state.chit - truth.chit).abs() < 1e-12);
    }

    #[test]
    fn tangent_flow_gradient_same_as_auto() {
        // Mirror of TestTangentFlowClosedForm::test_method_gradient_same_as_auto_for_tangent_flow.
        let field = TranslationField::TangentFlow(tangent_flow_field(0.5, 0.0));
        let truth = CanonicalState {
            chit: 0.7,
            gamma_AB: -0.3,
            k_frust: false,
        };
        let target = apply_translation(
            &truth,
            &field,
            1.5,
            DEFAULT_DOMAIN_DISTANCE_THRESHOLD,
            1.0,
        )
        .unwrap();
        let grid = [[0.0_f64, 0.0]];
        let auto =
            forward_sweep_invert(&target, &field, 1.5, &grid, None, None, Method::Auto, false)
                .unwrap();
        let grad = forward_sweep_invert(
            &target, &field, 1.5, &grid, None, None, Method::Gradient, false,
        )
        .unwrap();
        assert_eq!(auto.best_state.chit, grad.best_state.chit);
        assert_eq!(auto.best_state.gamma_AB, grad.best_state.gamma_AB);
    }

    #[test]
    fn tangent_flow_return_residuals_works_with_auto() {
        // Mirror of TestTangentFlowClosedForm::test_residual_field_still_returned_under_auto.
        let field = TranslationField::TangentFlow(tangent_flow_field(0.5, -0.3));
        let truth = CanonicalState {
            chit: 0.7,
            gamma_AB: -0.3,
            k_frust: false,
        };
        let target = apply_translation(
            &truth,
            &field,
            1.0,
            DEFAULT_DOMAIN_DISTANCE_THRESHOLD,
            1.0,
        )
        .unwrap();
        let grid: Vec<[f64; 2]> = (0..5)
            .flat_map(|i| {
                (0..5).map(move |j| {
                    [
                        (i as f64) * 0.25,
                        -0.5 + (j as f64) * 0.125,
                    ]
                })
            })
            .collect();
        let result = forward_sweep_invert(
            &target,
            &field,
            1.0,
            &grid,
            None,
            None,
            Method::Auto,
            true, // return_residuals
        )
        .unwrap();
        let residuals = result.residuals.expect("residuals populated when return_residuals=true");
        assert_eq!(residuals.len(), 25);
        // best_state is still the closed-form result (exact), not the grid argmin.
        assert!((result.best_state.chit - truth.chit).abs() < 1e-12);
    }

    #[test]
    fn forward_map_override_forces_grid() {
        // Mirror of TestTangentFlowClosedForm::test_forward_map_override_forces_grid.
        // The custom forward_map gets called per grid cell, proving the grid
        // path executed even with Method::Auto on a TangentFlowField.
        let field = TranslationField::TangentFlow(tangent_flow_field(0.5, -0.3));
        let truth = CanonicalState {
            chit: 0.5,
            gamma_AB: 0.0,
            k_frust: false,
        };
        let target = apply_translation(
            &truth,
            &field,
            1.0,
            DEFAULT_DOMAIN_DISTANCE_THRESHOLD,
            1.0,
        )
        .unwrap();
        let grid: Vec<[f64; 2]> = (0..5).map(|i| [(i as f64) * 0.25, 0.0]).collect();
        let call_count = std::cell::Cell::new(0_usize);
        let custom_forward = |c: &CanonicalState, t: f64| -> SubstrateState {
            call_count.set(call_count.get() + 1);
            apply_translation(c, &field, t, DEFAULT_DOMAIN_DISTANCE_THRESHOLD, 1.0).unwrap()
        };
        let _result = forward_sweep_invert(
            &target,
            &field,
            1.0,
            &grid,
            None,
            Some(&custom_forward),
            Method::Auto,
            false,
        )
        .unwrap();
        assert_eq!(call_count.get(), 5, "forward_map should be called once per grid cell");
    }

    #[test]
    fn learned_field_bfgs_sub_grid_recovery() {
        // Mirror of TestLearnedFieldBFGS::test_recovers_canonical_at_sub_grid_resolution.
        // The identity-MLP forward map is ~quadratic near the minimum;
        // 2D damped Newton with FD gradient converges to <0.01 per axis
        // (BLOCK_IN §v6 session-5 acceptance budget).
        let field = TranslationField::Learned(identity_learned_field());
        let truth = CanonicalState {
            chit: 0.15,
            gamma_AB: -0.25,
            k_frust: false,
        };
        let target = apply_translation(
            &truth,
            &field,
            1.0,
            DEFAULT_DOMAIN_DISTANCE_THRESHOLD,
            1.0,
        )
        .unwrap();
        // 5×5 warm-start grid, step 0.25 — L-BFGS-equivalent refines past it.
        let grid: Vec<[f64; 2]> = (0..5)
            .flat_map(|i| {
                (0..5).map(move |j| {
                    [
                        -0.5 + (i as f64) * 0.25,
                        -0.5 + (j as f64) * 0.25,
                    ]
                })
            })
            .collect();
        let result = forward_sweep_invert(
            &target,
            &field,
            1.0,
            &grid,
            None,
            None,
            Method::Auto,
            false,
        )
        .unwrap();
        assert!(
            (result.best_state.chit - truth.chit).abs() < 0.01,
            "chit recovery error {:.6e} >= 0.01",
            (result.best_state.chit - truth.chit).abs()
        );
        assert!(
            (result.best_state.gamma_AB - truth.gamma_AB).abs() < 0.01,
            "gamma_AB recovery error {:.6e} >= 0.01",
            (result.best_state.gamma_AB - truth.gamma_AB).abs()
        );
        assert!(result.best_residual < 0.01);
    }

    #[test]
    fn learned_field_grid_method_stays_at_grid_resolution() {
        // Mirror of TestLearnedFieldBFGS::test_grid_method_falls_back_to_v4_resolution.
        let field = TranslationField::Learned(identity_learned_field());
        let truth = CanonicalState {
            chit: 0.15,
            gamma_AB: -0.25,
            k_frust: false,
        };
        let target = apply_translation(
            &truth,
            &field,
            1.0,
            DEFAULT_DOMAIN_DISTANCE_THRESHOLD,
            1.0,
        )
        .unwrap();
        let coarse_grid: Vec<[f64; 2]> = (0..5)
            .flat_map(|i| {
                (0..5).map(move |j| {
                    [
                        -0.5 + (i as f64) * 0.25,
                        -0.5 + (j as f64) * 0.25,
                    ]
                })
            })
            .collect();
        let result = forward_sweep_invert(
            &target,
            &field,
            1.0,
            &coarse_grid,
            None,
            None,
            Method::Grid,
            false,
        )
        .unwrap();
        // Grid step 0.25 — recovery error of that scale.
        assert!((result.best_state.chit - truth.chit).abs() > 0.05);
    }

    // -----------------------------------------------------------------
    // Session 6: intent algebra (intent_map + intent_compose). Mirror
    // of tests/test_intents.py — the wrapped-variant tests
    // (TestValidation) are skipped until session 7 lands validation.rs +
    // *_wrapped.
    // -----------------------------------------------------------------

    fn gamut_unit() -> GamutSpec {
        GamutSpec {
            chit_range: (-1.0, 1.0),
            gamma_AB_range: (-1.0, 1.0),
            tau_obs_range: None,
            out_of_scope_residual_threshold: 0.05,
        }
    }

    fn gamut_chit(lo: f64, hi: f64) -> GamutSpec {
        GamutSpec {
            chit_range: (lo, hi),
            gamma_AB_range: (-1.0, 1.0),
            tau_obs_range: None,
            out_of_scope_residual_threshold: 0.05,
        }
    }

    fn gamut_gamma(lo: f64, hi: f64) -> GamutSpec {
        GamutSpec {
            chit_range: (-1.0, 1.0),
            gamma_AB_range: (lo, hi),
            tau_obs_range: None,
            out_of_scope_residual_threshold: 0.05,
        }
    }

    fn state(chit: f64, gamma: f64) -> CanonicalState {
        CanonicalState {
            chit,
            gamma_AB: gamma,
            k_frust: false,
        }
    }

    // ---- I1 ---------------------------------------------------------

    #[test]
    fn i1_clamp_preserves_regime_when_possible() {
        let (mapped, sac) = intent_map(&state(2.0, 0.3), 1.0, &gamut_unit(), IntentId::I1);
        assert!(sac.invariant_preserved);
        match sac.diagnostics {
            IntentDiagnostics::I1 {
                regime_preserved,
                original_regime,
                mapped_regime,
                ..
            } => {
                assert!(regime_preserved);
                assert_eq!(original_regime, RegimeLabel::DeepC);
                assert_eq!(mapped_regime, RegimeLabel::DeepC);
            }
            _ => panic!("expected I1 diagnostics"),
        }
        assert_eq!(mapped.chit, 1.0);
    }

    #[test]
    fn i1_regime_unreachable_flags_sacrifice() {
        let (mapped, sac) =
            intent_map(&state(2.0, 0.0), 1.0, &gamut_chit(-0.5, 0.5), IntentId::I1);
        assert!(!sac.invariant_preserved);
        match sac.diagnostics {
            IntentDiagnostics::I1 {
                regime_preserved,
                original_regime,
                mapped_regime,
                ..
            } => {
                assert!(!regime_preserved);
                assert_eq!(original_regime, RegimeLabel::DeepC);
                assert_ne!(mapped_regime, RegimeLabel::DeepC);
            }
            _ => panic!("expected I1 diagnostics"),
        }
        // naive clamp falls back to gamut max
        assert_eq!(mapped.chit, 0.5);
    }

    #[test]
    fn i1_in_gamut_no_change() {
        let (mapped, sac) = intent_map(&state(0.3, -0.2), 1.0, &gamut_unit(), IntentId::I1);
        assert!(sac.invariant_preserved);
        assert_eq!(mapped.chit, 0.3);
        assert_eq!(mapped.gamma_AB, -0.2);
    }

    #[test]
    fn i1_gamma_sign_flip_flags_sacrifice() {
        let (_mapped, sac) =
            intent_map(&state(0.0, 0.5), 1.0, &gamut_gamma(-1.0, -0.1), IntentId::I1);
        assert!(!sac.invariant_preserved);
        match sac.diagnostics {
            IntentDiagnostics::I1 {
                gamma_AB_sign_preserved,
                original_gamma_AB_sign,
                mapped_gamma_AB_sign,
                ..
            } => {
                assert!(!gamma_AB_sign_preserved);
                assert_eq!(original_gamma_AB_sign, 1);
                assert_eq!(mapped_gamma_AB_sign, -1);
            }
            _ => panic!("expected I1 diagnostics"),
        }
    }

    #[test]
    fn i1_k_frust_propagated() {
        let s = CanonicalState {
            chit: 2.0,
            gamma_AB: 0.0,
            k_frust: true,
        };
        let (mapped, sac) = intent_map(&s, 1.0, &gamut_unit(), IntentId::I1);
        assert!(mapped.k_frust);
        match sac.diagnostics {
            IntentDiagnostics::I1 {
                k_frust_preserved, ..
            } => assert!(k_frust_preserved),
            _ => panic!("expected I1 diagnostics"),
        }
    }

    // ---- I2 ---------------------------------------------------------

    #[test]
    fn i2_in_gamut_passthrough() {
        let s = state(0.3, -0.2);
        let (mapped, sac) = intent_map(&s, 1.0, &gamut_unit(), IntentId::I2);
        assert_eq!(mapped, s);
        assert!(sac.invariant_preserved);
        match &sac.diagnostics {
            IntentDiagnostics::I2 {
                out_of_gamut_rejected,
                ..
            } => assert!(!out_of_gamut_rejected),
            _ => panic!("expected I2 diagnostics"),
        }
    }

    #[test]
    fn i2_out_of_gamut_unchanged_and_flagged() {
        let s = state(2.0, 0.0);
        let (mapped, sac) = intent_map(&s, 1.0, &gamut_unit(), IntentId::I2);
        assert_eq!(mapped, s); // NOT clamped
        assert!(!sac.invariant_preserved);
        assert_eq!(sac.delta_chit, 0.0);
        match sac.diagnostics {
            IntentDiagnostics::I2 {
                out_of_gamut_rejected,
                out_of_gamut_axes,
            } => {
                assert!(out_of_gamut_rejected);
                assert!(out_of_gamut_axes.iter().any(|a| a == "chit"));
            }
            _ => panic!("expected I2 diagnostics"),
        }
    }

    #[test]
    fn i2_both_axes_oog_listed() {
        let (_mapped, sac) = intent_map(&state(2.0, 2.0), 1.0, &gamut_unit(), IntentId::I2);
        match sac.diagnostics {
            IntentDiagnostics::I2 {
                out_of_gamut_axes, ..
            } => {
                assert!(out_of_gamut_axes.iter().any(|a| a == "chit"));
                assert!(out_of_gamut_axes.iter().any(|a| a == "gamma_AB"));
            }
            _ => panic!("expected I2 diagnostics"),
        }
    }

    // ---- I3 ---------------------------------------------------------

    #[test]
    fn i3_deep_state_preserves_capacity_class() {
        let (mapped, sac) = intent_map(&state(0.9, 0.0), 1.0, &gamut_unit(), IntentId::I3);
        assert!(sac.invariant_preserved);
        match sac.diagnostics {
            IntentDiagnostics::I3 {
                capacity_class,
                mapped_capacity_class,
                ..
            } => {
                assert_eq!(capacity_class, CapacityClass::Deep);
                assert_eq!(mapped_capacity_class, CapacityClass::Deep);
            }
            _ => panic!("expected I3 diagnostics"),
        }
        assert_eq!(mapped.chit, 0.9);
    }

    #[test]
    fn i3_deep_state_demoted_to_shallow_flags() {
        let (mapped, sac) =
            intent_map(&state(0.9, 0.0), 1.0, &gamut_chit(-0.5, 0.5), IntentId::I3);
        assert!(!sac.invariant_preserved);
        match sac.diagnostics {
            IntentDiagnostics::I3 {
                capacity_class,
                mapped_capacity_class,
                ..
            } => {
                assert_eq!(capacity_class, CapacityClass::Deep);
                assert_eq!(mapped_capacity_class, CapacityClass::Shallow);
            }
            _ => panic!("expected I3 diagnostics"),
        }
        assert_eq!(mapped.chit, 0.5);
    }

    #[test]
    fn i3_deep_state_clamped_to_gamut_edge_when_still_deep() {
        let (mapped, sac) = intent_map(&state(2.0, 0.0), 1.0, &gamut_unit(), IntentId::I3);
        assert!(sac.invariant_preserved);
        assert_eq!(mapped.chit, 1.0);
        match sac.diagnostics {
            IntentDiagnostics::I3 {
                capacity_class,
                mapped_capacity_class,
                ..
            } => {
                assert_eq!(capacity_class, CapacityClass::Deep);
                assert_eq!(mapped_capacity_class, CapacityClass::Deep);
            }
            _ => panic!("expected I3 diagnostics"),
        }
    }

    #[test]
    fn i3_k_frust_propagated() {
        let s = CanonicalState {
            chit: 0.9,
            gamma_AB: 0.0,
            k_frust: true,
        };
        let (mapped, sac) = intent_map(&s, 1.0, &gamut_unit(), IntentId::I3);
        assert!(mapped.k_frust);
        match sac.diagnostics {
            IntentDiagnostics::I3 { k_frust, .. } => assert!(k_frust),
            _ => panic!("expected I3 diagnostics"),
        }
    }

    // ---- I4 ---------------------------------------------------------

    #[test]
    fn i4_positive_gamma_kept_positive() {
        let (mapped, sac) = intent_map(&state(0.0, 2.0), 1.0, &gamut_unit(), IntentId::I4);
        assert!(sac.invariant_preserved);
        match sac.diagnostics {
            IntentDiagnostics::I4 {
                original_gamma_AB_sign,
                mapped_gamma_AB_sign,
            } => {
                assert_eq!(original_gamma_AB_sign, 1);
                assert_eq!(mapped_gamma_AB_sign, 1);
            }
            _ => panic!("expected I4 diagnostics"),
        }
        assert_eq!(mapped.gamma_AB, 1.0);
    }

    #[test]
    fn i4_sign_flip_flagged_when_gamut_excludes_sign() {
        let (_mapped, sac) =
            intent_map(&state(0.0, 0.5), 1.0, &gamut_gamma(-1.0, -0.1), IntentId::I4);
        assert!(!sac.invariant_preserved);
        match sac.diagnostics {
            IntentDiagnostics::I4 {
                original_gamma_AB_sign,
                mapped_gamma_AB_sign,
            } => {
                assert_eq!(original_gamma_AB_sign, 1);
                assert_eq!(mapped_gamma_AB_sign, -1);
            }
            _ => panic!("expected I4 diagnostics"),
        }
    }

    #[test]
    fn i4_zero_gamma_treated_as_signless() {
        let (mapped, sac) = intent_map(&state(0.0, 0.0), 1.0, &gamut_unit(), IntentId::I4);
        assert!(sac.invariant_preserved);
        assert_eq!(mapped.gamma_AB, 0.0);
    }

    // ---- I5 (uniform keys + v1 back-compat) -------------------------

    #[test]
    fn i5_carries_v23_invariant_keys_and_v1_back_compat() {
        let (_mapped, sac) = intent_map(&state(2.0, 0.0), 1.0, &gamut_unit(), IntentId::I5);
        // v2.3 uniform: preserved_invariant derived; invariant_preserved on outer.
        assert_eq!(sac.preserved_invariant(), "regime_label");
        // v1 back-compat: regime_preserved present in diagnostics, and
        // matches invariant_preserved (for I5 they are the same boolean).
        match sac.diagnostics {
            IntentDiagnostics::I5 {
                regime_preserved, ..
            } => assert_eq!(sac.invariant_preserved, regime_preserved),
            _ => panic!("expected I5 diagnostics"),
        }
    }

    // ---- Composition (RFC-S §3) -------------------------------------

    #[test]
    fn compose_single_intent_equals_intent_map() {
        let s = state(2.0, 0.5);
        let g = gamut_unit();
        let (mapped_a, sacs) = intent_compose(&s, 1.0, &g, &[IntentId::I5]).unwrap();
        let (mapped_b, sac) = intent_map(&s, 1.0, &g, IntentId::I5);
        assert_eq!(mapped_a, mapped_b);
        assert_eq!(sacs.len(), 1);
        match (&sacs[0].diagnostics, &sac.diagnostics) {
            (
                IntentDiagnostics::I5 {
                    regime_preserved: rp_a,
                    ..
                },
                IntentDiagnostics::I5 {
                    regime_preserved: rp_b,
                    ..
                },
            ) => assert_eq!(rp_a, rp_b),
            _ => panic!("expected I5 diagnostics on both"),
        }
    }

    #[test]
    fn compose_idempotent_under_same_intent() {
        let s = state(2.0, 0.0);
        let g = gamut_unit();
        let (once, _) = intent_compose(&s, 1.0, &g, &[IntentId::I3]).unwrap();
        let (twice, _) = intent_compose(&s, 1.0, &g, &[IntentId::I3, IntentId::I3]).unwrap();
        assert_eq!(once, twice);
    }

    #[test]
    fn compose_i1_then_i3_composable() {
        let s = state(2.0, 0.3);
        let g = gamut_unit();
        let (mapped, sacs) = intent_compose(&s, 1.0, &g, &[IntentId::I1, IntentId::I3]).unwrap();
        assert_eq!(sacs.len(), 2);
        assert!(sacs[0].invariant_preserved);
        assert!(sacs[1].invariant_preserved);
        assert!((-1.0..=1.0).contains(&mapped.chit));
        assert!((-1.0..=1.0).contains(&mapped.gamma_AB));
    }

    #[test]
    fn compose_i3_then_i4_composable() {
        let s = state(0.9, 2.0);
        let g = gamut_unit();
        let (mapped, sacs) = intent_compose(&s, 1.0, &g, &[IntentId::I3, IntentId::I4]).unwrap();
        assert!(sacs[0].invariant_preserved); // capacity preserved
        assert!(sacs[1].invariant_preserved); // sign preserved
        assert_eq!(mapped.chit, 0.9);
        assert_eq!(mapped.gamma_AB, 1.0);
    }

    #[test]
    fn compose_i2_does_not_compose() {
        let g = gamut_unit();
        let s = state(0.0, 0.0);
        assert_eq!(
            intent_compose(&s, 1.0, &g, &[IntentId::I1, IntentId::I2]).unwrap_err(),
            OperationError::I2InComposition
        );
        assert_eq!(
            intent_compose(&s, 1.0, &g, &[IntentId::I2, IntentId::I1]).unwrap_err(),
            OperationError::I2InComposition
        );
    }

    #[test]
    fn compose_i2_alone_is_legal() {
        let s = state(0.3, -0.2);
        let g = gamut_unit();
        let (mapped, sacs) = intent_compose(&s, 1.0, &g, &[IntentId::I2]).unwrap();
        assert_eq!(mapped, s);
        assert_eq!(sacs.len(), 1);
        assert_eq!(sacs[0].intent(), IntentId::I2);
    }

    #[test]
    fn compose_empty_intents_errors() {
        let s = state(0.0, 0.0);
        let g = gamut_unit();
        assert_eq!(
            intent_compose(&s, 1.0, &g, &[]).unwrap_err(),
            OperationError::IntentComposeEmpty
        );
    }

    #[test]
    fn compose_conflict_surfaces_in_sacrifice_trace() {
        // I1 on a deep_c state in a chit=[-0.5, 0.5] gamut breaks regime.
        // Composing I3 after that succeeds on the now in-gamut shallow state.
        let g = gamut_chit(-0.5, 0.5);
        let s = state(2.0, 0.0);
        let (_mapped, sacs) =
            intent_compose(&s, 1.0, &g, &[IntentId::I1, IntentId::I3]).unwrap();
        assert!(!sacs[0].invariant_preserved);
        assert!(sacs[1].invariant_preserved);
    }

    #[test]
    fn sacrifice_record_intent_and_preserved_invariant_methods() {
        // Spot-check every variant returns its expected pair.
        let g = gamut_unit();
        let (_m1, s1) = intent_map(&state(0.0, 0.0), 1.0, &g, IntentId::I1);
        assert_eq!(s1.intent(), IntentId::I1);
        assert_eq!(s1.preserved_invariant(), "regime ∧ sign(gamma_AB) ∧ k_frust");

        let (_m2, s2) = intent_map(&state(0.0, 0.0), 1.0, &g, IntentId::I2);
        assert_eq!(s2.intent(), IntentId::I2);
        assert_eq!(s2.preserved_invariant(), "exact_drive_parameters");

        let (_m3, s3) = intent_map(&state(0.0, 0.0), 1.0, &g, IntentId::I3);
        assert_eq!(s3.intent(), IntentId::I3);
        assert_eq!(s3.preserved_invariant(), "capacity_class ∧ k_frust");

        let (_m4, s4) = intent_map(&state(0.0, 0.0), 1.0, &g, IntentId::I4);
        assert_eq!(s4.intent(), IntentId::I4);
        assert_eq!(s4.preserved_invariant(), "sign(gamma_AB)");

        let (_m5, s5) = intent_map(&state(0.0, 0.0), 1.0, &g, IntentId::I5);
        assert_eq!(s5.intent(), IntentId::I5);
        assert_eq!(s5.preserved_invariant(), "regime_label");
    }

    #[test]
    fn sacrifice_record_serde_flatten_round_trip() {
        // The #[serde(flatten)] + #[serde(tag="intent")] combination should
        // produce a single flat JSON object. Round-trip is the smoke check.
        let (_m, sac) = intent_map(&state(2.0, 0.3), 1.0, &gamut_unit(), IntentId::I1);
        let json = serde_json::to_string(&sac).expect("serialize");
        // Sanity: the flat JSON contains both the common and intent-specific keys.
        assert!(json.contains("\"invariant_preserved\":"));
        assert!(json.contains("\"intent\":\"I1\""));
        assert!(json.contains("\"regime_preserved\":"));
        let round: SacrificeRecord = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(round, sac);
    }

    #[test]
    fn tau_obs_sweep_broadcast() {
        let field = TranslationField::TangentFlow(tangent_flow_field(0.0, 0.0));
        let target = SubstrateState {
            tau_obs: 1.0,
            label: None,
            axes: BTreeMap::new(),
            observables: {
                let mut m = BTreeMap::new();
                m.insert("substrate_chit".to_string(), 0.5);
                m.insert("substrate_gamma_AB".to_string(), 0.5);
                m
            },
        };
        let tau_grid = [0.5, 1.0, 2.0];
        let canonical_grid: Vec<[f64; 2]> = (0..11)
            .flat_map(|i| (0..11).map(move |j| [(i as f64) * 0.1, (j as f64) * 0.1]))
            .collect();
        let trajectory = tau_obs_sweep_grid(
            SweepTargets::Broadcast(&target),
            &field,
            &tau_grid,
            &canonical_grid,
            None,
            None,
        )
        .unwrap();
        assert_eq!(trajectory.len(), 3);
        for state in &trajectory {
            assert!((state.chit - 0.5).abs() < 1e-12);
            assert!((state.gamma_AB - 0.5).abs() < 1e-12);
        }
    }

    // -----------------------------------------------------------------------
    // Wrapped variants (session 7) — mirrors of tests/test_validation.py
    // TestWrappedReports + tests/test_provenance.py TestProvenanceOnEachWrappedOp.
    // -----------------------------------------------------------------------

    use crate::provenance::{SOLVER_VERSION, provenance_hash};
    use crate::validation::validation_flags_bitfield;

    fn substrate_obs(chit: f64, gamma_AB: f64, tau_obs: f64) -> SubstrateState {
        SubstrateState {
            tau_obs,
            label: None,
            axes: BTreeMap::new(),
            observables: {
                let mut m = BTreeMap::new();
                m.insert("substrate_chit".to_string(), chit);
                m.insert("substrate_gamma_AB".to_string(), gamma_AB);
                m
            },
        }
    }

    fn trivial_canonical() -> CanonicalState {
        CanonicalState {
            chit: 0.5,
            gamma_AB: -0.3,
            k_frust: false,
        }
    }

    fn trivial_field() -> TranslationField {
        // Identity tangent-flow at tau_obs=1 — substrate(chit) = chit (at ref).
        TranslationField::TangentFlow(tangent_flow_field(0.0, 0.0))
    }

    #[test]
    fn wrapped_apply_translation_report_shape() {
        let out = apply_translation_wrapped(
            &trivial_canonical(),
            &trivial_field(),
            1.0,
            DEFAULT_DOMAIN_DISTANCE_THRESHOLD,
            1.0,
            None,
        )
        .unwrap();
        assert!(out.validation.asymptotic_closure_compliant);
        assert!(out.validation.k_frust_invariant);
        assert!(out.validation.round_trip_residual.is_none());
        assert_eq!(out.provenance.operation, "apply_translation");
        assert_eq!(out.provenance.dispatch_path, DispatchPath::DirectCompute);
        assert_eq!(out.provenance.solver_version, SOLVER_VERSION);
    }

    #[test]
    fn wrapped_apply_translation_flags_zero_input() {
        let zero_chit = CanonicalState {
            chit: 0.0,
            gamma_AB: -0.3,
            k_frust: false,
        };
        let out = apply_translation_wrapped(
            &zero_chit,
            &trivial_field(),
            1.0,
            DEFAULT_DOMAIN_DISTANCE_THRESHOLD,
            1.0,
            None,
        )
        .unwrap();
        assert!(!out.validation.asymptotic_closure_compliant);
        assert!(out.validation.notes.iter().any(|n| n.contains("chit")));
    }

    #[test]
    fn wrapped_forward_sweep_invert_round_trip_small() {
        // Identity tangent flow at tau=1: every grid candidate's forward
        // map matches itself. Grid contains the truth -> residual ~ 0.
        let target = substrate_obs(0.4, -0.2, 1.0);
        let grid: Vec<[f64; 2]> = vec![[0.4, -0.2], [0.5, -0.3]];
        let out = forward_sweep_invert_wrapped(
            &target,
            &trivial_field(),
            1.0,
            &grid,
            None,
            None,
            None,
            true,
            Method::Grid,
        )
        .unwrap();
        let rt = out.validation.round_trip_residual.expect("residual set");
        assert!(rt < 1e-10, "round-trip residual {rt} not near zero");
        assert_eq!(out.provenance.operation, "forward_sweep_invert");
    }

    #[test]
    fn wrapped_tau_obs_sweep_notes_carry_frame_counts() {
        let target = substrate_obs(0.5, 0.5, 1.0);
        let tau_grid = [0.5, 1.0, 2.0];
        let canonical_grid: Vec<[f64; 2]> = (0..11)
            .flat_map(|i| (0..11).map(move |j| [(i as f64) * 0.1, (j as f64) * 0.1]))
            .collect();
        let out = tau_obs_sweep_wrapped(
            SweepTargets::Broadcast(&target),
            &trivial_field(),
            &tau_grid,
            &canonical_grid,
            None,
            None,
            None,
        )
        .unwrap();
        assert_eq!(out.value.len(), 3);
        assert!(
            out.provenance
                .notes
                .iter()
                .any(|n| n.starts_with("frames:"))
        );
        // No sidecar → every frame is direct_compute → aggregate is direct.
        assert_eq!(out.provenance.dispatch_path, DispatchPath::DirectCompute);
        assert_eq!(out.provenance.operation, "tau_obs_sweep");
    }

    #[test]
    fn wrapped_regime_at_carries_report() {
        let out = regime_at_wrapped(&trivial_canonical(), 1.0);
        assert!(out.validation.asymptotic_closure_compliant);
        assert_eq!(out.value.regime, RegimeLabel::CNearS);
        assert_eq!(out.provenance.operation, "regime_at");
    }

    #[test]
    fn wrapped_gamut_classify_carries_report() {
        let gamut = GamutSpec {
            chit_range: (-1.0, 1.0),
            gamma_AB_range: (-1.0, 1.0),
            tau_obs_range: None,
            out_of_scope_residual_threshold: 0.05,
        };
        let out = gamut_classify_wrapped(&trivial_canonical(), 1.0, &gamut);
        assert!(out.value.in_gamut);
        assert!(out.validation.asymptotic_closure_compliant);
        assert_eq!(out.provenance.operation, "gamut_classify");
    }

    #[test]
    fn wrapped_intent_map_flags_regime_break() {
        // out-of-gamut deep_c → I5 maps into [-0.5, 0.5] gamut at c_near_s.
        let gamut = GamutSpec {
            chit_range: (-0.5, 0.5),
            gamma_AB_range: (-1.0, 1.0),
            tau_obs_range: None,
            out_of_scope_residual_threshold: 0.05,
        };
        let oog = CanonicalState {
            chit: 2.0,
            gamma_AB: 0.0,
            k_frust: false,
        };
        let out = intent_map_wrapped(&oog, 1.0, &gamut, IntentId::I5);
        let (_mapped, sac) = &out.value;
        assert!(!sac.invariant_preserved);
        assert!(!out.validation.k_frust_invariant);
        assert!(out.validation.notes.iter().any(|n| n.contains("regime")));
        assert_eq!(out.provenance.operation, "intent_map");
    }

    #[test]
    fn wrapped_intent_compose_empty_errors() {
        let gamut = GamutSpec {
            chit_range: (-1.0, 1.0),
            gamma_AB_range: (-1.0, 1.0),
            tau_obs_range: None,
            out_of_scope_residual_threshold: 0.05,
        };
        let err = intent_compose_wrapped(&trivial_canonical(), 1.0, &gamut, &[])
            .err()
            .unwrap();
        assert_eq!(err, OperationError::IntentComposeEmpty);
    }

    #[test]
    fn wrapped_intent_compose_i2_in_composition_errors() {
        let gamut = GamutSpec {
            chit_range: (-1.0, 1.0),
            gamma_AB_range: (-1.0, 1.0),
            tau_obs_range: None,
            out_of_scope_residual_threshold: 0.05,
        };
        let err = intent_compose_wrapped(
            &trivial_canonical(),
            1.0,
            &gamut,
            &[IntentId::I1, IntentId::I2],
        )
        .err()
        .unwrap();
        assert_eq!(err, OperationError::I2InComposition);
    }

    #[test]
    fn wrapped_intent_compose_notes_carry_intent_repr() {
        let gamut = GamutSpec {
            chit_range: (-1.0, 1.0),
            gamma_AB_range: (-1.0, 1.0),
            tau_obs_range: None,
            out_of_scope_residual_threshold: 0.05,
        };
        let out = intent_compose_wrapped(
            &trivial_canonical(),
            1.0,
            &gamut,
            &[IntentId::I1, IntentId::I3],
        )
        .unwrap();
        assert_eq!(out.provenance.operation, "intent_compose");
        assert!(
            out.provenance
                .notes
                .iter()
                .any(|n| n.contains("intents=(I1, I3)"))
        );
    }

    #[test]
    fn wrapped_validate_driver_profile_one_cell() {
        let dataset = vec![ReferenceDatasetEntry {
            canonical_state: trivial_canonical(),
            tau_obs: 1.0,
            expected_substrate: None,
        }];
        let grid: Vec<[f64; 2]> = vec![[0.5, -0.3], [0.4, -0.2]];
        let out = validate_driver_profile_wrapped(
            &trivial_field(),
            &dataset,
            &grid,
            IntentId::I5,
            None,
        )
        .unwrap();
        assert_eq!(out.value.intent, IntentId::I5);
        assert_eq!(out.value.forward_residuals.len(), 1);
        assert_eq!(out.value.round_trip_residuals.len(), 1);
        // Round-trip ≈ 0 because the grid contains the truth.
        assert!(out.value.round_trip_mean < 1e-10);
        assert_eq!(out.provenance.operation, "validate_driver_profile");
    }

    #[test]
    fn wrapped_validate_driver_profile_empty_dataset_short_shape() {
        // Empty dataset → per_intent aggregate is {intent, n_cells} only.
        let grid: Vec<[f64; 2]> = vec![[0.0, 0.0]];
        let out = validate_driver_profile_wrapped(
            &trivial_field(),
            &[],
            &grid,
            IntentId::I5,
            None,
        )
        .unwrap();
        let per_intent = &out.value.per_intent;
        assert_eq!(per_intent.len(), 2);
        assert_eq!(per_intent["intent"], serde_json::json!("I5"));
        assert_eq!(per_intent["n_cells"], serde_json::json!(0));
    }

    #[test]
    fn wrapped_bitfield_all_pass_is_three() {
        // No round-trip residual on apply_translation_wrapped → bit 2 = 0;
        // bits 0,1 = 1 → 3.
        let out = apply_translation_wrapped(
            &trivial_canonical(),
            &trivial_field(),
            1.0,
            DEFAULT_DOMAIN_DISTANCE_THRESHOLD,
            1.0,
            None,
        )
        .unwrap();
        assert_eq!(validation_flags_bitfield(&out.validation), 3.0);
    }

    #[test]
    fn wrapped_bitfield_asymptotic_flag_drops_bit_zero() {
        let zero_chit = CanonicalState {
            chit: 0.0,
            gamma_AB: -0.3,
            k_frust: false,
        };
        let out = apply_translation_wrapped(
            &zero_chit,
            &trivial_field(),
            1.0,
            DEFAULT_DOMAIN_DISTANCE_THRESHOLD,
            1.0,
            None,
        )
        .unwrap();
        // bit 0 cleared, bits 1 still set → 2.
        assert_eq!(validation_flags_bitfield(&out.validation), 2.0);
    }

    #[test]
    fn wrapped_provenance_hash_stable_across_wrapped_calls() {
        // Two wrapped calls with the same shape stamp provenance records
        // whose hashes are byte-equal (timestamps excluded from the hash).
        let a = regime_at_wrapped(&trivial_canonical(), 1.0);
        let b = regime_at_wrapped(&trivial_canonical(), 1.0);
        assert_eq!(provenance_hash(&a.provenance), provenance_hash(&b.provenance));
    }

    // -----------------------------------------------------------------
    // Session 8 — posterior surface
    // -----------------------------------------------------------------

    fn lookup_field_three_rules() -> LookupTableField {
        // Three rules at (0.5, -0.3), (0.8, 0.2), (-0.4, 0.4).
        LookupTableField {
            direction: Direction::Forward,
            rule: vec![
                rule("a", 0.5, -0.3, Some(1.0)),
                rule("b", 0.8, 0.2, Some(1.0)),
                rule("c", -0.4, 0.4, Some(1.0)),
            ],
            description: None,
        }
    }

    #[test]
    fn tangent_flow_posterior_map_matches_closed_form_inverse() {
        // Recover canonical (1.2, 0.7) via the closed-form inverse — the
        // MAP point should be exact (no grid).
        let field = tangent_flow_field(0.3, 0.5);
        let canonical = CanonicalState {
            chit: 1.2,
            gamma_AB: 0.7,
            k_frust: false,
        };
        let s = apply_translation(
            &CanonicalState { ..canonical.clone() },
            &TranslationField::TangentFlow(field.clone()),
            2.0,
            DEFAULT_DOMAIN_DISTANCE_THRESHOLD,
            1.0,
        )
        .unwrap();
        let p = tangent_flow_posterior(&s, &field, 2.0, 1.0, false);
        // MAP recovers (1.2, 0.7) at float64 precision.
        assert!((p.mean.chit - 1.2).abs() < 1e-12);
        assert!((p.mean.gamma_AB - 0.7).abs() < 1e-12);
        // Covariance non-singular and symmetric.
        assert!(p.covariance[0][0] > 0.0);
        assert!(p.covariance[1][1] > 0.0);
        assert_eq!(p.covariance[0][1], p.covariance[1][0]);
        // Log evidence finite (residual at MAP is zero).
        let le = p.log_evidence.expect("closed-form posterior emits log_evidence");
        assert!(le.is_finite());
        assert_eq!(p.notes, vec!["laplace_from_closed_form_jacobian".to_string()]);
        assert!(p.modes.is_empty());
    }

    #[test]
    fn tangent_flow_posterior_k_frust_propagates_to_map() {
        let field = tangent_flow_field(0.0, 0.0);
        let s = substrate_obs(0.5, -0.3, 1.0);
        let p = tangent_flow_posterior(&s, &field, 1.0, 1.0, true);
        assert!(p.mean.k_frust);
    }

    #[test]
    fn tangent_flow_posterior_identity_jacobian_gives_isotropic_cov() {
        // delta_gamma=0 → Jacobian = I → cov = sigma^2 * I.
        let field = tangent_flow_field(0.0, 0.0);
        let s = substrate_obs(0.5, -0.3, 1.0);
        let p = tangent_flow_posterior(&s, &field, 1.0, 0.25, false);
        assert!((p.covariance[0][0] - 0.25).abs() < 1e-12);
        assert!((p.covariance[1][1] - 0.25).abs() < 1e-12);
        assert!(p.covariance[0][1].abs() < 1e-12);
    }

    #[test]
    fn lookup_table_posterior_k_eq_1_returns_delta_with_noise_floor() {
        let field = lookup_field_three_rules();
        // Substrate close to rule "a"'s canonical (chit=0.5, gamma=-0.3).
        let s = substrate_obs(0.5, -0.3, 1.0);
        let grid: Vec<[f64; 2]> = vec![[0.5, -0.3], [0.8, 0.2], [-0.4, 0.4]];
        let p = lookup_table_posterior(&s, &field, 1.0, &grid, 0.04, false, None, 1).unwrap();
        // Delta-posterior at MAP grid point.
        assert_eq!(p.mean.chit, 0.5);
        assert_eq!(p.mean.gamma_AB, -0.3);
        // Noise-floor covariance on the diagonal; off-diagonal zero.
        assert_eq!(p.covariance[0][0], 0.04);
        assert_eq!(p.covariance[1][1], 0.04);
        assert_eq!(p.covariance[0][1], 0.0);
        assert_eq!(p.covariance[1][0], 0.0);
        // No log_evidence on the discrete-grid path.
        assert!(p.log_evidence.is_none());
        assert!(p.modes.is_empty());
        assert!(p.notes[0].starts_with("lookup_table_grid_top_k=1"));
    }

    #[test]
    fn lookup_table_posterior_top_k_weighted_moments() {
        let field = lookup_field_three_rules();
        let s = substrate_obs(0.5, -0.3, 1.0);
        let grid: Vec<[f64; 2]> = vec![[0.5, -0.3], [0.8, 0.2], [-0.4, 0.4]];
        let p = lookup_table_posterior(&s, &field, 1.0, &grid, 1.0, false, None, 3).unwrap();
        // Weighted moments fall inside the convex hull of the grid points.
        assert!(p.mean.chit >= -0.4 && p.mean.chit <= 0.8);
        assert!(p.mean.gamma_AB >= -0.3 && p.mean.gamma_AB <= 0.4);
        // Covariance positive on the diagonal (multiple distinct support points).
        assert!(p.covariance[0][0] > 0.0);
        assert!(p.covariance[1][1] > 0.0);
        // modes carries the MAP point if it differs from the weighted mean.
        let map = (0.5_f64.to_bits(), (-0.3_f64).to_bits());
        let mean = (p.mean.chit.to_bits(), p.mean.gamma_AB.to_bits());
        if map != mean {
            assert_eq!(p.modes.len(), 1);
            assert_eq!(p.modes[0].chit, 0.5);
            assert_eq!(p.modes[0].gamma_AB, -0.3);
        } else {
            assert!(p.modes.is_empty());
        }
        assert!(p.notes[0].starts_with("lookup_table_weighted_moments_top_k="));
    }

    #[test]
    fn lookup_table_posterior_stable_tiebreak_on_residual() {
        // Equidistant grid around the target: any tiebreak must be
        // deterministic. Two back-to-back calls produce identical output.
        let field = lookup_field_three_rules();
        let s = substrate_obs(0.0, 0.0, 1.0);
        let grid: Vec<[f64; 2]> = vec![[1.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]];
        let p1 = lookup_table_posterior(&s, &field, 1.0, &grid, 1.0, false, None, 4).unwrap();
        let p2 = lookup_table_posterior(&s, &field, 1.0, &grid, 1.0, false, None, 4).unwrap();
        assert_eq!(p1.mean.chit.to_bits(), p2.mean.chit.to_bits());
        assert_eq!(p1.mean.gamma_AB.to_bits(), p2.mean.gamma_AB.to_bits());
        assert_eq!(p1.covariance, p2.covariance);
    }

    #[test]
    fn forward_sweep_invert_posterior_dispatches_per_shape() {
        let field_tf = TranslationField::TangentFlow(tangent_flow_field(0.0, 0.0));
        let field_lt = TranslationField::LookupTable(lookup_field_three_rules());
        let s = substrate_obs(0.5, -0.3, 1.0);
        let grid: Vec<[f64; 2]> = vec![[0.5, -0.3], [0.8, 0.2], [-0.4, 0.4]];

        // TangentFlow: grid optional; closed-form path.
        let p_tf = forward_sweep_invert_posterior(&s, &field_tf, 1.0, None, 1.0, false, None, 5)
            .unwrap();
        assert!(p_tf.log_evidence.is_some());

        // LookupTable + grid: discrete path.
        let p_lt = forward_sweep_invert_posterior(
            &s,
            &field_lt,
            1.0,
            Some(&grid),
            1.0,
            false,
            None,
            3,
        )
        .unwrap();
        assert!(p_lt.log_evidence.is_none());
    }

    #[test]
    fn forward_sweep_invert_posterior_lookup_without_grid_errors() {
        let field = TranslationField::LookupTable(lookup_field_three_rules());
        let s = substrate_obs(0.5, -0.3, 1.0);
        let err = forward_sweep_invert_posterior(&s, &field, 1.0, None, 1.0, false, None, 5)
            .unwrap_err();
        assert_eq!(err, OperationError::PosteriorRequiresCanonicalGrid);
    }

    #[test]
    fn forward_sweep_invert_posterior_learned_field_unsupported() {
        // Build a minimal LearnedField (one identity layer, 3→2).
        let layer = MlpLayer {
            w: vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]],
            b: vec![0.0, 0.0],
        };
        let lf = LearnedField {
            direction: Direction::Forward,
            rule_at_origin: rule("origin", 0.0, 0.0, None),
            weights: vec![layer],
            architecture: vec![3, 2],
            activation: Activation::Tanh,
            tau_obs_ref: 1.0,
            description: None,
        };
        let field = TranslationField::Learned(lf);
        let s = substrate_obs(0.5, -0.3, 1.0);
        let err = forward_sweep_invert_posterior(&s, &field, 1.0, None, 1.0, false, None, 5)
            .unwrap_err();
        assert_eq!(err, OperationError::PosteriorUnsupportedFieldShape);
    }

    #[test]
    fn forward_sweep_invert_posterior_wrapped_stamps_provenance() {
        let field = TranslationField::TangentFlow(tangent_flow_field(0.0, 0.0));
        let s = substrate_obs(0.5, -0.3, 1.0);
        let out = forward_sweep_invert_posterior_wrapped(
            &s, &field, 1.0, None, 1.0, false, None, 5,
        )
        .unwrap();
        assert_eq!(out.provenance.operation, "forward_sweep_invert_posterior");
        assert_eq!(out.provenance.dispatch_path, DispatchPath::DirectCompute);
        // round_trip_residual is None on the posterior wrapped path.
        assert!(out.validation.round_trip_residual.is_none());
        // Solver version stamps onto the provenance.
        assert_eq!(out.provenance.solver_version, crate::provenance::SOLVER_VERSION);
    }
}
