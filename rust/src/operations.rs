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
//!
//! Deferred to subsequent sessions (named in BLOCK_IN §v6):
//!   * `intent_map`, `intent_compose`, the five intent handlers (session 6)
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
    Activation, MlpLayer, learned_field_substrate, lookup_squared_distance,
    tangent_flow_canonical_inverse, tangent_flow_substrate,
};
use crate::optim::minimize_smooth_2d;
use crate::types::{
    CanonicalState, DisplayBand, GamutSpec, LearnedField, LookupTableField, RegimeLabel,
    RegimeReading, SubstrateState, TangentFlowField, TranslationField, TranslationRule,
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
}
