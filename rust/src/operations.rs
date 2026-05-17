//! The seven scale-solver operations — port of
//! `mpa_scale_solver/operations.py`.
//!
//! Scope of session 4 (raw forward path only):
//!   * `apply_translation` + the three field-shape dispatch helpers
//!   * `forward_sweep_invert` **grid path only** (`method = "grid"` in
//!     Python). Closed-form tangent-flow inverse and L-BFGS for
//!     `LearnedField` (Python `method = "auto" / "gradient"`) land in a
//!     subsequent session — they need an optimizer crate + a tolerance
//!     decision, both of which are out of scope here.
//!   * `tau_obs_sweep` (forced grid path under the same scope)
//!   * `regime_at` + `regime_display_band`
//!   * `gamut_classify`
//!
//! Deferred to subsequent sessions (named in BLOCK_IN §v6):
//!   * `forward_sweep_invert` with gradient methods (session 5)
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
    tangent_flow_substrate,
};
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
        }
    }
}

impl std::error::Error for OperationError {}

/// Mirrors Python's `DEFAULT_DOMAIN_DISTANCE_THRESHOLD = 1e9` — tables
/// specify their own threshold; this is "effectively off".
pub const DEFAULT_DOMAIN_DISTANCE_THRESHOLD: f64 = 1e9;

pub type ScoreFn = dyn Fn(&SubstrateState, &SubstrateState) -> f64;
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
    score_fn: Option<&ScoreFn>,
    forward_map: Option<&ForwardMap>,
) -> Result<GridInversionResult, OperationError> {
    if canonical_grid.is_empty() {
        return Err(OperationError::EmptyCanonicalGrid);
    }
    let score: &ScoreFn = score_fn.unwrap_or(&default_substrate_score);

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
    score_fn: Option<&ScoreFn>,
    forward_map: Option<&ForwardMap>,
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
