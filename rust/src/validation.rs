//! Per-call self-validation — port of `mpa_scale_solver/validation.py`
//! (handoff §C.5; BLOCK_IN §v6 session 7).
//!
//! Three flags ride on `ValidationReport`:
//!
//! - **Asymptotic-Closure compliance** (v9 §Asymptotic closure): no
//!   framework-prediction observable attains exact 0 or 1 at
//!   non-asymptotic points. The Banach substrate is the documented
//!   exception — it sits at the asymptotic limits by construction. For
//!   real substrates an exact 0.0 or 1.0 in a canonical / substrate
//!   float is a flag.
//! - **k_frust invariance** (v9 §Scale-relativity): the topological
//!   invariant is preserved across trajectory operations.
//! - **Round-trip residual**: optional; populated by inversion-side
//!   validators that recompose forward-then-back and report the gap.
//!
//! Flags are reported, not raised. Consumers decide whether to trust
//! borderline outputs.
//!
//! Per-intent RFC-S §5 metrics (`per_intent_cell_metric`,
//! `aggregate_per_intent_metrics`) return `BTreeMap<String, Value>` to
//! match Python's `dict[str, Any]` wire format — the consumer surface
//! (`DriverProfileSummary::per_intent` and the wrapped variants' JSON
//! emission) is dict-shaped in Python and the thin-discipline call is
//! to match Python rather than introduce a parallel typed shape that
//! callers would have to translate at the JSON boundary.

#![allow(non_snake_case)]

use std::collections::BTreeMap;

use serde_json::{json, Value};

use crate::gfdr_model::vertex_regime;
use crate::types::{
    CanonicalState, IntentId, RegimeLabel, SacrificeRecord, SubstrateState, ValidationReport,
};

// ---------------------------------------------------------------------------
// Asymptotic-Closure + k_frust checkers
// ---------------------------------------------------------------------------

/// Float literals that flag the Asymptotic-Closure Principle. Comparison
/// is exact equality on purpose: floats that arrived at literal 0.0 or
/// 1.0 are either inputs the framework forbids at non-asymptotic points,
/// or a numerical degeneracy worth surfacing.
const ASYMPTOTIC_FLOATS: [f64; 2] = [0.0, 1.0];

fn is_asymptotic_literal(value: f64) -> bool {
    ASYMPTOTIC_FLOATS.iter().any(|&lit| value == lit)
}

/// Asymptotic-Closure check on a CanonicalState's float channels.
pub fn check_asymptotic_closure_canonical(canonical: &CanonicalState) -> (bool, Vec<String>) {
    let mut notes = Vec::new();
    for (name, val) in [("chit", canonical.chit), ("gamma_AB", canonical.gamma_AB)] {
        if is_asymptotic_literal(val) {
            notes.push(format!(
                "canonical.{name} == {val} (asymptotic-closure flag; \
                 Banach substrate is the documented exception)"
            ));
        }
    }
    let ok = notes.is_empty();
    (ok, notes)
}

/// Asymptotic-Closure check on substrate observables.
///
/// `excluded_keys` carries the substrate's declared normalization
/// conventions (e.g. a substrate whose unit interval is `[0, 1]` by
/// construction). Keys in `excluded_keys` are skipped.
pub fn check_asymptotic_closure_substrate(
    substrate: &SubstrateState,
    excluded_keys: &[&str],
) -> (bool, Vec<String>) {
    let mut notes = Vec::new();
    for (k, &v) in &substrate.observables {
        if excluded_keys.iter().any(|ex| *ex == k.as_str()) {
            continue;
        }
        if is_asymptotic_literal(v) {
            notes.push(format!(
                "substrate.observables[{k:?}] == {v} (asymptotic-closure flag)"
            ));
        }
    }
    let ok = notes.is_empty();
    (ok, notes)
}

/// k_frust must not flip across a trajectory (v9 §Scale-relativity).
pub fn check_k_frust_invariance(trajectory: &[CanonicalState]) -> (bool, Vec<String>) {
    if trajectory.is_empty() {
        return (true, Vec::new());
    }
    let initial = trajectory[0].k_frust;
    let flips: Vec<usize> = trajectory
        .iter()
        .enumerate()
        .skip(1)
        .filter_map(|(i, s)| if s.k_frust != initial { Some(i) } else { None })
        .collect();
    if !flips.is_empty() {
        let note = format!(
            "k_frust flipped at frames {} (was {} initially)",
            python_list_repr(&flips),
            python_bool_repr(initial),
        );
        return (false, vec![note]);
    }
    (true, Vec::new())
}

// ---------------------------------------------------------------------------
// Per-operation report builders
// ---------------------------------------------------------------------------

pub fn report_for_apply_translation(
    canonical_in: &CanonicalState,
    substrate_out: &SubstrateState,
    excluded_substrate_keys: &[&str],
) -> ValidationReport {
    let mut notes = Vec::new();
    let (ac_c, n_c) = check_asymptotic_closure_canonical(canonical_in);
    let (ac_s, n_s) = check_asymptotic_closure_substrate(substrate_out, excluded_substrate_keys);
    notes.extend(n_c);
    notes.extend(n_s);
    ValidationReport {
        asymptotic_closure_compliant: ac_c && ac_s,
        k_frust_invariant: true, // vacuously: apply_translation is per-frame
        round_trip_residual: None,
        notes,
    }
}

pub fn report_for_forward_sweep_invert(
    target: &SubstrateState,
    recovered: &CanonicalState,
    round_trip_residual: Option<f64>,
    excluded_substrate_keys: &[&str],
) -> ValidationReport {
    let mut notes = Vec::new();
    let (ac_c, n_c) = check_asymptotic_closure_canonical(recovered);
    let (ac_s, n_s) = check_asymptotic_closure_substrate(target, excluded_substrate_keys);
    notes.extend(n_c);
    notes.extend(n_s);
    ValidationReport {
        asymptotic_closure_compliant: ac_c && ac_s,
        k_frust_invariant: true,
        round_trip_residual,
        notes,
    }
}

pub fn report_for_tau_obs_sweep(trajectory: &[CanonicalState]) -> ValidationReport {
    let mut notes = Vec::new();
    let (inv_ok, inv_notes) = check_k_frust_invariance(trajectory);
    notes.extend(inv_notes);
    let mut ac_ok = true;
    for (i, state) in trajectory.iter().enumerate() {
        let (ok, n) = check_asymptotic_closure_canonical(state);
        if !ok {
            ac_ok = false;
            for line in n {
                notes.push(format!("frame {i}: {line}"));
            }
        }
    }
    ValidationReport {
        asymptotic_closure_compliant: ac_ok,
        k_frust_invariant: inv_ok,
        round_trip_residual: None,
        notes,
    }
}

pub fn report_for_regime_at(canonical: &CanonicalState) -> ValidationReport {
    let (ac_c, n_c) = check_asymptotic_closure_canonical(canonical);
    ValidationReport {
        asymptotic_closure_compliant: ac_c,
        k_frust_invariant: true,
        round_trip_residual: None,
        notes: n_c,
    }
}

pub fn report_for_gamut_classify(canonical: &CanonicalState) -> ValidationReport {
    let (ac_c, n_c) = check_asymptotic_closure_canonical(canonical);
    ValidationReport {
        asymptotic_closure_compliant: ac_c,
        k_frust_invariant: true,
        round_trip_residual: None,
        notes: n_c,
    }
}

/// Intent invariance: the intent's named invariant rides the
/// `k_frust_invariant` slot on `ValidationReport` (v1 convention extended
/// by v2.3 — the field-name reuse is documented in the Python).
///
/// The Rust port takes a typed `SacrificeRecord` rather than the Python
/// `dict[str, Any]`: `invariant_preserved` is the outer struct field,
/// `intent` and `preserved_invariant` are derived methods. The v1 I5
/// fallback key (`regime_preserved`) is no longer relevant since the
/// typed record always carries `invariant_preserved`.
pub fn report_for_intent_map(
    original: &CanonicalState,
    mapped: &CanonicalState,
    sacrifice: &SacrificeRecord,
) -> ValidationReport {
    let mut notes = Vec::new();
    let (ac_orig, n_orig) = check_asymptotic_closure_canonical(original);
    let (ac_map, n_map) = check_asymptotic_closure_canonical(mapped);
    notes.extend(n_orig);
    notes.extend(n_map);

    let invariant_preserved = sacrifice.invariant_preserved;
    if !invariant_preserved {
        let intent = sacrifice.intent();
        match (intent, &sacrifice.diagnostics) {
            (
                IntentId::I5,
                crate::types::IntentDiagnostics::I5 {
                    original_regime,
                    mapped_regime,
                    ..
                },
            ) => {
                notes.push(format!(
                    "I5 regime not preserved: {} -> {}",
                    regime_label_str(*original_regime),
                    regime_label_str(*mapped_regime),
                ));
            }
            _ => {
                notes.push(format!(
                    "{} did not preserve {}",
                    intent_id_str(intent),
                    sacrifice.preserved_invariant(),
                ));
            }
        }
    }

    ValidationReport {
        asymptotic_closure_compliant: ac_orig && ac_map,
        k_frust_invariant: invariant_preserved,
        round_trip_residual: None,
        notes,
    }
}

/// Composition: `k_frust_invariant` is True iff every intent in the
/// chain preserved its invariant — a one-line `.iter().all()` per the
/// BLOCK_IN session-7-prep sketch.
pub fn report_for_intent_compose(
    original: &CanonicalState,
    mapped: &CanonicalState,
    sacrifices: &[SacrificeRecord],
) -> ValidationReport {
    let mut notes = Vec::new();
    let (ac_orig, n_orig) = check_asymptotic_closure_canonical(original);
    let (ac_map, n_map) = check_asymptotic_closure_canonical(mapped);
    notes.extend(n_orig);
    notes.extend(n_map);

    let all_preserved = sacrifices.iter().all(|sac| sac.invariant_preserved);
    for sac in sacrifices {
        if !sac.invariant_preserved {
            notes.push(format!(
                "{} did not preserve {}",
                intent_id_str(sac.intent()),
                sac.preserved_invariant(),
            ));
        }
    }

    ValidationReport {
        asymptotic_closure_compliant: ac_orig && ac_map,
        k_frust_invariant: all_preserved,
        round_trip_residual: None,
        notes,
    }
}

/// Typed return of `validate_driver_profile`. Mirrors Python's
/// `dict[str, Any]` shape — every key is present (no Optional fields).
/// The `per_intent` block stays dict-shaped (`BTreeMap<String, Value>`)
/// to match Python's per-intent aggregate output verbatim.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DriverProfileSummary {
    pub intent: IntentId,
    pub forward_residuals: Vec<f64>,
    pub round_trip_residuals: Vec<f64>,
    pub regime_agreements: Vec<bool>,
    pub forward_mean: f64,
    pub round_trip_mean: f64,
    pub regime_agreement_rate: f64,
    pub per_intent: BTreeMap<String, Value>,
}

pub fn report_for_validate_driver_profile(summary: &DriverProfileSummary) -> ValidationReport {
    let rt_mean = summary.round_trip_mean;
    let regime_rate = summary.regime_agreement_rate;
    let mut notes = Vec::new();
    if rt_mean > 0.0 && !rt_mean.is_finite() {
        notes.push(format!("round_trip_mean non-finite: {rt_mean}"));
    }
    if regime_rate < 1.0 {
        notes.push(format!(
            "regime_agreement_rate {:.4} < 1.0 ({} round-trip)",
            regime_rate,
            intent_id_str(summary.intent),
        ));
    }
    ValidationReport {
        asymptotic_closure_compliant: true, // not checked at the summary level
        k_frust_invariant: regime_rate == 1.0,
        round_trip_residual: Some(rt_mean),
        notes,
    }
}

// ---------------------------------------------------------------------------
// Per-intent RFC-S §5 metrics
// ---------------------------------------------------------------------------

fn capacity_class_str(chit: f64) -> &'static str {
    if chit.abs() >= 0.7 {
        "deep"
    } else {
        "shallow"
    }
}

fn sign_int(x: f64) -> i32 {
    if x > 0.0 {
        1
    } else if x < 0.0 {
        -1
    } else {
        0
    }
}

/// Per-cell metric components for the named intent (RFC-S §5).
///
/// Returns a dict whose keys are intent-specific. `validate_driver_profile`
/// aggregates these into summary statistics via
/// `aggregate_per_intent_metrics`.
///
/// `in_gamut` (optional) supplies the cell's gamut residency for the I4
/// survival-declaration component. `None` when the caller has not run
/// `gamut_classify` for this cell.
pub fn per_intent_cell_metric(
    intent_id: IntentId,
    original: &CanonicalState,
    recovered: &CanonicalState,
    in_gamut: Option<bool>,
) -> BTreeMap<String, Value> {
    let mut out: BTreeMap<String, Value> = BTreeMap::new();
    match intent_id {
        IntentId::I1 => {
            let regime_match = vertex_regime(original.chit) == vertex_regime(recovered.chit);
            let sign_match = sign_int(original.gamma_AB) == sign_int(recovered.gamma_AB);
            let k_frust_match = original.k_frust == recovered.k_frust;
            out.insert("regime_match".into(), json!(regime_match));
            out.insert("edge_type_match".into(), json!(sign_match));
            out.insert("k_frust_match".into(), json!(k_frust_match));
            out.insert(
                "hamming".into(),
                json!(if regime_match && sign_match { 0 } else { 1 }),
            );
        }
        IntentId::I2 => {
            let d_chit = recovered.chit - original.chit;
            let d_gamma = recovered.gamma_AB - original.gamma_AB;
            out.insert(
                "l2_drive".into(),
                json!((d_chit * d_chit + d_gamma * d_gamma).sqrt()),
            );
            out.insert("gamma_deviation".into(), json!(d_gamma.abs()));
        }
        IntentId::I3 => {
            let gamma_star_original = original.chit.abs() - 0.7;
            let gamma_star_recovered = recovered.chit.abs() - 0.7;
            out.insert(
                "gamma_star_deviation".into(),
                json!((gamma_star_recovered - gamma_star_original).abs()),
            );
            out.insert(
                "capacity_class_match".into(),
                json!(capacity_class_str(original.chit) == capacity_class_str(recovered.chit)),
            );
            out.insert(
                "k_frust_match".into(),
                json!(original.k_frust == recovered.k_frust),
            );
        }
        IntentId::I4 => {
            let sign_match = sign_int(original.gamma_AB) == sign_int(recovered.gamma_AB);
            out.insert(
                "epsilon_sequence_distance".into(),
                json!(if sign_match { 0 } else { 1 }),
            );
            out.insert("survival".into(), json!(in_gamut.unwrap_or(true)));
        }
        IntentId::I5 => {
            let original_class = vertex_regime(original.chit);
            let recovered_class = vertex_regime(recovered.chit);
            let class_match = original_class == recovered_class;
            let d_chit = recovered.chit - original.chit;
            let d_gamma = recovered.gamma_AB - original.gamma_AB;
            out.insert("universality_class_match".into(), json!(class_match));
            out.insert(
                "intra_class_l2".into(),
                if class_match {
                    json!((d_chit * d_chit + d_gamma * d_gamma).sqrt())
                } else {
                    Value::Null
                },
            );
            out.insert("original_class".into(), json!(original_class));
            out.insert("recovered_class".into(), json!(recovered_class));
        }
    }
    out
}

/// Aggregate per-cell intent metrics into summary statistics.
///
/// Returns the intent-specific summary block that rides next to the
/// `DriverProfileSummary`'s `forward_residuals` / `round_trip_residuals`
/// keys. Matches Python's `dict[str, Any]` wire format including the
/// `n == 0` short-shape: when there are no cells, returns
/// `{"intent": "<id>", "n_cells": 0}` — strictly two keys, no
/// aggregates. The shape divergence is by design and matches Python.
pub fn aggregate_per_intent_metrics(
    intent_id: IntentId,
    cells: &[BTreeMap<String, Value>],
) -> BTreeMap<String, Value> {
    let mut out: BTreeMap<String, Value> = BTreeMap::new();
    let n = cells.len();
    out.insert("intent".into(), json!(intent_id_str(intent_id)));
    out.insert("n_cells".into(), json!(n));
    if n == 0 {
        return out;
    }
    let n_f = n as f64;
    match intent_id {
        IntentId::I1 => {
            let hamming = cells
                .iter()
                .map(|c| c["hamming"].as_i64().expect("hamming i64") as f64)
                .sum::<f64>()
                / n_f;
            let regime_rate = bool_rate(cells, "regime_match");
            let edge_rate = bool_rate(cells, "edge_type_match");
            let k_frust_rate = bool_rate(cells, "k_frust_match");
            out.insert("hamming_rate".into(), json!(hamming));
            out.insert("regime_match_rate".into(), json!(regime_rate));
            out.insert("edge_type_match_rate".into(), json!(edge_rate));
            out.insert("k_frust_match_rate".into(), json!(k_frust_rate));
        }
        IntentId::I2 => {
            let l2s: Vec<f64> = cells
                .iter()
                .map(|c| c["l2_drive"].as_f64().expect("l2_drive f64"))
                .collect();
            let gdevs: Vec<f64> = cells
                .iter()
                .map(|c| c["gamma_deviation"].as_f64().expect("gamma_deviation f64"))
                .collect();
            out.insert("l2_drive_mean".into(), json!(mean(&l2s)));
            out.insert("l2_drive_max".into(), json!(max(&l2s)));
            out.insert("gamma_deviation_max".into(), json!(max(&gdevs)));
            out.insert("gamma_deviation_mean".into(), json!(mean(&gdevs)));
        }
        IntentId::I3 => {
            let gsds: Vec<f64> = cells
                .iter()
                .map(|c| {
                    c["gamma_star_deviation"]
                        .as_f64()
                        .expect("gamma_star_deviation f64")
                })
                .collect();
            out.insert("gamma_star_deviation_mean".into(), json!(mean(&gsds)));
            out.insert("gamma_star_deviation_max".into(), json!(max(&gsds)));
            out.insert(
                "capacity_class_match_rate".into(),
                json!(bool_rate(cells, "capacity_class_match")),
            );
            out.insert(
                "k_frust_match_rate".into(),
                json!(bool_rate(cells, "k_frust_match")),
            );
        }
        IntentId::I4 => {
            let seq: Vec<f64> = cells
                .iter()
                .map(|c| {
                    c["epsilon_sequence_distance"]
                        .as_i64()
                        .expect("epsilon_sequence_distance i64") as f64
                })
                .collect();
            out.insert("epsilon_sequence_distance_mean".into(), json!(mean(&seq)));
            out.insert("survival_rate".into(), json!(bool_rate(cells, "survival")));
        }
        IntentId::I5 => {
            let matches = bool_rate(cells, "universality_class_match");
            let intra: Vec<f64> = cells
                .iter()
                .filter_map(|c| c["intra_class_l2"].as_f64())
                .collect();
            out.insert("universality_class_agreement_rate".into(), json!(matches));
            out.insert(
                "intra_class_l2_mean".into(),
                json!(if intra.is_empty() {
                    0.0
                } else {
                    mean(&intra)
                }),
            );
            out.insert(
                "intra_class_l2_max".into(),
                json!(if intra.is_empty() { 0.0 } else { max(&intra) }),
            );
        }
    }
    out
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn max(xs: &[f64]) -> f64 {
    xs.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

fn bool_rate(cells: &[BTreeMap<String, Value>], key: &str) -> f64 {
    let n = cells.len() as f64;
    cells
        .iter()
        .filter(|c| c[key].as_bool().expect("bool field"))
        .count() as f64
        / n
}

// ---------------------------------------------------------------------------
// Bitfield encoder
// ---------------------------------------------------------------------------

/// Float32-encoded bitfield of the report's pass/fail flags
/// (handoff §A.5).
///
/// Bit 0: `asymptotic_closure_compliant`
/// Bit 1: `k_frust_invariant`
/// Bit 2: `round_trip_residual` present (1) or None (0)
///
/// Encoded as a small integer cast to float so it survives the EXR
/// channel's float32 storage without quantization.
pub fn validation_flags_bitfield(report: &ValidationReport) -> f64 {
    let mut bits: u32 = 0;
    if report.asymptotic_closure_compliant {
        bits |= 1 << 0;
    }
    if report.k_frust_invariant {
        bits |= 1 << 1;
    }
    if report.round_trip_residual.is_some() {
        bits |= 1 << 2;
    }
    bits as f64
}

// ---------------------------------------------------------------------------
// Python-parity string helpers (for note formatting)
// ---------------------------------------------------------------------------

fn intent_id_str(id: IntentId) -> &'static str {
    match id {
        IntentId::I1 => "I1",
        IntentId::I2 => "I2",
        IntentId::I3 => "I3",
        IntentId::I4 => "I4",
        IntentId::I5 => "I5",
    }
}

fn regime_label_str(label: RegimeLabel) -> &'static str {
    match label {
        RegimeLabel::DeepC => "deep_c",
        RegimeLabel::CNearS => "c_near_s",
        RegimeLabel::SCritical => "s_critical",
        RegimeLabel::RNearS => "r_near_s",
        RegimeLabel::DeepR => "deep_r",
    }
}

/// `f"{flips!r}"` for a list of usize. Matches Python's default
/// `repr([1, 2, 3])` = `"[1, 2, 3]"`.
fn python_list_repr(flips: &[usize]) -> String {
    let body = flips
        .iter()
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{body}]")
}

/// `f"{initial!r}"` for a bool. Matches Python's `repr(True)` = `"True"`,
/// `repr(False)` = `"False"`.
fn python_bool_repr(b: bool) -> &'static str {
    if b {
        "True"
    } else {
        "False"
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CapacityClass, IntentDiagnostics};

    fn canonical(chit: f64, gamma_AB: f64) -> CanonicalState {
        CanonicalState {
            chit,
            gamma_AB,
            k_frust: false,
        }
    }

    fn substrate(observables: &[(&str, f64)]) -> SubstrateState {
        SubstrateState {
            tau_obs: 1.0,
            label: None,
            axes: BTreeMap::new(),
            observables: observables
                .iter()
                .map(|(k, v)| (k.to_string(), *v))
                .collect(),
        }
    }

    // -------- Asymptotic closure --------

    #[test]
    fn asymptotic_closure_zero_chit_flags() {
        let (ok, notes) = check_asymptotic_closure_canonical(&canonical(0.0, 0.5));
        assert!(!ok);
        assert!(notes.iter().any(|n| n.contains("chit")));
    }

    #[test]
    fn asymptotic_closure_one_gamma_flags() {
        let (ok, notes) = check_asymptotic_closure_canonical(&canonical(0.5, 1.0));
        assert!(!ok);
        assert!(notes.iter().any(|n| n.contains("gamma_AB")));
    }

    #[test]
    fn asymptotic_closure_clean_state_passes() {
        let (ok, notes) = check_asymptotic_closure_canonical(&canonical(0.5, -0.3));
        assert!(ok);
        assert!(notes.is_empty());
    }

    #[test]
    fn asymptotic_closure_substrate_observable_zero_flags() {
        let s = substrate(&[("substrate_chit", 0.0), ("substrate_gamma_AB", -0.2)]);
        let (ok, notes) = check_asymptotic_closure_substrate(&s, &[]);
        assert!(!ok);
        assert!(notes.iter().any(|n| n.contains("substrate_chit")));
    }

    #[test]
    fn asymptotic_closure_excluded_keys_skipped() {
        let s = substrate(&[("normalized_unit", 1.0), ("substrate_chit", 0.3)]);
        let (ok, notes) = check_asymptotic_closure_substrate(&s, &["normalized_unit"]);
        assert!(ok);
        assert!(notes.is_empty());
    }

    // -------- k_frust invariance --------

    #[test]
    fn k_frust_constant_trajectory_passes() {
        let traj = vec![
            CanonicalState { chit: 0.5, gamma_AB: 0.0, k_frust: true };
            5
        ];
        let (ok, notes) = check_k_frust_invariance(&traj);
        assert!(ok);
        assert!(notes.is_empty());
    }

    #[test]
    fn k_frust_flipped_flags() {
        let traj = vec![
            CanonicalState { chit: 0.5, gamma_AB: 0.0, k_frust: true },
            CanonicalState { chit: 0.4, gamma_AB: 0.0, k_frust: true },
            CanonicalState { chit: 0.3, gamma_AB: 0.0, k_frust: false },
            CanonicalState { chit: 0.2, gamma_AB: 0.0, k_frust: false },
        ];
        let (ok, notes) = check_k_frust_invariance(&traj);
        assert!(!ok);
        assert!(notes.iter().any(|n| n.contains("flipped")));
        // Python repr parity: list of frame indices and initial bool.
        assert!(notes[0].contains("[2, 3]"));
        assert!(notes[0].contains("True"));
    }

    #[test]
    fn k_frust_empty_trajectory_passes() {
        let (ok, notes) = check_k_frust_invariance(&[]);
        assert!(ok);
        assert!(notes.is_empty());
    }

    // -------- Report shapes --------

    #[test]
    fn report_apply_translation_clean() {
        let canonical_in = canonical(0.5, -0.3);
        let substrate_out = substrate(&[("substrate_chit", 0.4), ("substrate_gamma_AB", -0.25)]);
        let r = report_for_apply_translation(&canonical_in, &substrate_out, &[]);
        assert!(r.asymptotic_closure_compliant);
        assert!(r.k_frust_invariant);
        assert!(r.round_trip_residual.is_none());
    }

    #[test]
    fn report_apply_translation_zero_input_flags() {
        let canonical_in = canonical(0.0, -0.3);
        let substrate_out = substrate(&[("substrate_chit", 0.4)]);
        let r = report_for_apply_translation(&canonical_in, &substrate_out, &[]);
        assert!(!r.asymptotic_closure_compliant);
        assert!(r.notes.iter().any(|n| n.contains("chit")));
    }

    #[test]
    fn report_forward_sweep_invert_carries_round_trip() {
        let target = substrate(&[("substrate_chit", 0.3)]);
        let recovered = canonical(0.5, -0.3);
        let r = report_for_forward_sweep_invert(&target, &recovered, Some(1.0e-12), &[]);
        assert_eq!(r.round_trip_residual, Some(1.0e-12));
        assert!(r.asymptotic_closure_compliant);
    }

    #[test]
    fn report_tau_obs_sweep_flags_k_frust_flip() {
        let traj = vec![
            CanonicalState { chit: 0.5, gamma_AB: -0.3, k_frust: false },
            CanonicalState { chit: 0.4, gamma_AB: -0.2, k_frust: true },
        ];
        let r = report_for_tau_obs_sweep(&traj);
        assert!(!r.k_frust_invariant);
        assert!(r.notes.iter().any(|n| n.contains("flipped")));
    }

    #[test]
    fn report_tau_obs_sweep_flags_per_frame_asymptotic() {
        let traj = vec![
            CanonicalState { chit: 0.5, gamma_AB: -0.3, k_frust: false },
            CanonicalState { chit: 0.0, gamma_AB: -0.2, k_frust: false },
        ];
        let r = report_for_tau_obs_sweep(&traj);
        assert!(r.k_frust_invariant);
        assert!(!r.asymptotic_closure_compliant);
        assert!(r.notes.iter().any(|n| n.starts_with("frame 1:")));
    }

    #[test]
    fn report_intent_map_invariant_break_flags() {
        // I5 with regime break → invariant_preserved false → k_frust_invariant false
        let original = canonical(0.5, 0.1);
        let mapped = canonical(0.4, 0.1);
        let sac = SacrificeRecord {
            invariant_preserved: false,
            delta_chit: -0.1,
            delta_gamma_AB: 0.0,
            diagnostics: IntentDiagnostics::I5 {
                regime_preserved: false,
                original_regime: RegimeLabel::DeepC,
                mapped_regime: RegimeLabel::CNearS,
            },
        };
        let r = report_for_intent_map(&original, &mapped, &sac);
        assert!(!r.k_frust_invariant);
        assert!(
            r.notes
                .iter()
                .any(|n| n.contains("I5 regime not preserved"))
        );
    }

    #[test]
    fn report_intent_compose_all_preserved_passes() {
        let original = canonical(0.5, 0.1);
        let mapped = canonical(0.4, 0.1);
        let sacrifices = vec![
            SacrificeRecord {
                invariant_preserved: true,
                delta_chit: -0.1,
                delta_gamma_AB: 0.0,
                diagnostics: IntentDiagnostics::I3 {
                    capacity_class: CapacityClass::Shallow,
                    mapped_capacity_class: CapacityClass::Shallow,
                    k_frust: false,
                    k_frust_preserved: true,
                },
            },
            SacrificeRecord {
                invariant_preserved: true,
                delta_chit: 0.0,
                delta_gamma_AB: 0.0,
                diagnostics: IntentDiagnostics::I5 {
                    regime_preserved: true,
                    original_regime: RegimeLabel::CNearS,
                    mapped_regime: RegimeLabel::CNearS,
                },
            },
        ];
        let r = report_for_intent_compose(&original, &mapped, &sacrifices);
        assert!(r.k_frust_invariant);
    }

    #[test]
    fn report_intent_compose_one_failure_flags() {
        let original = canonical(0.5, 0.1);
        let mapped = canonical(0.4, 0.1);
        let sacrifices = vec![
            SacrificeRecord {
                invariant_preserved: false,
                delta_chit: -0.1,
                delta_gamma_AB: 0.0,
                diagnostics: IntentDiagnostics::I3 {
                    capacity_class: CapacityClass::Deep,
                    mapped_capacity_class: CapacityClass::Shallow,
                    k_frust: false,
                    k_frust_preserved: true,
                },
            },
            SacrificeRecord {
                invariant_preserved: true,
                delta_chit: 0.0,
                delta_gamma_AB: 0.0,
                diagnostics: IntentDiagnostics::I5 {
                    regime_preserved: true,
                    original_regime: RegimeLabel::CNearS,
                    mapped_regime: RegimeLabel::CNearS,
                },
            },
        ];
        let r = report_for_intent_compose(&original, &mapped, &sacrifices);
        assert!(!r.k_frust_invariant);
        assert!(r.notes.iter().any(|n| n.contains("I3 did not preserve")));
    }

    // -------- Per-intent metrics --------

    #[test]
    fn per_intent_cell_metric_i1_shape() {
        let metric = per_intent_cell_metric(
            IntentId::I1,
            &canonical(0.5, 0.1),
            &canonical(0.5, 0.1),
            None,
        );
        assert_eq!(metric["regime_match"], json!(true));
        assert_eq!(metric["edge_type_match"], json!(true));
        assert_eq!(metric["k_frust_match"], json!(true));
        assert_eq!(metric["hamming"], json!(0));
    }

    #[test]
    fn per_intent_cell_metric_i2_l2_drive() {
        let metric = per_intent_cell_metric(
            IntentId::I2,
            &canonical(0.5, 0.0),
            &canonical(0.6, 0.0),
            None,
        );
        let l2 = metric["l2_drive"].as_f64().unwrap();
        assert!((l2 - 0.1).abs() < 1e-12);
    }

    #[test]
    fn per_intent_cell_metric_i5_intra_class_l2_present_on_match() {
        let metric = per_intent_cell_metric(
            IntentId::I5,
            &canonical(0.5, 0.0),
            &canonical(0.55, 0.0),
            None,
        );
        assert_eq!(metric["universality_class_match"], json!(true));
        assert!(metric["intra_class_l2"].as_f64().is_some());
    }

    #[test]
    fn per_intent_cell_metric_i5_intra_class_l2_null_on_mismatch() {
        let metric = per_intent_cell_metric(
            IntentId::I5,
            &canonical(0.95, 0.0),  // deep_c
            &canonical(0.5, 0.0),   // c_near_s
            None,
        );
        assert_eq!(metric["universality_class_match"], json!(false));
        assert_eq!(metric["intra_class_l2"], Value::Null);
    }

    #[test]
    fn aggregate_per_intent_metrics_empty_short_shape() {
        let agg = aggregate_per_intent_metrics(IntentId::I1, &[]);
        assert_eq!(agg.len(), 2, "n_cells==0 returns just intent + n_cells");
        assert_eq!(agg["intent"], json!("I1"));
        assert_eq!(agg["n_cells"], json!(0));
    }

    #[test]
    fn aggregate_per_intent_metrics_i1_rates() {
        let cells = vec![
            per_intent_cell_metric(
                IntentId::I1,
                &canonical(0.5, 0.1),
                &canonical(0.5, 0.1),
                None,
            ),
            per_intent_cell_metric(
                IntentId::I1,
                &canonical(0.95, 0.1),
                &canonical(0.5, -0.1),
                None,
            ),
        ];
        let agg = aggregate_per_intent_metrics(IntentId::I1, &cells);
        assert_eq!(agg["n_cells"], json!(2));
        // First cell matches everything (rate contribution 1); second is full mismatch.
        let hamming = agg["hamming_rate"].as_f64().unwrap();
        assert!((hamming - 0.5).abs() < 1e-12);
    }

    // -------- Driver-profile summary report --------

    #[test]
    fn report_validate_driver_profile_clean() {
        let summary = DriverProfileSummary {
            intent: IntentId::I5,
            forward_residuals: vec![0.0, 0.0],
            round_trip_residuals: vec![0.0, 0.0],
            regime_agreements: vec![true, true],
            forward_mean: 0.0,
            round_trip_mean: 0.0,
            regime_agreement_rate: 1.0,
            per_intent: BTreeMap::new(),
        };
        let r = report_for_validate_driver_profile(&summary);
        assert!(r.k_frust_invariant);
        assert!(r.notes.is_empty());
    }

    #[test]
    fn report_validate_driver_profile_regime_break() {
        let summary = DriverProfileSummary {
            intent: IntentId::I5,
            forward_residuals: vec![0.0],
            round_trip_residuals: vec![0.1],
            regime_agreements: vec![false],
            forward_mean: 0.0,
            round_trip_mean: 0.1,
            regime_agreement_rate: 0.5,
            per_intent: BTreeMap::new(),
        };
        let r = report_for_validate_driver_profile(&summary);
        assert!(!r.k_frust_invariant);
        assert!(r.notes.iter().any(|n| n.contains("0.5000 < 1.0")));
        assert!(r.notes.iter().any(|n| n.contains("(I5 round-trip)")));
    }

    // -------- Bitfield --------

    #[test]
    fn bitfield_all_pass() {
        let r = ValidationReport {
            asymptotic_closure_compliant: true,
            k_frust_invariant: true,
            round_trip_residual: None,
            notes: Vec::new(),
        };
        // bits 0 + 1, no bit 2 → 3
        assert_eq!(validation_flags_bitfield(&r), 3.0);
    }

    #[test]
    fn bitfield_asymptotic_failure() {
        let r = ValidationReport {
            asymptotic_closure_compliant: false,
            k_frust_invariant: true,
            round_trip_residual: None,
            notes: Vec::new(),
        };
        // bit 0 cleared → 2
        assert_eq!(validation_flags_bitfield(&r), 2.0);
    }

    #[test]
    fn bitfield_residual_present() {
        let r = ValidationReport {
            asymptotic_closure_compliant: true,
            k_frust_invariant: true,
            round_trip_residual: Some(0.001),
            notes: Vec::new(),
        };
        // bits 0+1+2 → 7
        assert_eq!(validation_flags_bitfield(&r), 7.0);
    }
}
