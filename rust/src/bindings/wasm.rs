//! wasm-bindgen module.
//!
//! Built by `wasm-pack build --release --target nodejs|web --features wasm`
//! into `rust/pkg/`. mpa-auditor consumes the output for browser-side
//! native solving.
//!
//! Mirror of `python.rs`: 9 wrapped variants + `validate_driver_profile`
//! + `flow`. Dict-shape contract via `serde-wasm-bindgen` — JS objects
//! in, JS objects out, the existing `Serialize + Deserialize` derives
//! on `types.rs` carry the surface.
//!
//! JS naming: snake_case to match Python. mpa-auditor's binding layer
//! can paper over the convention if needed.

use wasm_bindgen::prelude::*;
use serde_wasm_bindgen as swb;

use crate::flow as flow_mod;
use crate::operations::{
    self as ops, Method, OperationError, ReferenceDatasetEntry, SweepTargets,
};
use crate::types as ty;

// ---------------------------------------------------------------------------
// Error / enum helpers
// ---------------------------------------------------------------------------

fn js_err<E: std::fmt::Display>(e: E) -> JsError {
    JsError::new(&e.to_string())
}

fn op_err(e: OperationError) -> JsError {
    JsError::new(&e.to_string())
}

/// Serialize a serde value as plain JS objects (rather than `Map`
/// instances). Matches the dict-shape contract — consumers do
/// `out.value.observables.substrate_chit`, not `.get("substrate_chit")`.
/// The Python side's `pythonize` already produces plain Python dicts,
/// so this keeps both bindings symmetric.
fn to_js<T: serde::Serialize + ?Sized>(value: &T) -> Result<JsValue, JsError> {
    let serializer = swb::Serializer::new().serialize_maps_as_objects(true);
    value.serialize(&serializer).map_err(js_err)
}

fn parse_method(s: &str) -> Result<Method, JsError> {
    match s {
        "auto" => Ok(Method::Auto),
        "grid" => Ok(Method::Grid),
        "gradient" => Ok(Method::Gradient),
        _ => Err(JsError::new(&format!(
            "unknown method '{s}'; expected 'auto', 'grid', or 'gradient'"
        ))),
    }
}

fn parse_intent(s: &str) -> Result<ty::IntentId, JsError> {
    match s {
        "I1" => Ok(ty::IntentId::I1),
        "I2" => Ok(ty::IntentId::I2),
        "I3" => Ok(ty::IntentId::I3),
        "I4" => Ok(ty::IntentId::I4),
        "I5" => Ok(ty::IntentId::I5),
        _ => Err(JsError::new(&format!(
            "unknown intent '{s}'; expected one of I1, I2, I3, I4, I5"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Wrapped variants
// ---------------------------------------------------------------------------

#[wasm_bindgen]
pub fn regime_at_wrapped(canonical: JsValue, tau_obs: f64) -> Result<JsValue, JsError> {
    let c: ty::CanonicalState = swb::from_value(canonical).map_err(js_err)?;
    let out = ops::regime_at_wrapped(&c, tau_obs);
    to_js(&out)
}

#[wasm_bindgen]
pub fn gamut_classify_wrapped(
    canonical: JsValue,
    tau_obs: f64,
    gamut: JsValue,
) -> Result<JsValue, JsError> {
    let c: ty::CanonicalState = swb::from_value(canonical).map_err(js_err)?;
    let g: ty::GamutSpec = swb::from_value(gamut).map_err(js_err)?;
    let out = ops::gamut_classify_wrapped(&c, tau_obs, &g);
    to_js(&out)
}

#[wasm_bindgen]
pub fn apply_translation_wrapped(
    canonical: JsValue,
    field: JsValue,
    tau_obs: f64,
    domain_distance_threshold: Option<f64>,
    tau_obs_weight: Option<f64>,
    sidecar: JsValue,
) -> Result<JsValue, JsError> {
    let c: ty::CanonicalState = swb::from_value(canonical).map_err(js_err)?;
    let f: ty::TranslationField = swb::from_value(field).map_err(js_err)?;
    let s: Option<ty::InverseLookupSidecar> = if sidecar.is_null() || sidecar.is_undefined() {
        None
    } else {
        Some(swb::from_value(sidecar).map_err(js_err)?)
    };
    let dth = domain_distance_threshold.unwrap_or(ops::DEFAULT_DOMAIN_DISTANCE_THRESHOLD);
    let tw = tau_obs_weight.unwrap_or(1.0);
    let out = ops::apply_translation_wrapped(&c, &f, tau_obs, dth, tw, s.as_ref())
        .map_err(op_err)?;
    to_js(&out)
}

#[wasm_bindgen]
pub fn forward_sweep_invert_wrapped(
    target_substrate: JsValue,
    field: JsValue,
    tau_obs: f64,
    canonical_grid: JsValue,
    sidecar: JsValue,
    compute_round_trip: Option<bool>,
    method: Option<String>,
) -> Result<JsValue, JsError> {
    let t: ty::SubstrateState = swb::from_value(target_substrate).map_err(js_err)?;
    let f: ty::TranslationField = swb::from_value(field).map_err(js_err)?;
    let grid: Vec<[f64; 2]> = swb::from_value(canonical_grid).map_err(js_err)?;
    let s: Option<ty::InverseLookupSidecar> = if sidecar.is_null() || sidecar.is_undefined() {
        None
    } else {
        Some(swb::from_value(sidecar).map_err(js_err)?)
    };
    let rt = compute_round_trip.unwrap_or(true);
    let m = parse_method(method.as_deref().unwrap_or("auto"))?;
    let out = ops::forward_sweep_invert_wrapped(
        &t,
        &f,
        tau_obs,
        &grid,
        None,
        None,
        s.as_ref(),
        rt,
        m,
    )
    .map_err(op_err)?;
    to_js(&out)
}

#[wasm_bindgen]
pub fn tau_obs_sweep_wrapped(
    target_substrates: JsValue,
    field: JsValue,
    tau_obs_grid: JsValue,
    canonical_search_grid: JsValue,
    sidecar: JsValue,
) -> Result<JsValue, JsError> {
    let f: ty::TranslationField = swb::from_value(field).map_err(js_err)?;
    let grid: Vec<[f64; 2]> = swb::from_value(canonical_search_grid).map_err(js_err)?;
    let tau_grid: Vec<f64> = swb::from_value(tau_obs_grid).map_err(js_err)?;
    let s: Option<ty::InverseLookupSidecar> = if sidecar.is_null() || sidecar.is_undefined() {
        None
    } else {
        Some(swb::from_value(sidecar).map_err(js_err)?)
    };

    // JS `Array.isArray` → PerFrame; otherwise Broadcast. Detect via
    // js_sys::Array::is_array on the JsValue ref.
    let is_array = target_substrates.is_array();
    let targets_owned: Vec<ty::SubstrateState> = if is_array {
        swb::from_value(target_substrates).map_err(js_err)?
    } else {
        let single: ty::SubstrateState = swb::from_value(target_substrates).map_err(js_err)?;
        vec![single]
    };
    let sweep_targets = if is_array {
        SweepTargets::PerFrame(&targets_owned)
    } else {
        SweepTargets::Broadcast(&targets_owned[0])
    };

    let out = ops::tau_obs_sweep_wrapped(
        sweep_targets,
        &f,
        &tau_grid,
        &grid,
        None,
        None,
        s.as_ref(),
    )
    .map_err(op_err)?;
    to_js(&out)
}

#[wasm_bindgen]
pub fn intent_map_wrapped(
    state: JsValue,
    tau_obs: f64,
    gamut: JsValue,
    intent: &str,
) -> Result<JsValue, JsError> {
    let s: ty::CanonicalState = swb::from_value(state).map_err(js_err)?;
    let g: ty::GamutSpec = swb::from_value(gamut).map_err(js_err)?;
    let i = parse_intent(intent)?;
    let out = ops::intent_map_wrapped(&s, tau_obs, &g, i);
    to_js(&out)
}

#[wasm_bindgen]
pub fn intent_compose_wrapped(
    state: JsValue,
    tau_obs: f64,
    gamut: JsValue,
    intents: Vec<String>,
) -> Result<JsValue, JsError> {
    let s: ty::CanonicalState = swb::from_value(state).map_err(js_err)?;
    let g: ty::GamutSpec = swb::from_value(gamut).map_err(js_err)?;
    let ids: Vec<ty::IntentId> = intents
        .iter()
        .map(|x| parse_intent(x.as_str()))
        .collect::<Result<_, _>>()?;
    let out = ops::intent_compose_wrapped(&s, tau_obs, &g, &ids).map_err(op_err)?;
    to_js(&out)
}

#[wasm_bindgen]
pub fn validate_driver_profile_wrapped(
    field: JsValue,
    reference_dataset: JsValue,
    canonical_search_grid: JsValue,
    intent_id: &str,
    gamut: JsValue,
) -> Result<JsValue, JsError> {
    let f: ty::TranslationField = swb::from_value(field).map_err(js_err)?;
    let dataset: Vec<ReferenceDatasetEntry> = swb::from_value(reference_dataset).map_err(js_err)?;
    let grid: Vec<[f64; 2]> = swb::from_value(canonical_search_grid).map_err(js_err)?;
    let i = parse_intent(intent_id)?;
    let g: Option<ty::GamutSpec> = if gamut.is_null() || gamut.is_undefined() {
        None
    } else {
        Some(swb::from_value(gamut).map_err(js_err)?)
    };
    let out = ops::validate_driver_profile_wrapped(&f, &dataset, &grid, i, g.as_ref())
        .map_err(op_err)?;
    to_js(&out)
}

#[wasm_bindgen]
pub fn validate_driver_profile(
    field: JsValue,
    reference_dataset: JsValue,
    canonical_search_grid: JsValue,
    intent_id: &str,
    gamut: JsValue,
) -> Result<JsValue, JsError> {
    let f: ty::TranslationField = swb::from_value(field).map_err(js_err)?;
    let dataset: Vec<ReferenceDatasetEntry> = swb::from_value(reference_dataset).map_err(js_err)?;
    let grid: Vec<[f64; 2]> = swb::from_value(canonical_search_grid).map_err(js_err)?;
    let i = parse_intent(intent_id)?;
    let g: Option<ty::GamutSpec> = if gamut.is_null() || gamut.is_undefined() {
        None
    } else {
        Some(swb::from_value(gamut).map_err(js_err)?)
    };
    let out = ops::validate_driver_profile(&f, &dataset, &grid, i, g.as_ref())
        .map_err(op_err)?;
    to_js(&out)
}

#[wasm_bindgen]
pub fn forward_sweep_invert_posterior_wrapped(
    target: JsValue,
    field: JsValue,
    tau_obs: f64,
    canonical_grid: JsValue,
    noise_variance: Option<f64>,
    k_frust: Option<bool>,
    top_k: Option<usize>,
) -> Result<JsValue, JsError> {
    let t: ty::SubstrateState = swb::from_value(target).map_err(js_err)?;
    let f: ty::TranslationField = swb::from_value(field).map_err(js_err)?;
    let grid: Option<Vec<[f64; 2]>> = if canonical_grid.is_null() || canonical_grid.is_undefined() {
        None
    } else {
        Some(swb::from_value(canonical_grid).map_err(js_err)?)
    };
    let nv = noise_variance.unwrap_or(1.0);
    let kf = k_frust.unwrap_or(false);
    let tk = top_k.unwrap_or(8);
    let out = ops::forward_sweep_invert_posterior_wrapped(
        &t,
        &f,
        tau_obs,
        grid.as_deref(),
        nv,
        kf,
        None,
        tk,
    )
    .map_err(op_err)?;
    to_js(&out)
}

#[wasm_bindgen]
pub fn flow(
    canonical_initial: JsValue,
    nu: f64,
    field: JsValue,
) -> Result<JsValue, JsError> {
    let c: ty::CanonicalState = swb::from_value(canonical_initial).map_err(js_err)?;
    let f: ty::TranslationField = swb::from_value(field).map_err(js_err)?;
    let out = flow_mod::flow(&c, nu, &f).map_err(js_err)?;
    to_js(&out)
}
