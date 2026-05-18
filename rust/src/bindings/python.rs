//! pyo3 extension module `_mpa_scale_solver_native`.
//!
//! Built by `maturin develop --release --features python`. Loaded by
//! the pure-Python `mpa_scale_solver/__init__.py` shim, which routes
//! the 9 wrapped variants + `validate_driver_profile` + `flow` through
//! this module when it imports successfully.
//!
//! Every binding follows the same dict-shape pattern:
//!   1. Accept inputs as `Bound<PyAny>` (Python dict / list / scalar).
//!   2. `pythonize::depythonize` into the Rust `types::` struct via serde.
//!   3. Call the wrapped variant.
//!   4. `pythonize::pythonize` the `OperationOutput<T>` back to a Python
//!      dict with keys `{value, validation, provenance}`.
//!
//! Closure-typed Rust parameters (`score_fn`, `forward_map`) are not
//! exposed — bindings pass `None`, matching the Python defaults.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList};
use pythonize::{depythonize, pythonize};

use crate::flow as flow_mod;
use crate::operations::{
    self as ops, Method, OperationError, ReferenceDatasetEntry, SweepTargets,
};
use crate::types as ty;

// ---------------------------------------------------------------------------
// Error mapping — `OperationError` / `FlowError` → Python `ValueError`
// ---------------------------------------------------------------------------

fn op_err(e: OperationError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn flow_err(e: flow_mod::FlowError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

// ---------------------------------------------------------------------------
// Enum parsing — Python passes strings; Rust enums via match.
// ---------------------------------------------------------------------------

fn parse_method(s: &str) -> PyResult<Method> {
    match s {
        "auto" => Ok(Method::Auto),
        "grid" => Ok(Method::Grid),
        "gradient" => Ok(Method::Gradient),
        _ => Err(PyValueError::new_err(format!(
            "unknown method '{s}'; expected 'auto', 'grid', or 'gradient'"
        ))),
    }
}

fn parse_intent(s: &str) -> PyResult<ty::IntentId> {
    match s {
        "I1" => Ok(ty::IntentId::I1),
        "I2" => Ok(ty::IntentId::I2),
        "I3" => Ok(ty::IntentId::I3),
        "I4" => Ok(ty::IntentId::I4),
        "I5" => Ok(ty::IntentId::I5),
        _ => Err(PyValueError::new_err(format!(
            "unknown intent '{s}'; expected one of I1, I2, I3, I4, I5"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Wrapped variant: regime_at_wrapped
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (canonical, tau_obs))]
fn regime_at_wrapped<'py>(
    py: Python<'py>,
    canonical: &Bound<'py, PyAny>,
    tau_obs: f64,
) -> PyResult<Bound<'py, PyAny>> {
    let c: ty::CanonicalState = depythonize(canonical)?;
    let out = ops::regime_at_wrapped(&c, tau_obs);
    Ok(pythonize(py, &out)?)
}

// ---------------------------------------------------------------------------
// Wrapped variant: gamut_classify_wrapped
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (canonical, tau_obs, gamut))]
fn gamut_classify_wrapped<'py>(
    py: Python<'py>,
    canonical: &Bound<'py, PyAny>,
    tau_obs: f64,
    gamut: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let c: ty::CanonicalState = depythonize(canonical)?;
    let g: ty::GamutSpec = depythonize(gamut)?;
    let out = ops::gamut_classify_wrapped(&c, tau_obs, &g);
    Ok(pythonize(py, &out)?)
}

// ---------------------------------------------------------------------------
// Wrapped variant: apply_translation_wrapped
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (
    canonical, field, tau_obs,
    domain_distance_threshold = ops::DEFAULT_DOMAIN_DISTANCE_THRESHOLD,
    tau_obs_weight = 1.0,
    sidecar = None,
))]
fn apply_translation_wrapped<'py>(
    py: Python<'py>,
    canonical: &Bound<'py, PyAny>,
    field: &Bound<'py, PyAny>,
    tau_obs: f64,
    domain_distance_threshold: f64,
    tau_obs_weight: f64,
    sidecar: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let c: ty::CanonicalState = depythonize(canonical)?;
    let f: ty::TranslationField = depythonize(field)?;
    let s: Option<ty::InverseLookupSidecar> = sidecar.map(depythonize).transpose()?;
    let out = ops::apply_translation_wrapped(
        &c,
        &f,
        tau_obs,
        domain_distance_threshold,
        tau_obs_weight,
        s.as_ref(),
    )
    .map_err(op_err)?;
    Ok(pythonize(py, &out)?)
}

// ---------------------------------------------------------------------------
// Wrapped variant: forward_sweep_invert_wrapped
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (
    target_substrate, field, tau_obs, canonical_grid,
    sidecar = None,
    compute_round_trip = true,
    method = "auto",
))]
fn forward_sweep_invert_wrapped<'py>(
    py: Python<'py>,
    target_substrate: &Bound<'py, PyAny>,
    field: &Bound<'py, PyAny>,
    tau_obs: f64,
    canonical_grid: &Bound<'py, PyAny>,
    sidecar: Option<&Bound<'py, PyAny>>,
    compute_round_trip: bool,
    method: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let t: ty::SubstrateState = depythonize(target_substrate)?;
    let f: ty::TranslationField = depythonize(field)?;
    let grid: Vec<[f64; 2]> = depythonize(canonical_grid)?;
    let s: Option<ty::InverseLookupSidecar> = sidecar.map(depythonize).transpose()?;
    let m = parse_method(method)?;
    let out = ops::forward_sweep_invert_wrapped(
        &t,
        &f,
        tau_obs,
        &grid,
        None,
        None,
        s.as_ref(),
        compute_round_trip,
        m,
    )
    .map_err(op_err)?;
    Ok(pythonize(py, &out)?)
}

// ---------------------------------------------------------------------------
// Wrapped variant: tau_obs_sweep_wrapped
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (
    target_substrates, field, tau_obs_grid, canonical_search_grid,
    sidecar = None,
))]
fn tau_obs_sweep_wrapped<'py>(
    py: Python<'py>,
    target_substrates: &Bound<'py, PyAny>,
    field: &Bound<'py, PyAny>,
    tau_obs_grid: Vec<f64>,
    canonical_search_grid: &Bound<'py, PyAny>,
    sidecar: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let f: ty::TranslationField = depythonize(field)?;
    let grid: Vec<[f64; 2]> = depythonize(canonical_search_grid)?;
    let s: Option<ty::InverseLookupSidecar> = sidecar.map(depythonize).transpose()?;

    // Python's signature is `Union[SubstrateState, list[SubstrateState]]`.
    // Detect by isinstance(list); fall through to single. Mirrors the
    // Python wrapped variant's branch.
    let targets_owned: Vec<ty::SubstrateState> = if target_substrates.is_instance_of::<PyList>() {
        depythonize(target_substrates)?
    } else {
        let single: ty::SubstrateState = depythonize(target_substrates)?;
        vec![single]
    };
    let sweep_targets = if target_substrates.is_instance_of::<PyList>() {
        SweepTargets::PerFrame(&targets_owned)
    } else {
        SweepTargets::Broadcast(&targets_owned[0])
    };

    let out = ops::tau_obs_sweep_wrapped(
        sweep_targets,
        &f,
        &tau_obs_grid,
        &grid,
        None,
        None,
        s.as_ref(),
    )
    .map_err(op_err)?;
    Ok(pythonize(py, &out)?)
}

// ---------------------------------------------------------------------------
// Wrapped variant: intent_map_wrapped
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (state, tau_obs, gamut, intent))]
fn intent_map_wrapped<'py>(
    py: Python<'py>,
    state: &Bound<'py, PyAny>,
    tau_obs: f64,
    gamut: &Bound<'py, PyAny>,
    intent: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let s: ty::CanonicalState = depythonize(state)?;
    let g: ty::GamutSpec = depythonize(gamut)?;
    let i = parse_intent(intent)?;
    let out = ops::intent_map_wrapped(&s, tau_obs, &g, i);
    Ok(pythonize(py, &out)?)
}

// ---------------------------------------------------------------------------
// Wrapped variant: intent_compose_wrapped
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (state, tau_obs, gamut, intents))]
fn intent_compose_wrapped<'py>(
    py: Python<'py>,
    state: &Bound<'py, PyAny>,
    tau_obs: f64,
    gamut: &Bound<'py, PyAny>,
    intents: Vec<String>,
) -> PyResult<Bound<'py, PyAny>> {
    let s: ty::CanonicalState = depythonize(state)?;
    let g: ty::GamutSpec = depythonize(gamut)?;
    let ids: Vec<ty::IntentId> = intents
        .iter()
        .map(|x| parse_intent(x.as_str()))
        .collect::<PyResult<_>>()?;
    let out = ops::intent_compose_wrapped(&s, tau_obs, &g, &ids).map_err(op_err)?;
    Ok(pythonize(py, &out)?)
}

// ---------------------------------------------------------------------------
// Wrapped variant: validate_driver_profile_wrapped
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (
    field, reference_dataset, canonical_search_grid, intent_id,
    gamut = None,
))]
fn validate_driver_profile_wrapped<'py>(
    py: Python<'py>,
    field: &Bound<'py, PyAny>,
    reference_dataset: &Bound<'py, PyAny>,
    canonical_search_grid: &Bound<'py, PyAny>,
    intent_id: &str,
    gamut: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let f: ty::TranslationField = depythonize(field)?;
    let dataset: Vec<ReferenceDatasetEntry> = depythonize(reference_dataset)?;
    let grid: Vec<[f64; 2]> = depythonize(canonical_search_grid)?;
    let i = parse_intent(intent_id)?;
    let g: Option<ty::GamutSpec> = gamut.map(depythonize).transpose()?;
    let out = ops::validate_driver_profile_wrapped(&f, &dataset, &grid, i, g.as_ref())
        .map_err(op_err)?;
    Ok(pythonize(py, &out)?)
}

// ---------------------------------------------------------------------------
// Raw: validate_driver_profile
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (
    field, reference_dataset, canonical_search_grid, intent_id,
    gamut = None,
))]
fn validate_driver_profile<'py>(
    py: Python<'py>,
    field: &Bound<'py, PyAny>,
    reference_dataset: &Bound<'py, PyAny>,
    canonical_search_grid: &Bound<'py, PyAny>,
    intent_id: &str,
    gamut: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let f: ty::TranslationField = depythonize(field)?;
    let dataset: Vec<ReferenceDatasetEntry> = depythonize(reference_dataset)?;
    let grid: Vec<[f64; 2]> = depythonize(canonical_search_grid)?;
    let i = parse_intent(intent_id)?;
    let g: Option<ty::GamutSpec> = gamut.map(depythonize).transpose()?;
    let out = ops::validate_driver_profile(&f, &dataset, &grid, i, g.as_ref())
        .map_err(op_err)?;
    Ok(pythonize(py, &out)?)
}

// ---------------------------------------------------------------------------
// Wrapped variant: forward_sweep_invert_posterior_wrapped
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (
    target, field, tau_obs,
    canonical_grid = None,
    noise_variance = 1.0,
    k_frust = false,
    top_k = 8,
))]
fn forward_sweep_invert_posterior_wrapped<'py>(
    py: Python<'py>,
    target: &Bound<'py, PyAny>,
    field: &Bound<'py, PyAny>,
    tau_obs: f64,
    canonical_grid: Option<&Bound<'py, PyAny>>,
    noise_variance: f64,
    k_frust: bool,
    top_k: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let t: ty::SubstrateState = depythonize(target)?;
    let f: ty::TranslationField = depythonize(field)?;
    let grid: Option<Vec<[f64; 2]>> = canonical_grid.map(depythonize).transpose()?;
    let out = ops::forward_sweep_invert_posterior_wrapped(
        &t,
        &f,
        tau_obs,
        grid.as_deref(),
        noise_variance,
        k_frust,
        None,
        top_k,
    )
    .map_err(op_err)?;
    Ok(pythonize(py, &out)?)
}

// ---------------------------------------------------------------------------
// flow
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (canonical_initial, nu, field))]
fn flow<'py>(
    py: Python<'py>,
    canonical_initial: &Bound<'py, PyAny>,
    nu: f64,
    field: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let c: ty::CanonicalState = depythonize(canonical_initial)?;
    let f: ty::TranslationField = depythonize(field)?;
    let out = flow_mod::flow(&c, nu, &f).map_err(flow_err)?;
    Ok(pythonize(py, &out)?)
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

/// Native extension module loaded by `mpa_scale_solver/__init__.py`.
///
/// Symbol contract: the names match the Python wrapped-variant entry
/// points one-for-one so the shim can `from _mpa_scale_solver_native
/// import regime_at_wrapped as regime_at_wrapped` without renaming.
#[pymodule]
fn _mpa_scale_solver_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(apply_translation_wrapped, m)?)?;
    m.add_function(wrap_pyfunction!(forward_sweep_invert_wrapped, m)?)?;
    m.add_function(wrap_pyfunction!(tau_obs_sweep_wrapped, m)?)?;
    m.add_function(wrap_pyfunction!(regime_at_wrapped, m)?)?;
    m.add_function(wrap_pyfunction!(gamut_classify_wrapped, m)?)?;
    m.add_function(wrap_pyfunction!(intent_map_wrapped, m)?)?;
    m.add_function(wrap_pyfunction!(intent_compose_wrapped, m)?)?;
    m.add_function(wrap_pyfunction!(validate_driver_profile_wrapped, m)?)?;
    m.add_function(wrap_pyfunction!(forward_sweep_invert_posterior_wrapped, m)?)?;
    m.add_function(wrap_pyfunction!(validate_driver_profile, m)?)?;
    m.add_function(wrap_pyfunction!(flow, m)?)?;
    Ok(())
}
