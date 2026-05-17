//! Runtime dataclasses — port of `mpa_scale_solver/types.py`.
//!
//! Plain value types, no methods beyond constructors. Mirrors the Python
//! `@dataclass(frozen=True)` shapes. `_repr_html_` and `__repr__` overrides
//! in the Python are display-only (Jupyter / REPL) and do not port.
//!
//! Naming divergence from Python (intentional, documented in CLAUDE.md):
//! Python's `TranslationField` is the lookup-table struct, with
//! `LookupTableField` as an alias and `AnyTranslationField` as the
//! `Union[lookup_table, tangent_flow, learned]` accepted by operations.
//! In Rust the struct is named `LookupTableField` (matching its shape)
//! and `TranslationField` is the tagged enum over the three shapes. The
//! variant tag is the Python `shape` field discriminator.
//!
//! `Activation` and `MlpLayer` are re-exported from `math` so that
//! `LearnedField.weights: Vec<MlpLayer>` passes directly into
//! `math::learned_field_substrate` with no conversion shim.
//!
//! Field-name discipline: schema field names from the Python types.py
//! (e.g. `gamma_AB`) are preserved verbatim per the "Python is the
//! pseudo-code spec" rule. The non-snake-case allow below is scoped to
//! this module.

#![allow(non_snake_case)]

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

pub use crate::math::{Activation, MlpLayer};

// ---------------------------------------------------------------------------
// Enums (mirror the Python `Literal[...]` type pins)
// ---------------------------------------------------------------------------

/// Translation-field direction. Forward-only per RFC-S §Q13 (backward
/// map is structurally ill-posed).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Direction {
    Forward,
}

/// Operating-point graph type — cross-substrate vertex regime token
/// (`c`, `s`, `r`, `k`). Mirrors `Literal["c", "s", "r", "k"]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Gt {
    C,
    S,
    R,
    K,
}

/// Five-bucket canonical regime label (matches the auditor's gfdr-model.js
/// five-bucket classifier). `DisplayBand` collapses these to three.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegimeLabel {
    DeepC,
    CNearS,
    SCritical,
    RNearS,
    DeepR,
}

/// Three-bucket display band — `regime_display_band` projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DisplayBand {
    C,
    S,
    R,
}

/// Which dispatch path the wrapped operation took. Recorded on every
/// `Provenance` so consumers can distinguish table-hits from compute
/// fallbacks without re-running the op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DispatchPath {
    TableHit,
    ComputeFallback,
    DirectCompute,
}

/// One of the five RFC-S §3 mapping intents accepted by `intent_map`
/// and `intent_compose`. Typed rather than stringly — Python's
/// `intent_id: str` produces an "unknown intent" ValueError at runtime;
/// the Rust port pushes that check into the type system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IntentId {
    I1,
    I2,
    I3,
    I4,
    I5,
}

/// Capacity bucket used by I3 (capacity-preserving intent). `Deep` is
/// `|chit| >= 0.7` — the framework's fixed-point-stability boundary
/// (`gfdr_model::vertex_regime`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CapacityClass {
    Deep,
    Shallow,
}

// ---------------------------------------------------------------------------
// Runtime working states (§A.3)
// ---------------------------------------------------------------------------

/// Canonical-frame state at the call-site's `tau_obs`. tau_obs is NOT
/// stored on the state — every operation that needs it takes it as an
/// explicit argument. Keeps the state pair substrate-neutral.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CanonicalState {
    pub chit: f64,
    pub gamma_AB: f64,
    #[serde(default)]
    pub k_frust: bool,
}

/// Substrate-native observation at one tau_obs frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SubstrateState {
    pub tau_obs: f64,
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub axes: BTreeMap<String, serde_json::Value>,
    #[serde(default)]
    pub observables: BTreeMap<String, f64>,
}

// ---------------------------------------------------------------------------
// Schema dataclasses (driver-profile.v2.0)
// ---------------------------------------------------------------------------

/// Canonical-coordinate target a TranslationRule projects to.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CanonicalPoint {
    pub chit: f64,
    pub gamma_AB: f64,
    pub k_frust: bool,
    pub method: String,
    #[serde(default)]
    pub extras: BTreeMap<String, serde_json::Value>,
}

/// A cell in the substrate's operating envelope.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OperatingPoint {
    pub label: String,
    pub gt: Gt,
    #[serde(default)]
    pub axes: BTreeMap<String, serde_json::Value>,
}

/// One (operating_point, xdot_choice, canonical) triple — a single rule
/// in a translation field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranslationRule {
    pub operating_point: OperatingPoint,
    pub xdot_choice: String,
    pub canonical: CanonicalPoint,
}

/// Lookup-table translation field — the v0 production shape. Python's
/// `TranslationField` / `LookupTableField` alias.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LookupTableField {
    pub direction: Direction,
    pub rule: Vec<TranslationRule>,
    #[serde(default)]
    pub description: Option<String>,
}

/// Banach-canonical leading-order tangent-flow rule (RFC-S Appendix B
/// item 1). Substrate-conditional refinements ride in `refinement`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScalingRule {
    pub tau_obs_ref: f64,
    #[serde(default)]
    pub delta_gamma: f64,
    #[serde(default)]
    pub delta_chit: f64,
    #[serde(default)]
    pub refinement: Option<BTreeMap<String, serde_json::Value>>,
}

/// Tangent-flow translation field — closed-form sibling of
/// `LookupTableField`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TangentFlowField {
    pub direction: Direction,
    pub rule_at_origin: TranslationRule,
    pub scaling: ScalingRule,
    #[serde(default)]
    pub description: Option<String>,
}

/// Learned translation field — small MLP forward map (v3 BLOCK_IN §v3).
/// Third translation-field shape alongside lookup_table (v0) and
/// tangent_flow (v1). Input to the MLP is
/// `(chit, gamma_AB, log(tau_obs / tau_obs_ref))`; output is
/// `(substrate_chit, substrate_gamma_AB)`.
///
/// Training is curator-side (mpa-conform); the solver only evaluates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LearnedField {
    pub direction: Direction,
    pub rule_at_origin: TranslationRule,
    pub weights: Vec<MlpLayer>,
    pub architecture: Vec<usize>,
    #[serde(default = "default_activation")]
    pub activation: Activation,
    #[serde(default = "default_tau_obs_ref")]
    pub tau_obs_ref: f64,
    #[serde(default)]
    pub description: Option<String>,
}

fn default_activation() -> Activation {
    Activation::Tanh
}

fn default_tau_obs_ref() -> f64 {
    1.0
}

/// Tagged-enum dispatch over the three translation-field shapes. Python's
/// `AnyTranslationField = Union[TranslationField, TangentFlowField, LearnedField]`.
/// The serde tag is `shape` (matches Python's per-struct `shape` field).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "shape", rename_all = "snake_case")]
pub enum TranslationField {
    LookupTable(LookupTableField),
    TangentFlow(TangentFlowField),
    Learned(LearnedField),
}

// ---------------------------------------------------------------------------
// Gamut / regime
// ---------------------------------------------------------------------------

/// Substrate gamut — image of the RG trajectory in canonical space.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GamutSpec {
    pub chit_range: (f64, f64),
    pub gamma_AB_range: (f64, f64),
    #[serde(default)]
    pub tau_obs_range: Option<(f64, f64)>,
    #[serde(default = "default_out_of_scope_threshold")]
    pub out_of_scope_residual_threshold: f64,
}

fn default_out_of_scope_threshold() -> f64 {
    0.05
}

/// Five-bucket vertex regime at a tau_obs frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegimeReading {
    pub regime: RegimeLabel,
    #[serde(default)]
    pub k_frust: bool,
}

// ---------------------------------------------------------------------------
// Provenance + validation (handoff §C.5 / §C.6)
// ---------------------------------------------------------------------------

/// Per-call provenance trail. Fields are primitive so the record
/// serializes cleanly into mpa-conform's bundle audit record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    pub solver_version: String,
    pub operation: String,
    pub timestamp_ns: i64,
    pub dispatch_path: DispatchPath,
    #[serde(default)]
    pub table_version: Option<String>,
    #[serde(default)]
    pub notes: Vec<String>,
}

/// Per-call validation flags — reported, not raised. Default-True
/// convention: an op that has no constraint on a flag still reports True
/// (vacuously satisfied), distinct from a flag that fired.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidationReport {
    #[serde(default = "default_true")]
    pub asymptotic_closure_compliant: bool,
    #[serde(default = "default_true")]
    pub k_frust_invariant: bool,
    #[serde(default)]
    pub round_trip_residual: Option<f64>,
    #[serde(default)]
    pub notes: Vec<String>,
}

fn default_true() -> bool {
    true
}

impl Default for ValidationReport {
    fn default() -> Self {
        Self {
            asymptotic_closure_compliant: true,
            k_frust_invariant: true,
            round_trip_residual: None,
            notes: Vec::new(),
        }
    }
}

/// Wrapped operation result. Returned by every `*_wrapped` operation;
/// unwrapped operations keep their raw return types for back-compat.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OperationOutput<T> {
    pub value: T,
    pub validation: ValidationReport,
    pub provenance: Provenance,
}

// ---------------------------------------------------------------------------
// v2.1: Laplace-approximation posterior
// ---------------------------------------------------------------------------

/// Laplace-approximation posterior over canonical states. Captures the
/// MAP point estimate (`mean`) plus a Gaussian uncertainty estimate
/// (`covariance`) around it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Posterior {
    pub mean: CanonicalState,
    pub covariance: [[f64; 2]; 2],
    #[serde(default = "default_noise_variance")]
    pub noise_variance: f64,
    #[serde(default)]
    pub log_evidence: Option<f64>,
    #[serde(default)]
    pub modes: Vec<CanonicalState>,
    #[serde(default)]
    pub notes: Vec<String>,
}

fn default_noise_variance() -> f64 {
    1.0
}

// ---------------------------------------------------------------------------
// v1: inverse-lookup-table sidecar
// ---------------------------------------------------------------------------

/// Three-float key for sidecar lookup maps. Python keys these maps on
/// `(chit, gamma_AB, tau_obs)` tuples after rounding via `sidecar.round_key`
/// to a fixed decimal precision. Rust stores the rounded floats as their
/// raw bit patterns so the BTreeMap can order/compare them.
///
/// The producer-side convention (matching `sidecar.DEFAULT_ROUNDING_DECIMALS`)
/// is to round to 6 decimals before constructing the key. Both producer
/// and consumer must agree.
///
/// Wire format: ':'-joined raw `u64` bit-patterns (e.g. `"4576918229304087675:..."`).
/// Chosen because JSON requires string keys for objects; the raw-bits form
/// is lossless and exactly recovers the rounded floats. Sidecar.rs is
/// free to replace this with a Python-parity scheme (e.g. parallel
/// keys/values arrays) when cross-language sidecar I/O lands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SidecarKey(pub [u64; 3]);

impl SidecarKey {
    pub fn from_floats(chit: f64, gamma_ab: f64, tau_obs: f64) -> Self {
        SidecarKey([chit.to_bits(), gamma_ab.to_bits(), tau_obs.to_bits()])
    }

    pub fn as_floats(&self) -> (f64, f64, f64) {
        (
            f64::from_bits(self.0[0]),
            f64::from_bits(self.0[1]),
            f64::from_bits(self.0[2]),
        )
    }
}

impl Serialize for SidecarKey {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let s = format!("{}:{}:{}", self.0[0], self.0[1], self.0[2]);
        serializer.serialize_str(&s)
    }
}

impl<'de> Deserialize<'de> for SidecarKey {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        let parts: Vec<&str> = s.split(':').collect();
        if parts.len() != 3 {
            return Err(serde::de::Error::custom(format!(
                "SidecarKey: expected 3 ':'-separated u64 parts, got {}",
                parts.len()
            )));
        }
        let a = parts[0].parse::<u64>().map_err(serde::de::Error::custom)?;
        let b = parts[1].parse::<u64>().map_err(serde::de::Error::custom)?;
        let c = parts[2].parse::<u64>().map_err(serde::de::Error::custom)?;
        Ok(SidecarKey([a, b, c]))
    }
}

// ---------------------------------------------------------------------------
// v2.3: intent algebra — sacrifice records
// ---------------------------------------------------------------------------

/// Per-state sacrifice record emitted by `intent_map`. Three fields
/// are common to every intent (the outer struct); intent-specific
/// diagnostics ride in the `diagnostics` enum, flattened so the JSON
/// wire format is a single flat dict matching Python's `sac` shape.
///
/// `intent` and `preserved_invariant` are derived (not stored) — both
/// are statically determined by the handler that built the record.
/// Use the `intent()` and `preserved_invariant()` methods to read them.
/// If cross-language JSON parity ever needs the `preserved_invariant`
/// string in the serialized form (it lives in Python's sac dict), wire
/// a custom serializer at that boundary; the in-memory shape stays
/// minimal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SacrificeRecord {
    pub invariant_preserved: bool,
    pub delta_chit: f64,
    pub delta_gamma_AB: f64,
    #[serde(flatten)]
    pub diagnostics: IntentDiagnostics,
}

impl SacrificeRecord {
    /// Which intent emitted this record (statically determined by the
    /// `diagnostics` variant).
    pub fn intent(&self) -> IntentId {
        match self.diagnostics {
            IntentDiagnostics::I1 { .. } => IntentId::I1,
            IntentDiagnostics::I2 { .. } => IntentId::I2,
            IntentDiagnostics::I3 { .. } => IntentId::I3,
            IntentDiagnostics::I4 { .. } => IntentId::I4,
            IntentDiagnostics::I5 { .. } => IntentId::I5,
        }
    }

    /// Per-intent preserved-invariant string. Matches Python's
    /// `sac["preserved_invariant"]` value verbatim (including the
    /// Unicode `∧` in I1 / I3).
    pub fn preserved_invariant(&self) -> &'static str {
        match self.diagnostics {
            IntentDiagnostics::I1 { .. } => "regime ∧ sign(gamma_AB) ∧ k_frust",
            IntentDiagnostics::I2 { .. } => "exact_drive_parameters",
            IntentDiagnostics::I3 { .. } => "capacity_class ∧ k_frust",
            IntentDiagnostics::I4 { .. } => "sign(gamma_AB)",
            IntentDiagnostics::I5 { .. } => "regime_label",
        }
    }
}

/// Intent-specific diagnostic fields. Tagged by `intent` so the JSON
/// is a single flat dict (combined with `SacrificeRecord`'s
/// `#[serde(flatten)]`). For I5 the v0/v1 keys (`regime_preserved`,
/// `original_regime`, `mapped_regime`) are preserved verbatim — the
/// uniform v2.3 keys (`preserved_invariant`, `invariant_preserved`) are
/// the outer struct + the derived method.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "intent")]
pub enum IntentDiagnostics {
    I1 {
        regime_preserved: bool,
        gamma_AB_sign_preserved: bool,
        k_frust_preserved: bool,
        original_regime: RegimeLabel,
        mapped_regime: RegimeLabel,
        original_gamma_AB_sign: i32,
        mapped_gamma_AB_sign: i32,
    },
    I2 {
        out_of_gamut_rejected: bool,
        out_of_gamut_axes: Vec<String>,
    },
    I3 {
        capacity_class: CapacityClass,
        mapped_capacity_class: CapacityClass,
        k_frust: bool,
        k_frust_preserved: bool,
    },
    I4 {
        original_gamma_AB_sign: i32,
        mapped_gamma_AB_sign: i32,
    },
    I5 {
        regime_preserved: bool,
        original_regime: RegimeLabel,
        mapped_regime: RegimeLabel,
    },
}

/// Curator-precomputed inverse-lookup table. Sidecar production lives in
/// mpa-conform's curator path; this crate consumes via
/// `operations::forward_sweep_invert`'s optional `sidecar` argument.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InverseLookupSidecar {
    pub version: String,
    pub driver_profile_id: String,
    pub driver_profile_version: String,
    pub tau_obs_grid: Vec<f64>,
    pub substrate_grid: Vec<SubstrateState>,
    pub canonical_grid: Vec<CanonicalState>,
    pub forward_lookup: BTreeMap<SidecarKey, SubstrateState>,
    pub inverse_lookup: BTreeMap<SidecarKey, CanonicalState>,
    #[serde(default)]
    pub ambiguity_regions: Vec<BTreeMap<String, serde_json::Value>>,
}
