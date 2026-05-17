//! Smoke test for `mpa_scale_solver::types` — port of `mpa_scale_solver/types.py`.
//!
//! Construct an instance of every public type, round-trip via serde_json,
//! assert equality. Catches derive misconfiguration (missing fields, tag
//! mismatches on `TranslationField`, custom serde defaults).
//!
//! This is NOT a cross-language bit-identity test. Cross-language JSON
//! parity lands when `sidecar.rs` / `mcp_server.rs` port — those modules
//! own the Python-side JSON producers. See BLOCK_IN.md §v6 Open / watch.

use std::collections::BTreeMap;

use mpa_scale_solver::math::{Activation, MlpLayer};
use mpa_scale_solver::types::{
    CanonicalPoint, CanonicalState, Direction, DispatchPath, DisplayBand, GamutSpec, Gt,
    InverseLookupSidecar, LearnedField, LookupTableField, OperatingPoint, OperationOutput,
    Posterior, Provenance, RegimeLabel, RegimeReading, ScalingRule, SidecarKey, SubstrateState,
    TangentFlowField, TranslationField, TranslationRule, ValidationReport,
};

/// Round-trip helper — serialize, deserialize, assert equal.
fn roundtrip<T>(value: &T)
where
    T: serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
{
    let json = serde_json::to_string(value).expect("serialize");
    let parsed: T = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(value, &parsed, "round-trip mismatch — json was: {json}");
}

fn sample_translation_rule() -> TranslationRule {
    TranslationRule {
        operating_point: OperatingPoint {
            label: "origin".into(),
            gt: Gt::S,
            axes: BTreeMap::from([("T".to_string(), serde_json::json!(1.0))]),
        },
        xdot_choice: "default".into(),
        canonical: CanonicalPoint {
            chit: 0.5,
            gamma_AB: 1.25,
            k_frust: false,
            method: "fixture".into(),
            extras: BTreeMap::new(),
        },
    }
}

#[test]
fn canonical_state_roundtrip() {
    roundtrip(&CanonicalState {
        chit: 0.7,
        gamma_AB: 1.1,
        k_frust: true,
    });
}

#[test]
fn substrate_state_roundtrip() {
    roundtrip(&SubstrateState {
        tau_obs: 2.5,
        label: Some("cell-0".into()),
        axes: BTreeMap::from([("T".to_string(), serde_json::json!(300.0))]),
        observables: BTreeMap::from([
            ("substrate_chit".to_string(), 0.42),
            ("substrate_gamma_AB".to_string(), 0.91),
        ]),
    });
}

#[test]
fn lookup_table_field_roundtrip() {
    roundtrip(&LookupTableField {
        direction: Direction::Forward,
        rule: vec![sample_translation_rule()],
        description: Some("lookup fixture".into()),
    });
}

#[test]
fn tangent_flow_field_roundtrip() {
    roundtrip(&TangentFlowField {
        direction: Direction::Forward,
        rule_at_origin: sample_translation_rule(),
        scaling: ScalingRule {
            tau_obs_ref: 1.0,
            delta_gamma: -0.5,
            delta_chit: 0.25,
            refinement: Some(BTreeMap::from([(
                "beta_mem".to_string(),
                serde_json::json!(0.8),
            )])),
        },
        description: None,
    });
}

#[test]
fn learned_field_roundtrip() {
    roundtrip(&LearnedField {
        direction: Direction::Forward,
        rule_at_origin: sample_translation_rule(),
        weights: vec![
            MlpLayer {
                w: vec![vec![0.1, 0.2, 0.3], vec![-0.1, 0.0, 0.4]],
                b: vec![0.0, 0.05],
            },
            MlpLayer {
                w: vec![vec![0.7, -0.2], vec![0.3, 0.1]],
                b: vec![0.0, 0.0],
            },
        ],
        architecture: vec![3, 2, 2],
        activation: Activation::Tanh,
        tau_obs_ref: 1.0,
        description: None,
    });
}

#[test]
fn translation_field_enum_tag_lookup_table() {
    let field = TranslationField::LookupTable(LookupTableField {
        direction: Direction::Forward,
        rule: vec![sample_translation_rule()],
        description: None,
    });
    let json = serde_json::to_string(&field).unwrap();
    // The serde tag is `shape`; variant rename is snake_case.
    assert!(
        json.contains(r#""shape":"lookup_table""#),
        "expected shape tag in JSON, got: {json}"
    );
    roundtrip(&field);
}

#[test]
fn translation_field_enum_tag_tangent_flow() {
    let field = TranslationField::TangentFlow(TangentFlowField {
        direction: Direction::Forward,
        rule_at_origin: sample_translation_rule(),
        scaling: ScalingRule {
            tau_obs_ref: 1.0,
            delta_gamma: 0.0,
            delta_chit: 0.0,
            refinement: None,
        },
        description: None,
    });
    let json = serde_json::to_string(&field).unwrap();
    assert!(
        json.contains(r#""shape":"tangent_flow""#),
        "expected shape tag in JSON, got: {json}"
    );
    roundtrip(&field);
}

#[test]
fn translation_field_enum_tag_learned() {
    let field = TranslationField::Learned(LearnedField {
        direction: Direction::Forward,
        rule_at_origin: sample_translation_rule(),
        weights: vec![MlpLayer {
            w: vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]],
            b: vec![0.0, 0.0],
        }],
        architecture: vec![3, 2],
        activation: Activation::Relu,
        tau_obs_ref: 1.0,
        description: None,
    });
    let json = serde_json::to_string(&field).unwrap();
    assert!(
        json.contains(r#""shape":"learned""#),
        "expected shape tag in JSON, got: {json}"
    );
    roundtrip(&field);
}

#[test]
fn gamut_spec_roundtrip() {
    roundtrip(&GamutSpec {
        chit_range: (-1.0, 1.0),
        gamma_AB_range: (0.1, 2.0),
        tau_obs_range: Some((0.5, 10.0)),
        out_of_scope_residual_threshold: 0.05,
    });
}

#[test]
fn regime_reading_roundtrip() {
    for label in [
        RegimeLabel::DeepC,
        RegimeLabel::CNearS,
        RegimeLabel::SCritical,
        RegimeLabel::RNearS,
        RegimeLabel::DeepR,
    ] {
        roundtrip(&RegimeReading {
            regime: label,
            k_frust: false,
        });
    }
}

#[test]
fn display_band_roundtrip() {
    for band in [DisplayBand::C, DisplayBand::S, DisplayBand::R] {
        roundtrip(&band);
    }
}

#[test]
fn provenance_roundtrip() {
    for path in [
        DispatchPath::TableHit,
        DispatchPath::ComputeFallback,
        DispatchPath::DirectCompute,
    ] {
        roundtrip(&Provenance {
            solver_version: "0.1.0".into(),
            operation: "forward_sweep_invert".into(),
            timestamp_ns: 1_700_000_000_000_000_000,
            dispatch_path: path,
            table_version: Some("v1".into()),
            notes: vec!["one".into(), "two".into()],
        });
    }
}

#[test]
fn validation_report_defaults() {
    let report = ValidationReport::default();
    assert!(report.asymptotic_closure_compliant);
    assert!(report.k_frust_invariant);
    assert!(report.round_trip_residual.is_none());
    assert!(report.notes.is_empty());
    roundtrip(&report);
}

#[test]
fn operation_output_generic_canonical() {
    let out = OperationOutput {
        value: CanonicalState {
            chit: 0.3,
            gamma_AB: 0.9,
            k_frust: false,
        },
        validation: ValidationReport::default(),
        provenance: Provenance {
            solver_version: "0.1.0".into(),
            operation: "apply_translation".into(),
            timestamp_ns: 1,
            dispatch_path: DispatchPath::DirectCompute,
            table_version: None,
            notes: vec![],
        },
    };
    roundtrip(&out);
}

#[test]
fn operation_output_generic_f64() {
    let out = OperationOutput {
        value: 0.42_f64,
        validation: ValidationReport::default(),
        provenance: Provenance {
            solver_version: "0.1.0".into(),
            operation: "regime_at".into(),
            timestamp_ns: 1,
            dispatch_path: DispatchPath::DirectCompute,
            table_version: None,
            notes: vec![],
        },
    };
    roundtrip(&out);
}

#[test]
fn posterior_roundtrip() {
    roundtrip(&Posterior {
        mean: CanonicalState {
            chit: 0.0,
            gamma_AB: 1.0,
            k_frust: false,
        },
        covariance: [[0.1, 0.01], [0.01, 0.2]],
        noise_variance: 0.5,
        log_evidence: Some(-12.5),
        modes: vec![
            CanonicalState {
                chit: 0.0,
                gamma_AB: 1.0,
                k_frust: false,
            },
            CanonicalState {
                chit: 0.5,
                gamma_AB: 0.8,
                k_frust: false,
            },
        ],
        notes: vec!["secondary mode within 1σ".into()],
    });
}

#[test]
fn sidecar_key_round_trip_bits() {
    let key = SidecarKey::from_floats(0.1, 0.2, 1.5);
    let (chit, gamma, tau) = key.as_floats();
    assert_eq!(chit, 0.1);
    assert_eq!(gamma, 0.2);
    assert_eq!(tau, 1.5);
    roundtrip(&key);
}

#[test]
fn inverse_lookup_sidecar_roundtrip() {
    let k = SidecarKey::from_floats(0.1, 0.2, 1.0);
    let canonical = CanonicalState {
        chit: 0.1,
        gamma_AB: 0.2,
        k_frust: false,
    };
    let substrate = SubstrateState {
        tau_obs: 1.0,
        label: Some("cell-0".into()),
        axes: BTreeMap::new(),
        observables: BTreeMap::from([
            ("substrate_chit".to_string(), 0.1),
            ("substrate_gamma_AB".to_string(), 0.2),
        ]),
    };
    let sidecar = InverseLookupSidecar {
        version: "v1".into(),
        driver_profile_id: "banach.reference".into(),
        driver_profile_version: "v0".into(),
        tau_obs_grid: vec![1.0],
        substrate_grid: vec![substrate.clone()],
        canonical_grid: vec![canonical.clone()],
        forward_lookup: BTreeMap::from([(k, substrate)]),
        inverse_lookup: BTreeMap::from([(k, canonical)]),
        ambiguity_regions: vec![],
    };
    roundtrip(&sidecar);
}
