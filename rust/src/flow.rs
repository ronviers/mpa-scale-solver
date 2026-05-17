//! Continuous-form flow `C^nu = exp(nu * ln C)` — port of
//! `mpa_scale_solver/flow.py`.
#![allow(non_snake_case)]


//!
//! Grounded by v9_receipts §RG closure in Markovian scope (`beta_mem = 1`).
//! Dispatch on `TranslationField` variant and the `refinement` map:
//!
//!  - `TangentFlow` + `beta_mem < 1`: v2.4 non-Markovian Caputo path.
//!    Curator-supplied Prony approximation `Σ_k a_k exp(-b_k x)` rides
//!    the refinement dict as `prony_terms = [[amp, decay], ...]`. β=1
//!    with single-term Prony `[[1.0, 1.0]]` reduces byte-identically
//!    to the Markovian path.
//!  - `TangentFlow` + `flow_kind == "banach_exponential"`: v1 Markovian
//!    exponential decay (Banach Q1 normalization).
//!  - `TangentFlow` + generic: v1 generic tangent-flow (ScalingRule with
//!    `nu` treated as `tau_obs`).
//!  - `LookupTable`: `Err(FlowError::LookupTableUnsupported)`.
//!  - `Learned`: not exposed by Python `flow()`; we keep the same scope
//!    here and return `Err(FlowError::LearnedUnsupported)`.

use serde_json::Value;

use crate::math::{banach_state, caputo_flow, tangent_flow_canonical};
use crate::types::{CanonicalState, TangentFlowField, TranslationField};

/// Errors `flow` raises in lieu of Python's `NotImplementedError` /
/// `ValueError`. The variants name the precondition that failed.
#[derive(Debug, Clone, PartialEq)]
pub enum FlowError {
    /// Called on a `LookupTable` translation field. Lookup tables sample
    /// the flow without an explicit generator.
    LookupTableUnsupported,
    /// Called on a `Learned` translation field. Python `flow()` does not
    /// dispatch the learned path; mirrors that scope.
    LearnedUnsupported,
    /// Caputo branch (`beta_mem < 1`) without an accompanying
    /// `prony_terms` array in the refinement dict.
    MissingPronyTerms,
    /// `prony_terms` entry is shaped wrong (not a `[amplitude, decay]`
    /// pair of numbers).
    MalformedPronyTerm,
}

impl std::fmt::Display for FlowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LookupTableUnsupported => write!(
                f,
                "flow() on lookup_table fields is unsupported (no explicit generator)"
            ),
            Self::LearnedUnsupported => write!(
                f,
                "flow() on learned fields is unsupported in this scope"
            ),
            Self::MissingPronyTerms => write!(
                f,
                "beta_mem < 1.0 requires prony_terms in refinement \
                 (curator-supplied Mittag-Leffler approximation)"
            ),
            Self::MalformedPronyTerm => write!(
                f,
                "prony_terms entries must be [amplitude, decay] pairs of numbers"
            ),
        }
    }
}

impl std::error::Error for FlowError {}

/// Continuous-form flow: canonical state at depth `nu`.
///
/// For integer `nu = N` this is equivalent to N successive applications
/// of the discrete map; for real `nu` it is the closed-form continuous
/// flow.
pub fn flow(
    canonical_initial: &CanonicalState,
    nu: f64,
    field: &TranslationField,
) -> Result<CanonicalState, FlowError> {
    match field {
        TranslationField::TangentFlow(tf) => flow_tangent(canonical_initial, nu, tf),
        TranslationField::LookupTable(_) => Err(FlowError::LookupTableUnsupported),
        TranslationField::Learned(_) => Err(FlowError::LearnedUnsupported),
    }
}

fn flow_tangent(
    initial: &CanonicalState,
    nu: f64,
    field: &TangentFlowField,
) -> Result<CanonicalState, FlowError> {
    let refinement = field.scaling.refinement.as_ref();
    let beta_mem = refinement
        .and_then(|m| m.get("beta_mem"))
        .and_then(Value::as_f64)
        .unwrap_or(1.0);
    let flow_kind = refinement
        .and_then(|m| m.get("flow_kind"))
        .and_then(Value::as_str);

    if beta_mem < 1.0 {
        return flow_caputo(initial, nu, refinement);
    }

    if flow_kind == Some("banach_exponential") {
        let lambda_chit = refinement
            .and_then(|m| m.get("lambda_chit"))
            .and_then(Value::as_f64)
            .unwrap_or(1.0);
        let lambda_gamma = refinement
            .and_then(|m| m.get("lambda_gamma"))
            .and_then(Value::as_f64)
            .unwrap_or(1.0);
        let (chit, gamma_AB) =
            banach_state(initial.chit, initial.gamma_AB, lambda_chit, lambda_gamma, nu);
        return Ok(CanonicalState {
            chit,
            gamma_AB,
            k_frust: initial.k_frust,
        });
    }

    let rule = &field.scaling;
    let (chit, gamma_AB) = tangent_flow_canonical(
        initial.chit,
        initial.gamma_AB,
        rule.delta_chit,
        rule.delta_gamma,
        nu,
        rule.tau_obs_ref,
    );
    Ok(CanonicalState {
        chit,
        gamma_AB,
        k_frust: initial.k_frust,
    })
}

fn flow_caputo(
    initial: &CanonicalState,
    nu: f64,
    refinement: Option<&std::collections::BTreeMap<String, Value>>,
) -> Result<CanonicalState, FlowError> {
    let prony = refinement
        .and_then(|m| m.get("prony_terms"))
        .and_then(Value::as_array)
        .filter(|a| !a.is_empty())
        .ok_or(FlowError::MissingPronyTerms)?;
    let lambda_chit = refinement
        .and_then(|m| m.get("lambda_chit"))
        .and_then(Value::as_f64)
        .unwrap_or(1.0);
    let lambda_gamma = refinement
        .and_then(|m| m.get("lambda_gamma"))
        .and_then(Value::as_f64)
        .unwrap_or(1.0);

    let mut amplitudes = Vec::with_capacity(prony.len());
    let mut decays = Vec::with_capacity(prony.len());
    for term in prony {
        let pair = term.as_array().ok_or(FlowError::MalformedPronyTerm)?;
        if pair.len() != 2 {
            return Err(FlowError::MalformedPronyTerm);
        }
        amplitudes.push(pair[0].as_f64().ok_or(FlowError::MalformedPronyTerm)?);
        decays.push(pair[1].as_f64().ok_or(FlowError::MalformedPronyTerm)?);
    }

    let (chit, gamma_AB) = caputo_flow(
        initial.chit,
        initial.gamma_AB,
        lambda_chit,
        lambda_gamma,
        nu,
        &amplitudes,
        &decays,
    );
    Ok(CanonicalState {
        chit,
        gamma_AB,
        k_frust: initial.k_frust,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        CanonicalPoint, Direction, Gt, LookupTableField, OperatingPoint, ScalingRule,
        TranslationRule,
    };
    use std::collections::BTreeMap;

    fn make_tf(refinement: Option<BTreeMap<String, Value>>) -> TangentFlowField {
        TangentFlowField {
            direction: Direction::Forward,
            rule_at_origin: TranslationRule {
                operating_point: OperatingPoint {
                    label: "origin".to_string(),
                    gt: Gt::S,
                    axes: BTreeMap::new(),
                },
                xdot_choice: "default".to_string(),
                canonical: CanonicalPoint {
                    chit: 0.0,
                    gamma_AB: 0.0,
                    k_frust: false,
                    method: "test".to_string(),
                    extras: BTreeMap::new(),
                },
            },
            scaling: ScalingRule {
                tau_obs_ref: 1.0,
                delta_chit: 0.0,
                delta_gamma: 0.0,
                refinement,
            },
            description: None,
        }
    }

    #[test]
    fn lookup_table_unsupported() {
        let field = TranslationField::LookupTable(LookupTableField {
            direction: Direction::Forward,
            rule: vec![],
            description: None,
        });
        let initial = CanonicalState {
            chit: 1.0,
            gamma_AB: 1.0,
            k_frust: false,
        };
        assert_eq!(
            flow(&initial, 1.0, &field),
            Err(FlowError::LookupTableUnsupported)
        );
    }

    #[test]
    fn banach_exponential_uses_banach_state() {
        let mut refinement = BTreeMap::new();
        refinement.insert(
            "flow_kind".to_string(),
            Value::String("banach_exponential".to_string()),
        );
        refinement.insert("lambda_chit".to_string(), Value::from(0.3));
        refinement.insert("lambda_gamma".to_string(), Value::from(0.4));
        let field = TranslationField::TangentFlow(make_tf(Some(refinement)));
        let initial = CanonicalState {
            chit: 2.0,
            gamma_AB: 3.0,
            k_frust: true,
        };
        let out = flow(&initial, 1.5, &field).unwrap();
        let (e_chit, e_gamma) = banach_state(2.0, 3.0, 0.3, 0.4, 1.5);
        assert_eq!(out.chit, e_chit);
        assert_eq!(out.gamma_AB, e_gamma);
        assert!(out.k_frust); // preserved
    }

    #[test]
    fn caputo_single_term_matches_banach_at_beta_one_amp_one() {
        // [[1.0, 1.0]] Prony at beta=0.999 (< 1, triggers Caputo) is
        // analytically identical to exp(-lambda * nu).
        let mut refinement = BTreeMap::new();
        refinement.insert("beta_mem".to_string(), Value::from(0.999));
        refinement.insert("lambda_chit".to_string(), Value::from(0.3));
        refinement.insert("lambda_gamma".to_string(), Value::from(0.4));
        refinement.insert(
            "prony_terms".to_string(),
            Value::Array(vec![Value::Array(vec![Value::from(1.0), Value::from(1.0)])]),
        );
        let field = TranslationField::TangentFlow(make_tf(Some(refinement)));
        let initial = CanonicalState {
            chit: 1.5,
            gamma_AB: 2.5,
            k_frust: false,
        };
        let out = flow(&initial, 1.0, &field).unwrap();
        let (e_chit, e_gamma) = banach_state(1.5, 2.5, 0.3, 0.4, 1.0);
        assert_eq!(out.chit, e_chit);
        assert_eq!(out.gamma_AB, e_gamma);
    }

    #[test]
    fn caputo_requires_prony_terms() {
        let mut refinement = BTreeMap::new();
        refinement.insert("beta_mem".to_string(), Value::from(0.7));
        let field = TranslationField::TangentFlow(make_tf(Some(refinement)));
        let initial = CanonicalState {
            chit: 1.0,
            gamma_AB: 1.0,
            k_frust: false,
        };
        assert_eq!(
            flow(&initial, 1.0, &field),
            Err(FlowError::MissingPronyTerms)
        );
    }

    #[test]
    fn generic_tangent_flow_at_nu_eq_ref_is_identity() {
        let field = TranslationField::TangentFlow(make_tf(None));
        let initial = CanonicalState {
            chit: 1.2,
            gamma_AB: 3.4,
            k_frust: false,
        };
        // tau_obs_ref = 1.0; nu = 1.0 → ratio = 1, ln(1) = 0, 1^delta = 1.
        let out = flow(&initial, 1.0, &field).unwrap();
        assert_eq!(out.chit, 1.2);
        assert_eq!(out.gamma_AB, 3.4);
    }
}
