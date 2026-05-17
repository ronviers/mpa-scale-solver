//! Analytical gFDR forward model — port of `mpa_scale_solver/gfdr_model.py`.
//!
//! Five pure functions plus a residual scorer. The JS canonical version
//! (`mpa-auditor/math/gfdr-model.js`) is the source of truth; the Python
//! is a faithful port and this Rust is a faithful port of the Python.
//!
//! `vertex_regime` is the canonical FIVE-bucket classifier. The
//! three-bucket cut lives in `operations::regime_display_band`.
//!
//! `generate_locus` depends on `chit` alone — the single-mode gFDR locus
//! does not constrain `gamma_AB` (RFC-S Appendix B item 4).

use serde::{Deserialize, Serialize};

use crate::types::RegimeLabel;

/// Number of points sampled per locus — matches Python `N_LOCUS_POINTS`.
pub const N_LOCUS_POINTS: usize = 80;

/// One point on a gFDR locus: `(tau, chi, C)`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LocusPoint {
    pub tau: f64,
    pub chi: f64,
    #[serde(rename = "C")]
    pub c: f64,
}

/// An empirical-locus row. Mirrors the Python `dict` shape `{tau, C, chi}`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EmpiricalRow {
    pub tau: f64,
    pub chi: f64,
    #[serde(rename = "C")]
    pub c: f64,
}

/// Five-bucket regime classifier. Canonical; matches `gfdr-model.js`.
///
/// ```text
///  chit >= 0.7   → DeepC
///  chit >= 0.2   → CNearS
///  |chit| < 0.2  → SCritical
///  chit > -0.7   → RNearS
///  else          → DeepR
/// ```
pub fn vertex_regime(chit: f64) -> RegimeLabel {
    if chit >= 0.7 {
        RegimeLabel::DeepC
    } else if chit >= 0.2 {
        RegimeLabel::CNearS
    } else if chit > -0.2 {
        RegimeLabel::SCritical
    } else if chit > -0.7 {
        RegimeLabel::RNearS
    } else {
        RegimeLabel::DeepR
    }
}

/// CK s-regime aging-diagonal slope. `cdv1 §gFDR signatures`.
pub fn alpha_s(chit: f64) -> f64 {
    0.5 + 0.3 * (-chit.abs() * 4.0).exp()
}

/// Plateau height of the s-critical locus.
pub fn plateau_height(chit: f64) -> f64 {
    let arg = -((chit + 0.2).max(0.0)) * 1.5;
    (1.0 - arg.exp()).max(0.05)
}

/// Analytical gFDR locus `chi(tau) / C(tau)`, log-spaced in tau.
///
/// Returns `N_LOCUS_POINTS` rows. Branch set matches the engines'
/// continuous-mode `generateLocus`.
pub fn generate_locus(chit: f64, regime: RegimeLabel) -> Vec<LocusPoint> {
    let tau_min: f64 = 0.01;
    let tau_max: f64 = 1000.0;
    let n = N_LOCUS_POINTS;
    let mut points = Vec::with_capacity(n);
    for i in 0..n {
        let t = (i as f64) / ((n - 1) as f64);
        let tau = tau_min * (tau_max / tau_min).powf(t);
        let (chi, c) = match regime {
            RegimeLabel::DeepC | RegimeLabel::CNearS => {
                let depth = (-chit * 1.5).exp();
                let tau_c = 4.0 + 6.0 / chit.max(0.1);
                let d_c = 0.18 * depth * (1.0 - (-tau / tau_c).exp());
                let c = 1.0 - d_c;
                let prefactor = if matches!(regime, RegimeLabel::DeepC) {
                    0.02
                } else {
                    0.08
                };
                (prefactor * d_c, c)
            }
            RegimeLabel::SCritical => {
                let a = alpha_s(chit);
                let p_s = plateau_height(chit);
                let d_c_short = (1.0 - p_s) * (1.0 - (-tau / 0.5).exp());
                let d_c_long = p_s * (1.0 - (1.0 + tau / 50.0).powf(-a));
                let d_c = d_c_short + d_c_long;
                let c = 1.0 - d_c;
                let chi = if d_c <= (1.0 - p_s) {
                    d_c
                } else {
                    (1.0 - p_s) + a * (d_c - (1.0 - p_s))
                };
                (chi, c)
            }
            RegimeLabel::RNearS | RegimeLabel::DeepR => {
                let tau_eq = (1.0 + 0.5 * chit.exp()).max(0.5);
                let d_c = 1.0 - (-tau / tau_eq).exp();
                (d_c, 1.0 - d_c)
            }
        };
        points.push(LocusPoint { tau, chi, c });
    }
    points
}

/// Log-tau interpolation of a gFDR locus at an arbitrary tau. Returns
/// `(C, chi)`.
///
/// Clamps to the endpoints outside `[model[0].tau, model[-1].tau]`.
pub fn interp_locus(model: &[LocusPoint], tau: f64) -> (f64, f64) {
    let first = &model[0];
    if tau <= first.tau {
        return (first.c, first.chi);
    }
    let last = &model[model.len() - 1];
    if tau >= last.tau {
        return (last.c, last.chi);
    }
    for i in 1..model.len() {
        if model[i].tau >= tau {
            let a = &model[i - 1];
            let b = &model[i];
            let f = (tau.ln() - a.tau.ln()) / (b.tau.ln() - a.tau.ln());
            return (a.c + f * (b.c - a.c), a.chi + f * (b.chi - a.chi));
        }
    }
    (last.c, last.chi)
}

/// Mean squared residual between an empirical locus and the analytical
/// at `chit`.
///
/// Empirical rows: `{tau, C, chi}`. Returns the score that
/// `forward_sweep_invert` minimizes when the empirical signal is a gFDR
/// locus.
pub fn locus_residual(empirical_rows: &[EmpiricalRow], chit: f64) -> f64 {
    let model = generate_locus(chit, vertex_regime(chit));
    let mut sse = 0.0_f64;
    for row in empirical_rows {
        let (c_m, chi_m) = interp_locus(&model, row.tau);
        let d_c = row.c - c_m;
        let d_chi = row.chi - chi_m;
        sse += d_c * d_c + d_chi * d_chi;
    }
    sse / empirical_rows.len().max(1) as f64
}

/// Vectorized scan of `locus_residual` over a chit candidate grid.
pub fn locus_residual_array(empirical_rows: &[EmpiricalRow], chit_grid: &[f64]) -> Vec<f64> {
    chit_grid
        .iter()
        .map(|&chit| locus_residual(empirical_rows, chit))
        .collect()
}
