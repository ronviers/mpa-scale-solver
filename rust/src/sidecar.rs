//! Inverse-lookup-table sidecar dispatch — port of
//! `mpa_scale_solver/sidecar.py`.
//!
//! The sidecar is a curator-precomputed table that lets
//! `forward_sweep_invert` short-circuit the brute-force grid search when
//! the `(substrate, tau_obs)` pair is in the table. Sidecar *production*
//! is mpa-conform's curator-path job; this module provides the dispatch
//! helpers.
//!
//! ## Cross-language parity caveat
//!
//! Python rounds the float key via the built-in `round(x, n)` which uses
//! banker's rounding via CPython's `dtoa`-based pipeline. Rust here uses
//! `(x * 10^n).round_ties_even() / 10^n`. These agree for the bulk of
//! double-precision inputs but can diverge for values exactly halfway
//! between two representable decimals — a producer/consumer pair must
//! either both be Python or both be Rust until the wire-format parity
//! check lands. (BLOCK_IN §v6: cross-language JSON parity for shape-
//! bearing types lands when the first module with JSON I/O ports.)
//! For now Rust-Rust round-trips are bit-identical; Python-Rust round-
//! trips are at the producer's mercy.

use crate::types::{CanonicalState, InverseLookupSidecar, SidecarKey, SubstrateState};

/// Default key-rounding precision. Producers and consumers must agree;
/// `banach::BanachSubstrate::build_sidecar` (future port) uses this value.
/// Mirrors Python `sidecar.DEFAULT_ROUNDING_DECIMALS`.
pub const DEFAULT_ROUNDING_DECIMALS: i32 = 6;

/// Round one float to `decimals` places using banker's rounding (round
/// half to even). Approximates Python's built-in `round(x, n)` to within
/// the divergence noted in this module's docstring.
fn round_decimal(x: f64, decimals: i32) -> f64 {
    if !x.is_finite() {
        return x;
    }
    let scale = 10.0_f64.powi(decimals);
    (x * scale).round_ties_even() / scale
}

/// Round a 3-tuple key to the agreed precision. Wraps `round_decimal`.
pub fn round_key(
    chit: f64,
    gamma_ab: f64,
    tau_obs: f64,
    decimals: i32,
) -> (f64, f64, f64) {
    (
        round_decimal(chit, decimals),
        round_decimal(gamma_ab, decimals),
        round_decimal(tau_obs, decimals),
    )
}

/// Build a `SidecarKey` from the unrounded floats at the default
/// precision. Most callers want this — Python `sidecar.round_key(...)`
/// followed by tuple-keying is one step here.
pub fn key_at_default(chit: f64, gamma_ab: f64, tau_obs: f64) -> SidecarKey {
    let (c, g, t) = round_key(chit, gamma_ab, tau_obs, DEFAULT_ROUNDING_DECIMALS);
    SidecarKey::from_floats(c, g, t)
}

/// Build a `SidecarKey` at a non-default precision (uncommon).
pub fn key_at_precision(
    chit: f64,
    gamma_ab: f64,
    tau_obs: f64,
    decimals: i32,
) -> SidecarKey {
    let (c, g, t) = round_key(chit, gamma_ab, tau_obs, decimals);
    SidecarKey::from_floats(c, g, t)
}

/// Table-first inverse lookup. Returns the recorded canonical state if
/// `(substrate, tau_obs)` is in the sidecar's inverse table; `None` on
/// miss. Callers fall through to the compute path on `None`.
///
/// Substrate-side keying uses `observables["substrate_chit"]` /
/// `observables["substrate_gamma_AB"]` — the canonical curator
/// convention. Substrates without those keys are a guaranteed miss.
pub fn lookup_inverse(
    sidecar: &InverseLookupSidecar,
    substrate: &SubstrateState,
    tau_obs: f64,
    decimals: i32,
) -> Option<CanonicalState> {
    let chit = substrate.observables.get("substrate_chit")?;
    let gamma = substrate.observables.get("substrate_gamma_AB")?;
    let key = key_at_precision(*chit, *gamma, tau_obs, decimals);
    sidecar.inverse_lookup.get(&key).cloned()
}

/// Table-first forward lookup. Returns the recorded substrate state on
/// hit; `None` on miss.
pub fn lookup_forward(
    sidecar: &InverseLookupSidecar,
    canonical: &CanonicalState,
    tau_obs: f64,
    decimals: i32,
) -> Option<SubstrateState> {
    let key = key_at_precision(canonical.chit, canonical.gamma_AB, tau_obs, decimals);
    sidecar.forward_lookup.get(&key).cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_decimal_banker() {
        // 1.5 → 2 (round to even), 2.5 → 2 (round to even).
        assert_eq!(round_decimal(1.5, 0), 2.0);
        assert_eq!(round_decimal(2.5, 0), 2.0);
        assert_eq!(round_decimal(-1.5, 0), -2.0);
        assert_eq!(round_decimal(-2.5, 0), -2.0);
    }

    #[test]
    fn round_key_truncates_to_six() {
        let (c, g, t) = round_key(
            1.234_567_891_234,
            -2.345_678_912_345,
            10.123_456_789,
            DEFAULT_ROUNDING_DECIMALS,
        );
        assert_eq!(c, 1.234_568);
        assert_eq!(g, -2.345_679);
        assert_eq!(t, 10.123_457);
    }

    #[test]
    fn round_decimal_passes_non_finite_through() {
        assert!(round_decimal(f64::NAN, 6).is_nan());
        assert_eq!(round_decimal(f64::INFINITY, 6), f64::INFINITY);
        assert_eq!(round_decimal(f64::NEG_INFINITY, 6), f64::NEG_INFINITY);
    }
}
