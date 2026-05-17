//! Per-call provenance builder — port of `mpa_scale_solver/provenance.py`
//! (BLOCK_IN §v6 session 7).
//!
//! Stateless. Wrapped operations build a `Provenance` at the end of each
//! call and stamp it onto the `OperationOutput`. mpa-conform extracts
//! these records into the bundle's audit trail; mpa-auditor's display
//! layer reads them directly.
//!
//! Cross-language hash parity is the load-bearing contract: a fixed
//! `(solver_version, operation, dispatch_path, table_version)` tuple
//! must produce the same `provenance_hash` in Python and Rust. The hash
//! deliberately excludes timestamps and notes so it is reproducible.
//! Bit-identity test: `tests/bit_identity.rs::provenance_hash_python_to_rust_parity`.
//!
//! `timestamp_ns` is a process-local monotonic counter (Python uses
//! `time.monotonic_ns()`; Rust uses `Instant::elapsed` against a static
//! `OnceLock` epoch). Timestamps are NOT bit-identity-comparable across
//! runs; consumers comparing runs should ignore them.

use std::sync::OnceLock;
use std::time::Instant;

use blake2::digest::{Update, VariableOutput};
use blake2::Blake2bVar;

use crate::types::{DispatchPath, Provenance};

/// Solver version stamped into every Provenance. Mirrors
/// `mpa_scale_solver/_version.py::__version__` and MUST match the Python
/// constant byte-for-byte — `provenance_hash` includes this string, so
/// any drift breaks cross-language provenance parity. The Cargo `version`
/// in `Cargo.toml` is independent; this constant tracks the Python
/// solver's `__version__` (currently `"5.0.0"` — bumps with each shipped
/// Python release).
pub const SOLVER_VERSION: &str = "5.0.0";

/// Process-local monotonic epoch. Initialized lazily on first
/// `monotonic_ns` call. The absolute value of `timestamp_ns` is
/// arbitrary; only the difference between two timestamps is meaningful,
/// matching Python `time.monotonic_ns()`.
static EPOCH: OnceLock<Instant> = OnceLock::new();

fn monotonic_ns() -> i64 {
    let epoch = EPOCH.get_or_init(Instant::now);
    // `as_nanos` returns u128; cast to i64 saturating (the field is i64
    // to match Python's int range). Saturation is the documented choice
    // for processes that outlive 2^63 ns ≈ 292 years.
    epoch.elapsed().as_nanos().min(i64::MAX as u128) as i64
}

/// Stamp a `Provenance` for the current call.
///
/// Python signature:
///
/// ```python
/// def make_provenance(
///     operation: str,
///     *,
///     dispatch_path: DispatchPath = DispatchPath.DIRECT_COMPUTE,
///     table_version: Optional[str] = None,
///     notes: Iterable[str] = (),
/// ) -> Provenance
/// ```
///
/// Rust takes all arguments explicitly — Rust has no keyword args, and
/// inventing a builder pattern would thicken the surface beyond what the
/// thin discipline allows. Callers pass defaults explicitly:
/// `make_provenance("regime_at", DispatchPath::DirectCompute, None, vec![])`.
pub fn make_provenance(
    operation: &str,
    dispatch_path: DispatchPath,
    table_version: Option<String>,
    notes: Vec<String>,
) -> Provenance {
    Provenance {
        solver_version: SOLVER_VERSION.to_string(),
        operation: operation.to_string(),
        timestamp_ns: monotonic_ns(),
        dispatch_path,
        table_version,
        notes,
    }
}

/// Raw 4-byte blake2b digest of a provenance record's hash inputs.
///
/// This is the load-bearing cross-language contract: the digest bytes
/// must match Python's `hashlib.blake2b(payload, digest_size=4).digest()`
/// byte-for-byte. The `provenance_hash` float view is derived; the
/// digest is the authoritative artifact.
///
/// Inputs are `solver_version | operation | dispatch_path | table_version`
/// (timestamps and notes are excluded so the digest is reproducible
/// across runs).
///
/// Exposed as a separate function so cross-language fixtures can compare
/// digest bytes directly (via hex) rather than the float view — JSON's
/// decimal-string float representation is lossy under naive parsers,
/// and the rational hash leaves no ULP tolerance budget. See
/// `tests/bit_identity.rs::provenance_hash_python_to_rust_parity`.
pub fn provenance_digest_bytes(prov: &Provenance) -> [u8; 4] {
    let payload = format!(
        "{}|{}|{}|{}",
        prov.solver_version,
        prov.operation,
        dispatch_path_value(prov.dispatch_path),
        prov.table_version.as_deref().unwrap_or(""),
    );
    let mut hasher = Blake2bVar::new(4).expect("blake2b-32 always valid");
    hasher.update(payload.as_bytes());
    let mut digest = [0u8; 4];
    hasher
        .finalize_variable(&mut digest)
        .expect("blake2b-32 always finalizes");
    digest
}

/// Stable float-encoded hash of a provenance record (handoff §A.5).
///
/// Used by the EXR channel `provenance_hash` so frame-level dispatch
/// fingerprints are queryable without unpacking the full provenance
/// payload. The 4-byte big-endian digest from `provenance_digest_bytes`
/// is mapped onto `[0, 1)` by dividing the unsigned int by `2^32`.
///
/// For cross-language equality tests, prefer `provenance_digest_bytes`
/// + hex comparison: JSON cannot round-trip f64 losslessly under naive
/// parsers, and the rational hash leaves no ULP tolerance budget. The
/// float view here is the consumer artifact, not the parity artifact.
pub fn provenance_hash(prov: &Provenance) -> f64 {
    let digest = provenance_digest_bytes(prov);
    let n = u32::from_be_bytes(digest);
    n as f64 / (1u64 << 32) as f64
}

/// Python `DispatchPath` enum's `.value` string. Used by both
/// `provenance_hash` (deterministic input) and JSON serialization
/// (matches the serde `rename_all = "snake_case"` on the Rust enum).
fn dispatch_path_value(dp: DispatchPath) -> &'static str {
    match dp {
        DispatchPath::TableHit => "table_hit",
        DispatchPath::ComputeFallback => "compute_fallback",
        DispatchPath::DirectCompute => "direct_compute",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(operation: &str) -> Provenance {
        make_provenance(operation, DispatchPath::DirectCompute, None, Vec::new())
    }

    #[test]
    fn make_provenance_carries_solver_version() {
        let prov = p("test_op");
        assert_eq!(prov.solver_version, SOLVER_VERSION);
        assert_eq!(prov.operation, "test_op");
        assert_eq!(prov.dispatch_path, DispatchPath::DirectCompute);
        assert!(prov.table_version.is_none());
        assert!(prov.notes.is_empty());
    }

    #[test]
    fn make_provenance_table_hit_path() {
        let prov = make_provenance(
            "apply_translation",
            DispatchPath::TableHit,
            Some("banach-1.0.0".to_string()),
            Vec::new(),
        );
        assert_eq!(prov.dispatch_path, DispatchPath::TableHit);
        assert_eq!(prov.table_version.as_deref(), Some("banach-1.0.0"));
    }

    #[test]
    fn make_provenance_timestamp_monotonic() {
        let p1 = p("a");
        let p2 = p("b");
        assert!(p2.timestamp_ns >= p1.timestamp_ns);
    }

    #[test]
    fn provenance_hash_excludes_timestamp_and_notes() {
        let p1 = p("apply_translation");
        // A second call has a later timestamp but no other distinguishing
        // input — hash must be byte-equal.
        let p2 = p("apply_translation");
        assert_eq!(provenance_hash(&p1), provenance_hash(&p2));

        // Adding notes must NOT change the hash either.
        let p_with_notes = make_provenance(
            "apply_translation",
            DispatchPath::DirectCompute,
            None,
            vec!["note A".to_string(), "note B".to_string()],
        );
        assert_eq!(provenance_hash(&p1), provenance_hash(&p_with_notes));
    }

    #[test]
    fn provenance_hash_differs_by_operation() {
        let p1 = p("apply_translation");
        let p2 = p("forward_sweep_invert");
        assert_ne!(provenance_hash(&p1), provenance_hash(&p2));
    }

    #[test]
    fn provenance_hash_differs_by_dispatch_path() {
        let p1 = make_provenance("apply_translation", DispatchPath::DirectCompute, None, vec![]);
        let p2 = make_provenance("apply_translation", DispatchPath::TableHit, None, vec![]);
        assert_ne!(provenance_hash(&p1), provenance_hash(&p2));
    }

    #[test]
    fn provenance_hash_differs_by_table_version() {
        let p1 = make_provenance("apply_translation", DispatchPath::TableHit, None, vec![]);
        let p2 = make_provenance(
            "apply_translation",
            DispatchPath::TableHit,
            Some("banach-1.0.0".to_string()),
            vec![],
        );
        assert_ne!(provenance_hash(&p1), provenance_hash(&p2));
    }

    #[test]
    fn provenance_hash_in_unit_interval() {
        for op in &["apply_translation", "forward_sweep_invert", "regime_at"] {
            let h = provenance_hash(&p(op));
            assert!(h >= 0.0 && h < 1.0, "hash for {op} out of [0, 1): {h}");
        }
    }

    #[test]
    fn dispatch_path_value_matches_serde() {
        // The dispatch_path string used in the hash MUST match the serde
        // serialization of the enum so that JSON round-tripping a
        // Provenance preserves the hash.
        for dp in [
            DispatchPath::DirectCompute,
            DispatchPath::TableHit,
            DispatchPath::ComputeFallback,
        ] {
            let serde_value = serde_json::to_value(dp).expect("serialize dispatch path");
            let serde_str = serde_value.as_str().expect("dispatch path serde -> str");
            assert_eq!(serde_str, dispatch_path_value(dp));
        }
    }
}
