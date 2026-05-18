//! Session 9 — bindings.
//!
//! Two consumer surfaces sit on top of the Rust port:
//!
//! * `python` (pyo3 + pythonize) — extension module
//!   `_mpa_scale_solver_native`. The pure-Python `mpa_scale_solver`
//!   package detects it at import time and routes the wrapped variants
//!   through it when present (the Python source remains as the
//!   executable reference).
//!
//! * `wasm` (wasm-bindgen + serde-wasm-bindgen) — built by
//!   `wasm-pack build --release --target nodejs|web --features wasm`.
//!   mpa-auditor consumes the output for browser-side native solving.
//!
//! Both bindings share the same **dict-shape contract**: callers pass
//! Python dicts / JS objects, serde deserializes into the Rust
//! `types::` structs, the wrapped variant runs, serde serializes the
//! `OperationOutput<T>` back. No per-`T` concrete wrappers — the
//! `Serialize + Deserialize` derives on `types.rs` carry the surface.
//!
//! Closure-typed parameters on the Rust API (`score_fn`,
//! `forward_map`) are NOT exposed across the binding boundary — the
//! bindings always pass `None`, matching the Python defaults.

#[cfg(feature = "python")]
pub mod python;

#[cfg(feature = "wasm")]
pub mod wasm;
