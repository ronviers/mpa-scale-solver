//! mpa-scale-solver — Rust port.
//!
//! v5 Python (`mpa_scale_solver/`) is the pseudo-code spec. Every function
//! here maps 1:1 to its Python counterpart, and bit-identity tests in
//! `tests/bit_identity.rs` compare the two over a seed grid.
//!
//! Module mapping (Python → Rust):
//!   jax_core.py        → math.rs        (session 1: 12 primitives)
//!   types.py           → types.rs       (session 3: dataclass shapes)
//!   operations.py      → operations.rs  (future session)
//!   sensitivity.py     → sensitivity.rs (future session)
//!   self_test.py       → self_test.rs   (future session)
//!   streaming.py       → streaming.rs   (future session)
//!   ...                → ...

pub mod math;
pub mod types;
