//! mpa-scale-solver — Rust port.
//!
//! v5 Python (`mpa_scale_solver/`) is the pseudo-code spec. Every function
//! here maps 1:1 to its Python counterpart, and bit-identity tests in
//! `tests/bit_identity.rs` compare the two over a seed grid.
//!
//! Module mapping (Python → Rust):
//!   jax_core.py        → math.rs        (session 1: 12 primitives)
//!   types.py           → types.rs       (session 3: dataclass shapes)
//!   gfdr_model.py      → gfdr_model.rs  (session 4)
//!   sidecar.py         → sidecar.rs     (session 4)
//!   flow.py            → flow.rs        (session 4)
//!   operations.py      → operations.rs  (session 4: raw forward path;
//!                                        session 5: gradient inversion
//!                                        dispatcher; session 6: intent
//!                                        algebra; wrapped variants +
//!                                        posterior land in subsequent
//!                                        sessions per BLOCK_IN §v6)
//!   sensitivity.py     → sensitivity.rs (future session)
//!   self_test.py       → self_test.rs   (future session)
//!   streaming.py       → streaming.rs   (future session)
//!   validation.py      → validation.rs  (future session)
//!   provenance.py      → provenance.rs  (future session)

pub mod flow;
pub mod gfdr_model;
pub mod math;
pub mod operations;
pub mod optim;
pub mod sidecar;
pub mod types;
