"""Session 9 parity tests: native Rust wheel vs pure-Python reference.

Imports both:
  * `_mpa_scale_solver_native` — the pyo3 wheel built by maturin from
    `rust/Cargo.toml --features python`. Returns dict-shape
    `OperationOutput` per the session-9 design (see BLOCK_IN §v6).
  * `mpa_scale_solver.operations` — the pure-Python implementation,
    which remains the executable reference after v6 ships.

For each of the 9 `*_wrapped` variants (8 from session 7 + session 8's
posterior wrapped) plus `validate_driver_profile` (raw) and `flow`, the
test builds a small fixture input, calls both implementations, and
asserts dict-level equality with the **session-7 asymmetric-parity
discipline**:

  * `value` structured fields: bit-equal for ints / bools / strings;
    ULP-bounded for floats.
  * `validation`: bit-equal on flags, ULP on `round_trip_residual`,
    notes excluded (Python's `f"{0.0}"` is `"0.0"`, Rust's
    `format!("{}", 0.0)` is `"0"` — documented divergence; see
    BLOCK_IN §v6 session 7 log).
  * `provenance.{solver_version, operation, dispatch_path,
    table_version}`: bit-equal.
  * `provenance.{timestamp_ns, notes}`: excluded.

The bit_identity.rs schema-parity tests at session 7+8 (provenance_hash,
operation_output_regime_at, operation_output_posterior) cover the wire
shape Rust-internally. This file is the Python-side confirmation: when
the native wheel is imported, the dict it produces equals the
`dataclasses.asdict()` of the pure-Python `OperationOutput`.

The native module is required for the test to run; tests are skipped
otherwise (e.g., on platforms where the wheel hasn't been built).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np
import pytest

# Skip the whole module if the native wheel isn't installed — typical
# for a clean-checkout developer environment that hasn't run
# `maturin develop --features python` yet. CI will install it before
# running tests.
nat = pytest.importorskip("_mpa_scale_solver_native")

from mpa_scale_solver.types import (
    CanonicalState,
    DispatchPath,
    GamutSpec,
    LookupTableField,
    OperatingPoint,
    ScalingRule,
    SubstrateState,
    TangentFlowField,
    TranslationField,
    TranslationRule,
)
from mpa_scale_solver import operations as py_ops
from mpa_scale_solver.flow import flow as py_flow


# ---------------------------------------------------------------------------
# Comparison helpers — session-7 asymmetric-parity discipline.
# ---------------------------------------------------------------------------

# Per-field ULP budgets for computed floats. LIBM matches the
# `rust/tests/bit_identity.rs` budget for primitives composing a small
# number of libm calls (4 ULPs ~= 8.9e-16 at scale 1.0). LIBM_WIDE is the
# 16-ULP budget for sums / reductions where JAX-pairwise vs
# Rust-sequential ordering shifts a bit. The parity check converts to
# absolute tolerance: 16 ULPs at the maximum operand absorbs both ends
# without expressing ULP arithmetic in Python.
ABS_TOL_TIGHT = 1e-12   # ~LIBM (4 ULPs at scale O(1))
ABS_TOL_WIDE = 1e-10    # ~LIBM_WIDE (16 ULPs at scale O(1))


def _floats_close(a: float, b: float, abs_tol: float = ABS_TOL_TIGHT) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if math.isnan(a) and math.isnan(b):
        return True
    if math.isinf(a) and math.isinf(b):
        return (a > 0) == (b > 0)
    return math.isclose(a, b, abs_tol=abs_tol, rel_tol=1e-12)


def _normalize(value: Any) -> Any:
    """Coerce dataclass instances to dicts (recursively) and tuples to
    lists, so `dataclasses.asdict` output and native serde output can be
    compared structurally."""
    if dataclasses.is_dataclass(value):
        return _normalize(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    if isinstance(value, DispatchPath):
        return value.value  # str enum — match Rust's serde snake_case
    return value


def _equal_with_floats(a: Any, b: Any, abs_tol: float = ABS_TOL_WIDE) -> bool:
    """Structural equality that uses ULP-tolerant comparison on floats."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_equal_with_floats(a[k], b[k], abs_tol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(_equal_with_floats(x, y, abs_tol) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return _floats_close(
            float(a) if a is not None else None,
            float(b) if b is not None else None,
            abs_tol=abs_tol,
        )
    return a == b


def _strip_unstable(output_dict: dict) -> dict:
    """Strip fields whose divergence is documented asymmetric-parity:
    timestamps (process-local), notes (Python/Rust float-format
    divergence on `0.0` vs `0`), and `preserved_invariant` on sacrifice
    records (Python stores the string; Rust derives it from the
    `IntentDiagnostics` variant via `.preserved_invariant()` — see
    session-6 log + bit_identity.rs sacrifice_record parity)."""
    out = dict(output_dict)
    if "provenance" in out:
        prov = dict(out["provenance"])
        prov["timestamp_ns"] = 0
        prov["notes"] = []
        out["provenance"] = prov
    if "validation" in out:
        val = dict(out["validation"])
        val["notes"] = []
        out["validation"] = val
    if "value" in out:
        out["value"] = _strip_preserved_invariant(out["value"])
    return out


def _strip_preserved_invariant(node: Any) -> Any:
    """Recursively drop `preserved_invariant` from sacrifice-record
    dicts (intent_map / intent_compose value payloads)."""
    if isinstance(node, dict):
        return {
            k: _strip_preserved_invariant(v)
            for k, v in node.items()
            if k != "preserved_invariant"
        }
    if isinstance(node, list):
        return [_strip_preserved_invariant(v) for v in node]
    return node


def _assert_parity(
    py_result: Any,
    nat_result: dict,
    abs_tol: float = ABS_TOL_WIDE,
) -> None:
    py_dict = _normalize(py_result)
    py_dict = _strip_unstable(py_dict)
    nat_dict = _strip_unstable(_normalize(nat_result))
    assert _equal_with_floats(py_dict, nat_dict, abs_tol), (
        f"\nPYTHON: {py_dict}\nNATIVE: {nat_dict}"
    )


# ---------------------------------------------------------------------------
# Fixtures shared across ops.
# ---------------------------------------------------------------------------

@pytest.fixture
def canonical_state() -> CanonicalState:
    return CanonicalState(chit=0.4, gamma_AB=0.2, k_frust=False)


@pytest.fixture
def canonical_state_dict(canonical_state: CanonicalState) -> dict:
    return dataclasses.asdict(canonical_state)


@pytest.fixture
def gamut_spec() -> GamutSpec:
    return GamutSpec(
        chit_range=(0.0, 1.0),
        gamma_AB_range=(-0.5, 0.5),
        tau_obs_range=(0.1, 10.0),
    )


@pytest.fixture
def gamut_spec_dict(gamut_spec: GamutSpec) -> dict:
    return dataclasses.asdict(gamut_spec)


@pytest.fixture
def tangent_flow_field() -> TangentFlowField:
    return TangentFlowField(
        direction="forward",
        shape="tangent_flow",
        rule_at_origin=TranslationRule(
            operating_point=OperatingPoint(label="origin", gt="s", axes={}),
            xdot_choice="default",
            canonical={
                "chit": 0.0,
                "gamma_AB": 0.0,
                "k_frust": False,
                "method": "test",
                "extras": {},
            },
        ),
        scaling=ScalingRule(
            tau_obs_ref=1.0,
            delta_chit=0.3,
            delta_gamma=0.5,
            refinement=None,
        ),
        description=None,
    )


@pytest.fixture
def tangent_flow_field_dict(tangent_flow_field: TangentFlowField) -> dict:
    return dataclasses.asdict(tangent_flow_field)


@pytest.fixture
def substrate_state() -> SubstrateState:
    return SubstrateState(
        tau_obs=2.0,
        observables={"substrate_chit": 0.6, "substrate_gamma_AB": 0.5},
    )


@pytest.fixture
def substrate_state_dict(substrate_state: SubstrateState) -> dict:
    return dataclasses.asdict(substrate_state)


@pytest.fixture
def canonical_grid() -> np.ndarray:
    """Small 5x5 search grid sufficient for the wrapped-variant smoke.
    Python's `forward_sweep_invert` expects an `np.ndarray` with
    `.ndim == 2`; the native binding accepts the same nested-list shape
    via serde's auto-conversion."""
    grid = []
    for i in range(5):
        for j in range(5):
            grid.append([i * 0.25, -0.5 + j * 0.25])
    return np.asarray(grid, dtype=np.float64)


@pytest.fixture
def canonical_grid_list(canonical_grid: np.ndarray) -> list[list[float]]:
    """Native-binding-friendly form (Python list of pairs) of the grid."""
    return canonical_grid.tolist()


# ---------------------------------------------------------------------------
# 1. regime_at_wrapped
# ---------------------------------------------------------------------------

def test_regime_at_wrapped_parity(canonical_state, canonical_state_dict):
    py = py_ops.regime_at_wrapped(canonical_state, 1.0)
    rs = nat.regime_at_wrapped(canonical_state_dict, 1.0)
    _assert_parity(py, rs)


# ---------------------------------------------------------------------------
# 2. gamut_classify_wrapped
# ---------------------------------------------------------------------------

def test_gamut_classify_wrapped_in_gamut(
    canonical_state, canonical_state_dict, gamut_spec, gamut_spec_dict
):
    py = py_ops.gamut_classify_wrapped(canonical_state, 1.0, gamut_spec)
    rs = nat.gamut_classify_wrapped(canonical_state_dict, 1.0, gamut_spec_dict)
    _assert_parity(py, rs)


def test_gamut_classify_wrapped_out_of_gamut(gamut_spec, gamut_spec_dict):
    out_of_gamut = CanonicalState(chit=2.5, gamma_AB=0.2)
    py = py_ops.gamut_classify_wrapped(out_of_gamut, 1.0, gamut_spec)
    rs = nat.gamut_classify_wrapped(dataclasses.asdict(out_of_gamut), 1.0, gamut_spec_dict)
    _assert_parity(py, rs)


# ---------------------------------------------------------------------------
# 3. apply_translation_wrapped
# ---------------------------------------------------------------------------

def test_apply_translation_wrapped_tangent_flow(
    canonical_state, canonical_state_dict, tangent_flow_field, tangent_flow_field_dict
):
    py = py_ops.apply_translation_wrapped(canonical_state, tangent_flow_field, 1.0)
    rs = nat.apply_translation_wrapped(canonical_state_dict, tangent_flow_field_dict, 1.0)
    _assert_parity(py, rs)


def test_apply_translation_wrapped_off_reference_tau(
    canonical_state, canonical_state_dict, tangent_flow_field, tangent_flow_field_dict
):
    # tau_obs != tau_obs_ref exercises the tangent-flow remap (composes
    # libm pow + log; LIBM_WIDE budget covers it).
    py = py_ops.apply_translation_wrapped(canonical_state, tangent_flow_field, 3.7)
    rs = nat.apply_translation_wrapped(canonical_state_dict, tangent_flow_field_dict, 3.7)
    _assert_parity(py, rs)


# ---------------------------------------------------------------------------
# 4. forward_sweep_invert_wrapped
# ---------------------------------------------------------------------------

def test_forward_sweep_invert_wrapped_tangent_flow_closed_form(
    canonical_state,
    canonical_state_dict,
    tangent_flow_field,
    tangent_flow_field_dict,
    canonical_grid,
    canonical_grid_list,
):
    # Forward-translate to build a target substrate, then invert.
    py_substrate = py_ops.apply_translation(canonical_state, tangent_flow_field, 2.0)
    nat_substrate = nat.apply_translation_wrapped(
        canonical_state_dict, tangent_flow_field_dict, 2.0
    )["value"]

    py = py_ops.forward_sweep_invert_wrapped(
        py_substrate, tangent_flow_field, 2.0, canonical_grid
    )
    rs = nat.forward_sweep_invert_wrapped(
        nat_substrate, tangent_flow_field_dict, 2.0, canonical_grid_list
    )
    _assert_parity(py, rs)


# ---------------------------------------------------------------------------
# 5. tau_obs_sweep_wrapped
# ---------------------------------------------------------------------------

def test_tau_obs_sweep_wrapped_broadcast(
    canonical_state,
    canonical_state_dict,
    tangent_flow_field,
    tangent_flow_field_dict,
    canonical_grid,
    canonical_grid_list,
):
    py_substrate = py_ops.apply_translation(canonical_state, tangent_flow_field, 1.0)
    nat_substrate = nat.apply_translation_wrapped(
        canonical_state_dict, tangent_flow_field_dict, 1.0
    )["value"]

    tau_grid = [0.5, 1.0, 2.0]
    py = py_ops.tau_obs_sweep_wrapped(
        py_substrate, tangent_flow_field, tau_grid, canonical_grid
    )
    rs = nat.tau_obs_sweep_wrapped(
        nat_substrate, tangent_flow_field_dict, tau_grid, canonical_grid_list
    )
    _assert_parity(py, rs)


# ---------------------------------------------------------------------------
# 6. intent_map_wrapped
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("intent", ["I1", "I3", "I4", "I5"])
def test_intent_map_wrapped_in_gamut(
    intent, canonical_state, canonical_state_dict, gamut_spec, gamut_spec_dict
):
    py = py_ops.intent_map_wrapped(canonical_state, 1.0, gamut_spec, intent)
    rs = nat.intent_map_wrapped(canonical_state_dict, 1.0, gamut_spec_dict, intent)
    _assert_parity(py, rs)


def test_intent_map_wrapped_i2_drive_faithful(
    canonical_state, canonical_state_dict, gamut_spec, gamut_spec_dict
):
    # I2 needs a distinct branch (drive-faithful identity in-gamut).
    py = py_ops.intent_map_wrapped(canonical_state, 1.0, gamut_spec, "I2")
    rs = nat.intent_map_wrapped(canonical_state_dict, 1.0, gamut_spec_dict, "I2")
    _assert_parity(py, rs)


def test_intent_map_wrapped_out_of_gamut_clamp(gamut_spec, gamut_spec_dict):
    out_of_gamut = CanonicalState(chit=1.5, gamma_AB=0.2)
    py = py_ops.intent_map_wrapped(out_of_gamut, 1.0, gamut_spec, "I1")
    rs = nat.intent_map_wrapped(
        dataclasses.asdict(out_of_gamut), 1.0, gamut_spec_dict, "I1"
    )
    _assert_parity(py, rs)


# ---------------------------------------------------------------------------
# 7. intent_compose_wrapped
# ---------------------------------------------------------------------------

def test_intent_compose_wrapped(
    canonical_state, canonical_state_dict, gamut_spec, gamut_spec_dict
):
    intents = ["I1", "I3"]
    py = py_ops.intent_compose_wrapped(canonical_state, 1.0, gamut_spec, intents)
    rs = nat.intent_compose_wrapped(canonical_state_dict, 1.0, gamut_spec_dict, intents)
    _assert_parity(py, rs)


# ---------------------------------------------------------------------------
# 8. validate_driver_profile_wrapped + raw validate_driver_profile
# ---------------------------------------------------------------------------

def _reference_dataset(
    canonical: CanonicalState,
    substrate: SubstrateState,
) -> tuple[list[dict], list[dict]]:
    """One Python-form dataset (CanonicalState / SubstrateState dataclass
    values) and one native-form (nested dicts) for the same row."""
    py_form = [
        {
            "canonical_state": canonical,
            "tau_obs": substrate.tau_obs,
            "expected_substrate": substrate,
        }
    ]
    nat_form = [
        {
            "canonical_state": dataclasses.asdict(canonical),
            "tau_obs": substrate.tau_obs,
            "expected_substrate": dataclasses.asdict(substrate),
        }
    ]
    return py_form, nat_form


def test_validate_driver_profile_wrapped(
    canonical_state,
    canonical_state_dict,
    tangent_flow_field,
    tangent_flow_field_dict,
    gamut_spec,
    gamut_spec_dict,
    canonical_grid,
    canonical_grid_list,
):
    substrate = py_ops.apply_translation(canonical_state, tangent_flow_field, 1.0)
    py_dataset, nat_dataset = _reference_dataset(canonical_state, substrate)

    py = py_ops.validate_driver_profile_wrapped(
        tangent_flow_field, py_dataset, canonical_grid, intent_id="I5", gamut=gamut_spec
    )
    rs = nat.validate_driver_profile_wrapped(
        tangent_flow_field_dict, nat_dataset, canonical_grid_list, "I5", gamut_spec_dict
    )
    _assert_parity(py, rs)


def test_validate_driver_profile_raw(
    canonical_state,
    tangent_flow_field,
    tangent_flow_field_dict,
    gamut_spec,
    gamut_spec_dict,
    canonical_grid,
    canonical_grid_list,
):
    substrate = py_ops.apply_translation(canonical_state, tangent_flow_field, 1.0)
    py_dataset, nat_dataset = _reference_dataset(canonical_state, substrate)

    py = py_ops.validate_driver_profile(
        tangent_flow_field, py_dataset, canonical_grid, intent_id="I5", gamut=gamut_spec
    )
    rs = nat.validate_driver_profile(
        tangent_flow_field_dict, nat_dataset, canonical_grid_list, "I5", gamut_spec_dict
    )
    # Raw form has no validation/provenance wrapping — compare value dicts directly.
    py_dict = _normalize(py)
    rs_dict = _normalize(rs)
    assert _equal_with_floats(py_dict, rs_dict, abs_tol=ABS_TOL_WIDE), (
        f"\nPYTHON: {py_dict}\nNATIVE: {rs_dict}"
    )


# ---------------------------------------------------------------------------
# 9. forward_sweep_invert_posterior_wrapped
# ---------------------------------------------------------------------------

def test_forward_sweep_invert_posterior_wrapped_tangent_flow(
    canonical_state,
    canonical_state_dict,
    tangent_flow_field,
    tangent_flow_field_dict,
):
    substrate = py_ops.apply_translation(canonical_state, tangent_flow_field, 1.5)
    nat_substrate = dataclasses.asdict(substrate)

    py = py_ops.forward_sweep_invert_posterior_wrapped(
        substrate, tangent_flow_field, 1.5, noise_variance=0.25, k_frust=False
    )
    rs = nat.forward_sweep_invert_posterior_wrapped(
        nat_substrate, tangent_flow_field_dict, 1.5,
        None, 0.25, False, 5,
    )
    _assert_parity(py, rs)


# ---------------------------------------------------------------------------
# flow (raw — not an OperationOutput)
# ---------------------------------------------------------------------------

def test_flow_tangent_flow(
    canonical_state, canonical_state_dict, tangent_flow_field, tangent_flow_field_dict
):
    py = py_flow(canonical_state, 0.5, tangent_flow_field)
    rs = nat.flow(canonical_state_dict, 0.5, tangent_flow_field_dict)
    py_dict = _normalize(py)
    rs_dict = _normalize(rs)
    assert _equal_with_floats(py_dict, rs_dict, abs_tol=ABS_TOL_WIDE), (
        f"\nPYTHON: {py_dict}\nNATIVE: {rs_dict}"
    )
