"""v4 streaming inversion (BLOCK_IN §v4).

Coverage:
  - per-frame InversionResult shape (frame_index, tau_obs, residual)
  - state-locality: each frame's inversion independent of others
  - tau_obs argument modes: fixed across stream vs per-frame from obs
  - lazy generator (no work until consumed)
  - from_iterable / from_stdin adapters
  - shape errors surface from the underlying forward_sweep_invert
"""

from __future__ import annotations

import io
import json

import numpy as np
import pytest

from mpa_scale_solver import (
    CanonicalState,
    InversionResult,
    SubstrateState,
    forward_sweep_invert_stream,
    from_iterable,
    from_stdin,
)
from mpa_scale_solver.symbolic_query import QueryParseError  # noqa: F401 — sanity

# Re-use the lookup-field helper shape from operations tests.
from mpa_scale_solver import (
    CanonicalPoint,
    OperatingPoint,
    TranslationField,
    TranslationRule,
)


def _three_cell_field() -> TranslationField:
    rules = []
    for chit in (-1.0, 0.0, 1.0):
        rules.append(TranslationRule(
            operating_point=OperatingPoint(
                label=f"chit={int(chit):+d}",
                gt="r" if chit < 0 else ("c" if chit > 0 else "s"),
                axes={"chit_label": chit},
            ),
            xdot_choice="x",
            canonical=CanonicalPoint(
                chit=chit, gamma_AB=0.0, k_frust=False, method="test",
            ),
        ))
    return TranslationField(direction="forward", shape="lookup_table", rule=rules)


def _grid() -> np.ndarray:
    return np.array([[c, 0.0] for c in [-1.0, 0.0, 1.0]])


# ---------------------------------------------------------------------------
# Core: forward_sweep_invert_stream
# ---------------------------------------------------------------------------


class TestStreamCore:
    def test_yields_inversion_result_per_observation(self):
        field = _three_cell_field()
        obs = [
            SubstrateState(tau_obs=1.0, axes={"chit_label": -1.0}),
            SubstrateState(tau_obs=1.0, axes={"chit_label": 0.0}),
            SubstrateState(tau_obs=1.0, axes={"chit_label": 1.0}),
        ]
        results = list(forward_sweep_invert_stream(obs, field, _grid(), tau_obs=1.0))
        assert len(results) == 3
        assert all(isinstance(r, InversionResult) for r in results)

    def test_recovers_table_canonical_per_frame(self):
        field = _three_cell_field()
        obs = [
            SubstrateState(tau_obs=1.0, axes={"chit_label": -1.0}),
            SubstrateState(tau_obs=1.0, axes={"chit_label": 0.0}),
            SubstrateState(tau_obs=1.0, axes={"chit_label": 1.0}),
        ]
        results = list(forward_sweep_invert_stream(obs, field, _grid(), tau_obs=1.0))
        assert results[0].state.chit == pytest.approx(-1.0)
        assert results[1].state.chit == pytest.approx(0.0)
        assert results[2].state.chit == pytest.approx(1.0)

    def test_frame_index_monotonic_from_zero(self):
        field = _three_cell_field()
        obs = [SubstrateState(tau_obs=1.0, axes={"chit_label": 0.0})] * 5
        results = list(forward_sweep_invert_stream(obs, field, _grid(), tau_obs=1.0))
        assert [r.frame_index for r in results] == [0, 1, 2, 3, 4]

    def test_residual_populated(self):
        field = _three_cell_field()
        obs = [SubstrateState(tau_obs=1.0, axes={"chit_label": 1.0})]
        results = list(forward_sweep_invert_stream(obs, field, _grid(), tau_obs=1.0))
        # Exact match — residual is zero.
        assert results[0].residual == pytest.approx(0.0)

    def test_state_local_no_cross_frame_leakage(self):
        """Running with a longer prefix must not change later frames."""
        field = _three_cell_field()
        obs = [
            SubstrateState(tau_obs=1.0, axes={"chit_label": c})
            for c in [-1.0, 0.0, 1.0]
        ]
        full = list(forward_sweep_invert_stream(obs, field, _grid(), tau_obs=1.0))
        tail = list(forward_sweep_invert_stream(obs[1:], field, _grid(), tau_obs=1.0))
        # Same logical observations -> same recovered state, regardless of
        # prefix. (frame_index will differ — by design.)
        assert full[1].state.chit == tail[0].state.chit
        assert full[2].state.chit == tail[1].state.chit

    def test_tau_obs_constant_across_stream(self):
        field = _three_cell_field()
        # Observation says tau_obs=999, but stream tau_obs=5.0 overrides.
        obs = [SubstrateState(tau_obs=999.0, axes={"chit_label": 0.0})]
        results = list(forward_sweep_invert_stream(obs, field, _grid(), tau_obs=5.0))
        assert results[0].tau_obs == pytest.approx(5.0)

    def test_tau_obs_per_frame_when_none(self):
        field = _three_cell_field()
        obs = [
            SubstrateState(tau_obs=0.5, axes={"chit_label": 0.0}),
            SubstrateState(tau_obs=2.0, axes={"chit_label": 0.0}),
        ]
        results = list(forward_sweep_invert_stream(obs, field, _grid(), tau_obs=None))
        assert results[0].tau_obs == pytest.approx(0.5)
        assert results[1].tau_obs == pytest.approx(2.0)

    def test_generator_is_lazy(self):
        """forward_sweep_invert_stream returns a generator that does no
        work until pulled."""
        field = _three_cell_field()

        def exploding_source():
            raise AssertionError("source should not be consumed")
            yield  # pragma: no cover

        gen = forward_sweep_invert_stream(
            exploding_source(), field, _grid(), tau_obs=1.0,
        )
        # Generator created but no exception yet — proves lazy.
        assert gen is not None

    def test_bad_grid_shape_raises(self):
        field = _three_cell_field()
        obs = [SubstrateState(tau_obs=1.0, axes={"chit_label": 0.0})]
        with pytest.raises(ValueError, match="shape"):
            list(forward_sweep_invert_stream(
                obs, field, np.array([1.0, 2.0]), tau_obs=1.0,
            ))


# ---------------------------------------------------------------------------
# Adapters: from_iterable / from_stdin
# ---------------------------------------------------------------------------


class TestAdapters:
    def test_from_iterable_passthrough(self):
        obs = [SubstrateState(tau_obs=1.0), SubstrateState(tau_obs=2.0)]
        out = list(from_iterable(obs))
        assert out == obs

    def test_from_stdin_parses_json_per_line(self):
        text = (
            json.dumps({"tau_obs": 1.0, "label": "a", "axes": {"chit_label": -1.0}})
            + "\n"
            + json.dumps({"tau_obs": 2.0, "label": "b", "observables": {"x": 0.5}})
            + "\n"
        )
        out = list(from_stdin(io.StringIO(text)))
        assert len(out) == 2
        assert out[0].tau_obs == 1.0 and out[0].label == "a"
        assert out[0].axes == {"chit_label": -1.0}
        assert out[1].observables == {"x": 0.5}

    def test_from_stdin_skips_blank_lines(self):
        text = (
            "\n"
            + json.dumps({"tau_obs": 1.0}) + "\n"
            + "\n"
        )
        out = list(from_stdin(io.StringIO(text)))
        assert len(out) == 1

    def test_from_stdin_strict_raises_on_bad_json(self):
        with pytest.raises(json.JSONDecodeError):
            list(from_stdin(io.StringIO("not json\n"), strict=True))

    def test_from_stdin_lax_skips_bad_json(self):
        text = (
            json.dumps({"tau_obs": 1.0}) + "\n"
            + "garbage\n"
            + json.dumps({"tau_obs": 2.0}) + "\n"
        )
        out = list(from_stdin(io.StringIO(text), strict=False))
        assert len(out) == 2


# ---------------------------------------------------------------------------
# End-to-end pipeline: from_iterable -> stream -> consume
# ---------------------------------------------------------------------------


class TestPipeline:
    def test_iterable_to_stream_to_recovered_trajectory(self):
        field = _three_cell_field()
        obs_seq = [
            SubstrateState(tau_obs=1.0, axes={"chit_label": -1.0}),
            SubstrateState(tau_obs=1.0, axes={"chit_label": 0.0}),
            SubstrateState(tau_obs=1.0, axes={"chit_label": 1.0}),
        ]
        recovered = [
            r.state
            for r in forward_sweep_invert_stream(
                from_iterable(obs_seq), field, _grid(), tau_obs=1.0,
            )
        ]
        chits = [s.chit for s in recovered]
        assert chits == pytest.approx([-1.0, 0.0, 1.0])
