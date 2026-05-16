"""v3 — MCP server (BLOCK_IN §v3).

The MCP server exposes the seven core operations + the v3 cross-substrate
and active-learning ops as stateless JSON tools. Tests drive
`dispatch_tool` directly rather than spinning a full stdio transport;
that's the unit-level surface that matters for correctness.

Acceptance:
  - 11 tools registered (7 core + 3 cross-substrate + 1 active learning).
  - Each tool round-trips a JSON-shape call through the dispatcher.
  - Errors are returned as JSON-encoded {error, message} payloads via the
    server's call_tool handler (not raised through the MCP transport).
  - The `build_server` factory builds an isolated Server instance.
"""

from __future__ import annotations

import json

import pytest

from mpa_scale_solver.mcp_server import (
    _tool_definitions,
    build_server,
    dispatch_tool,
)


def _identity_lookup_field_json() -> dict:
    return {
        "direction": "forward",
        "shape": "lookup_table",
        "rule": [
            {
                "operating_point": {"label": "p1", "gt": "c", "tau_obs": 1.0},
                "xdot_choice": "default",
                "canonical": {
                    "chit": 0.5, "gamma_AB": 0.2,
                    "k_frust": False, "method": "lookup",
                },
            },
            {
                "operating_point": {"label": "p2", "gt": "s", "tau_obs": 1.0},
                "xdot_choice": "default",
                "canonical": {
                    "chit": 0.0, "gamma_AB": 0.0,
                    "k_frust": False, "method": "lookup",
                },
            },
            {
                "operating_point": {"label": "p3", "gt": "r", "tau_obs": 1.0},
                "xdot_choice": "default",
                "canonical": {
                    "chit": -0.5, "gamma_AB": -0.2,
                    "k_frust": False, "method": "lookup",
                },
            },
        ],
    }


def _basic_gamut_json() -> dict:
    return {
        "chit_range": [-1.0, 1.0],
        "gamma_AB_range": [-1.0, 1.0],
    }


def _grid_json() -> list[list[float]]:
    return [
        [0.5, 0.2], [0.0, 0.0], [-0.5, -0.2],
        [0.5, -0.2], [-0.5, 0.2],
    ]


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

class TestToolRegistry:
    def test_eleven_tools_registered(self):
        tools = _tool_definitions()
        names = {t.name for t in tools}
        assert names == {
            "apply_translation", "forward_sweep_invert", "tau_obs_sweep",
            "regime_at", "gamut_classify", "intent_map", "validate_driver_profile",
            "gamut_overlap", "canonical_distance", "universality_agreement",
            "suggest_measurements",
        }

    def test_each_tool_has_input_schema(self):
        for tool in _tool_definitions():
            assert tool.inputSchema is not None
            assert tool.inputSchema.get("type") == "object"
            assert "properties" in tool.inputSchema

    def test_build_server_returns_server(self):
        server = build_server()
        assert server.name == "mpa-scale-solver"


# ---------------------------------------------------------------------------
# Core seven (dispatch_tool round-trip)
# ---------------------------------------------------------------------------

class TestCoreSeven:
    def test_apply_translation_dispatch(self):
        result = dispatch_tool("apply_translation", {
            "canonical": {"chit": 0.5, "gamma_AB": 0.2, "k_frust": False},
            "translation_field": _identity_lookup_field_json(),
            "tau_obs": 1.0,
        })
        assert result["label"] == "p1"
        # JSON serializable
        json.dumps(result)

    def test_forward_sweep_invert_dispatch(self):
        target = {
            "tau_obs": 1.0,
            "label": "p1",
            "axes": {"label": "p1", "gt": "c", "tau_obs": 1.0},
            "observables": {"canonical_chit": 0.5, "canonical_gamma_AB": 0.2},
        }
        result = dispatch_tool("forward_sweep_invert", {
            "target_substrate": target,
            "translation_field": _identity_lookup_field_json(),
            "tau_obs": 1.0,
            "canonical_grid": _grid_json(),
        })
        assert "recovered" in result
        assert result["recovered"]["chit"] == 0.5
        assert result["recovered"]["gamma_AB"] == 0.2

    def test_tau_obs_sweep_dispatch(self):
        result = dispatch_tool("tau_obs_sweep", {
            "target_substrate": {
                "tau_obs": 1.0,
                "label": "p2",
                "axes": {"label": "p2", "gt": "s", "tau_obs": 1.0},
                "observables": {"canonical_chit": 0.0, "canonical_gamma_AB": 0.0},
            },
            "translation_field": _identity_lookup_field_json(),
            "tau_obs_grid": [0.5, 1.0, 2.0],
            "canonical_search_grid": _grid_json(),
        })
        assert "trajectory" in result
        assert len(result["trajectory"]) == 3

    def test_regime_at_dispatch(self):
        result = dispatch_tool("regime_at", {
            "canonical": {"chit": 0.8, "gamma_AB": 0.0},
            "tau_obs": 1.0,
        })
        assert result["regime"] == "deep_c"

    def test_gamut_classify_dispatch(self):
        result = dispatch_tool("gamut_classify", {
            "canonical": {"chit": 0.5, "gamma_AB": 0.5},
            "tau_obs": 1.0,
            "gamut": _basic_gamut_json(),
        })
        assert result["in_gamut"] is True

    def test_intent_map_dispatch(self):
        result = dispatch_tool("intent_map", {
            "canonical": {"chit": 2.0, "gamma_AB": 0.0},
            "tau_obs": 1.0,
            "gamut": _basic_gamut_json(),
            "intent_id": "I5",
        })
        assert "mapped" in result
        assert "sacrifice" in result
        assert result["mapped"]["chit"] == 1.0  # clamped to gamut max

    def test_validate_driver_profile_dispatch(self):
        result = dispatch_tool("validate_driver_profile", {
            "translation_field": _identity_lookup_field_json(),
            "reference_dataset": [
                {
                    "canonical_state": {"chit": 0.5, "gamma_AB": 0.2},
                    "tau_obs": 1.0,
                },
            ],
            "canonical_search_grid": _grid_json(),
            "intent_id": "I5",
        })
        assert "per_intent" in result
        assert result["per_intent"]["intent"] == "I5"


# ---------------------------------------------------------------------------
# Cross-substrate dispatch
# ---------------------------------------------------------------------------

class TestCrossSubstrateDispatch:
    def test_gamut_overlap_dispatch(self):
        result = dispatch_tool("gamut_overlap", {
            "gamut_a": _basic_gamut_json(),
            "gamut_b": _basic_gamut_json(),
        })
        assert result["jaccard"] == 1.0
        assert result["compatible"] is True

    def test_canonical_distance_dispatch(self):
        result = dispatch_tool("canonical_distance", {
            "state_a": {"chit": 0.0, "gamma_AB": 0.0},
            "state_b": {"chit": 3.0, "gamma_AB": 4.0},
            "metric": "l2",
        })
        assert result == pytest.approx(5.0)

    def test_canonical_distance_default_metric_is_l2(self):
        result = dispatch_tool("canonical_distance", {
            "state_a": {"chit": 0.0, "gamma_AB": 0.0},
            "state_b": {"chit": 1.0, "gamma_AB": 0.0},
        })
        assert result == pytest.approx(1.0)

    def test_universality_agreement_dispatch(self):
        result = dispatch_tool("universality_agreement", {
            "profile_a_field": _identity_lookup_field_json(),
            "profile_a_gamut": _basic_gamut_json(),
            "profile_b_field": _identity_lookup_field_json(),
            "profile_b_gamut": _basic_gamut_json(),
            "canonical_grid": _grid_json(),
            "tau_obs": 1.0,
        })
        assert result["agreement_rate"] == 1.0


# ---------------------------------------------------------------------------
# Active learning dispatch
# ---------------------------------------------------------------------------

class TestActiveLearningDispatch:
    def test_suggest_measurements_dispatch(self):
        result = dispatch_tool("suggest_measurements", {
            "translation_field": _identity_lookup_field_json(),
            "gamut": _basic_gamut_json(),
            "canonical_grid": _grid_json(),
            "tau_obs": 1.0,
            "n": 3,
        })
        assert isinstance(result, list)
        assert len(result) <= 3
        if result:
            # Each candidate is a JSON dict (from dataclasses.asdict).
            assert "state" in result[0]
            assert "score" in result[0]
            assert "components" in result[0]


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_unknown_tool_raises(self):
        with pytest.raises(ValueError, match="unknown tool"):
            dispatch_tool("not_a_real_tool", {})

    def test_unknown_intent_raises(self):
        with pytest.raises(ValueError, match="unknown intent"):
            dispatch_tool("intent_map", {
                "canonical": {"chit": 0.0, "gamma_AB": 0.0},
                "tau_obs": 1.0,
                "gamut": _basic_gamut_json(),
                "intent_id": "I99",
            })
