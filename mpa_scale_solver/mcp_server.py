"""MCP server exposing the scale-solver as JSON tools (v3 — BLOCK_IN §v3).

Stateless. Read-only over driver profiles (no write surface). stdio
transport (the BLOCK_IN-default; matches the broader MCP convention).

Tools exposed (11):

Core seven (RFC-S surface):
  - apply_translation
  - forward_sweep_invert
  - tau_obs_sweep
  - regime_at
  - gamut_classify
  - intent_map
  - validate_driver_profile

Cross-substrate compositions (v3):
  - gamut_overlap
  - canonical_distance
  - universality_agreement

Active learning (v3):
  - suggest_measurements

Thin discipline: one dispatch function per tool, hardcoded schemas,
no framework gymnastics. Inputs are JSON-shape (lists / dicts /
primitives); the dispatch coerces them into the dataclasses the
operations consume. Outputs are JSON-serialized via the
`_to_jsonable` helper.

Run via the `mpa-scale-solver-mcp` console script (declared in
pyproject) or `python -m mpa_scale_solver.mcp_server`.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from typing import Any

import numpy as np

import mcp.types as mcp_types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from . import (
    CanonicalState,
    GamutSpec,
    SubstrateState,
    apply_translation,
    forward_sweep_invert,
    tau_obs_sweep,
    regime_at,
    gamut_classify,
    intent_map,
    validate_driver_profile,
    parse_translation_field,
    parse_gamut,
    __version__,
)
from .cross_substrate import (
    canonical_distance,
    gamut_overlap,
    universality_agreement,
)
from .active_learning import suggest_measurements


# ---------------------------------------------------------------------------
# JSON coercion helpers
# ---------------------------------------------------------------------------


def _to_jsonable(x: Any) -> Any:
    """Recursively coerce dataclasses, numpy types, tuples, sets to JSON shape."""
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return {k: _to_jsonable(v) for k, v in dataclasses.asdict(x).items()}
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def _canonical_state(d: dict) -> CanonicalState:
    return CanonicalState(
        chit=float(d["chit"]),
        gamma_AB=float(d["gamma_AB"]),
        k_frust=bool(d.get("k_frust", False)),
    )


def _substrate_state(d: dict) -> SubstrateState:
    return SubstrateState(
        tau_obs=float(d["tau_obs"]),
        label=d.get("label"),
        axes=dict(d.get("axes", {})),
        observables={k: float(v) for k, v in d.get("observables", {}).items()},
    )


def _grid_2d(g: list) -> np.ndarray:
    arr = np.asarray(g, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"expected (N, 2) array, got shape {arr.shape}")
    return arr


# ---------------------------------------------------------------------------
# Tool schemas (JSON-Schema fragments) and handlers
# ---------------------------------------------------------------------------


_CANONICAL_STATE_SCHEMA = {
    "type": "object",
    "properties": {
        "chit": {"type": "number"},
        "gamma_AB": {"type": "number"},
        "k_frust": {"type": "boolean"},
    },
    "required": ["chit", "gamma_AB"],
}

_GAMUT_SCHEMA = {
    "type": "object",
    "properties": {
        "chit_range": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
        "gamma_AB_range": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
        "tau_obs_range": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
    },
    "required": ["chit_range", "gamma_AB_range"],
}

_TRANSLATION_FIELD_SCHEMA = {
    "type": "object",
    "description": "driver-profile.v2.0+ translation_field block (lookup_table or learned)",
    "additionalProperties": True,
}


def _tool_definitions() -> list[mcp_types.Tool]:
    return [
        mcp_types.Tool(
            name="apply_translation",
            description="Forward map: canonical state -> substrate at tau_obs.",
            inputSchema={
                "type": "object",
                "properties": {
                    "canonical": _CANONICAL_STATE_SCHEMA,
                    "translation_field": _TRANSLATION_FIELD_SCHEMA,
                    "tau_obs": {"type": "number"},
                },
                "required": ["canonical", "translation_field", "tau_obs"],
            },
        ),
        mcp_types.Tool(
            name="forward_sweep_invert",
            description="Substrate observation -> canonical state at tau_obs, via grid search.",
            inputSchema={
                "type": "object",
                "properties": {
                    "target_substrate": {"type": "object"},
                    "translation_field": _TRANSLATION_FIELD_SCHEMA,
                    "tau_obs": {"type": "number"},
                    "canonical_grid": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                    },
                },
                "required": ["target_substrate", "translation_field", "tau_obs", "canonical_grid"],
            },
        ),
        mcp_types.Tool(
            name="tau_obs_sweep",
            description="Walk the RG-flow trajectory across tau_obs (per-frame fan-out).",
            inputSchema={
                "type": "object",
                "properties": {
                    "target_substrate": {"type": "object"},
                    "translation_field": _TRANSLATION_FIELD_SCHEMA,
                    "tau_obs_grid": {"type": "array", "items": {"type": "number"}},
                    "canonical_search_grid": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                    },
                },
                "required": ["target_substrate", "translation_field", "tau_obs_grid", "canonical_search_grid"],
            },
        ),
        mcp_types.Tool(
            name="regime_at",
            description="5-bucket vertex-regime classifier at the canonical state.",
            inputSchema={
                "type": "object",
                "properties": {
                    "canonical": _CANONICAL_STATE_SCHEMA,
                    "tau_obs": {"type": "number"},
                },
                "required": ["canonical", "tau_obs"],
            },
        ),
        mcp_types.Tool(
            name="gamut_classify",
            description="In-gamut / out-of-gamut diagnosis at the canonical state.",
            inputSchema={
                "type": "object",
                "properties": {
                    "canonical": _CANONICAL_STATE_SCHEMA,
                    "tau_obs": {"type": "number"},
                    "gamut": _GAMUT_SCHEMA,
                },
                "required": ["canonical", "tau_obs", "gamut"],
            },
        ),
        mcp_types.Tool(
            name="intent_map",
            description="Map an out-of-gamut state to in-gamut preserving the named invariant (I1-I5).",
            inputSchema={
                "type": "object",
                "properties": {
                    "canonical": _CANONICAL_STATE_SCHEMA,
                    "tau_obs": {"type": "number"},
                    "gamut": _GAMUT_SCHEMA,
                    "intent_id": {"type": "string", "enum": ["I1", "I2", "I3", "I4", "I5"]},
                },
                "required": ["canonical", "tau_obs", "gamut", "intent_id"],
            },
        ),
        mcp_types.Tool(
            name="validate_driver_profile",
            description="RFC-S §5 round-trip validation with per-intent metrics.",
            inputSchema={
                "type": "object",
                "properties": {
                    "translation_field": _TRANSLATION_FIELD_SCHEMA,
                    "reference_dataset": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "canonical_state": _CANONICAL_STATE_SCHEMA,
                                "tau_obs": {"type": "number"},
                                "expected_substrate": {"type": "object"},
                            },
                            "required": ["canonical_state", "tau_obs"],
                        },
                    },
                    "canonical_search_grid": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                    },
                    "intent_id": {"type": "string", "enum": ["I1", "I2", "I3", "I4", "I5"]},
                    "gamut": _GAMUT_SCHEMA,
                },
                "required": ["translation_field", "reference_dataset", "canonical_search_grid"],
            },
        ),
        mcp_types.Tool(
            name="gamut_overlap",
            description="Intersection of two gamuts in canonical (+ optional tau_obs) space.",
            inputSchema={
                "type": "object",
                "properties": {
                    "gamut_a": _GAMUT_SCHEMA,
                    "gamut_b": _GAMUT_SCHEMA,
                },
                "required": ["gamut_a", "gamut_b"],
            },
        ),
        mcp_types.Tool(
            name="canonical_distance",
            description="Distance between two canonical states under a named metric (l2/l1/regime/universality).",
            inputSchema={
                "type": "object",
                "properties": {
                    "state_a": _CANONICAL_STATE_SCHEMA,
                    "state_b": _CANONICAL_STATE_SCHEMA,
                    "metric": {"type": "string", "enum": ["l2", "l1", "regime", "universality"]},
                },
                "required": ["state_a", "state_b"],
            },
        ),
        mcp_types.Tool(
            name="universality_agreement",
            description="Cross-substrate universality test (RFC-S §3 I5 metric).",
            inputSchema={
                "type": "object",
                "properties": {
                    "profile_a_field": _TRANSLATION_FIELD_SCHEMA,
                    "profile_a_gamut": _GAMUT_SCHEMA,
                    "profile_b_field": _TRANSLATION_FIELD_SCHEMA,
                    "profile_b_gamut": _GAMUT_SCHEMA,
                    "canonical_grid": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                    },
                    "tau_obs": {"type": "number"},
                },
                "required": [
                    "profile_a_field", "profile_a_gamut",
                    "profile_b_field", "profile_b_gamut",
                    "canonical_grid", "tau_obs",
                ],
            },
        ),
        mcp_types.Tool(
            name="suggest_measurements",
            description="Active learning: suggest n canonical-space candidates ranked by composite informativeness.",
            inputSchema={
                "type": "object",
                "properties": {
                    "translation_field": _TRANSLATION_FIELD_SCHEMA,
                    "gamut": _GAMUT_SCHEMA,
                    "canonical_grid": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                    },
                    "tau_obs": {"type": "number"},
                    "n": {"type": "integer", "minimum": 1, "default": 5},
                },
                "required": ["translation_field", "gamut", "canonical_grid", "tau_obs"],
            },
        ),
    ]


def dispatch_tool(name: str, arguments: dict) -> Any:
    """Synchronous dispatch table: name + JSON args -> JSON-able result.

    Exposed at module level (not nested inside the async handler) so
    tests can drive the dispatch without running the full MCP stdio
    transport.
    """
    if name == "apply_translation":
        result = apply_translation(
            _canonical_state(arguments["canonical"]),
            parse_translation_field(arguments["translation_field"]),
            float(arguments["tau_obs"]),
        )
        return _to_jsonable(result)

    if name == "forward_sweep_invert":
        recovered, residual = forward_sweep_invert(
            _substrate_state(arguments["target_substrate"]),
            parse_translation_field(arguments["translation_field"]),
            float(arguments["tau_obs"]),
            _grid_2d(arguments["canonical_grid"]),
        )
        return {"recovered": _to_jsonable(recovered), "residual": float(residual)}

    if name == "tau_obs_sweep":
        trajectory = tau_obs_sweep(
            _substrate_state(arguments["target_substrate"]),
            parse_translation_field(arguments["translation_field"]),
            np.asarray(arguments["tau_obs_grid"], dtype=np.float64),
            _grid_2d(arguments["canonical_search_grid"]),
        )
        return {"trajectory": [_to_jsonable(s) for s in trajectory]}

    if name == "regime_at":
        reading = regime_at(
            _canonical_state(arguments["canonical"]),
            float(arguments["tau_obs"]),
        )
        return _to_jsonable(reading)

    if name == "gamut_classify":
        result = gamut_classify(
            _canonical_state(arguments["canonical"]),
            float(arguments["tau_obs"]),
            parse_gamut(arguments["gamut"]),
        )
        return _to_jsonable(result)

    if name == "intent_map":
        mapped, sacrifice = intent_map(
            _canonical_state(arguments["canonical"]),
            float(arguments["tau_obs"]),
            parse_gamut(arguments["gamut"]),
            str(arguments["intent_id"]),
        )
        return {"mapped": _to_jsonable(mapped), "sacrifice": _to_jsonable(sacrifice)}

    if name == "validate_driver_profile":
        gamut = parse_gamut(arguments["gamut"]) if "gamut" in arguments else None
        dataset = [
            {
                "canonical_state": _canonical_state(e["canonical_state"]),
                "tau_obs": float(e["tau_obs"]),
                "expected_substrate": (
                    _substrate_state(e["expected_substrate"])
                    if "expected_substrate" in e else None
                ),
            }
            for e in arguments["reference_dataset"]
        ]
        summary = validate_driver_profile(
            parse_translation_field(arguments["translation_field"]),
            dataset,
            _grid_2d(arguments["canonical_search_grid"]),
            intent_id=str(arguments.get("intent_id", "I5")),
            gamut=gamut,
        )
        return _to_jsonable(summary)

    if name == "gamut_overlap":
        return _to_jsonable(gamut_overlap(
            parse_gamut(arguments["gamut_a"]),
            parse_gamut(arguments["gamut_b"]),
        ))

    if name == "canonical_distance":
        return float(canonical_distance(
            _canonical_state(arguments["state_a"]),
            _canonical_state(arguments["state_b"]),
            str(arguments.get("metric", "l2")),
        ))

    if name == "universality_agreement":
        return _to_jsonable(universality_agreement(
            parse_translation_field(arguments["profile_a_field"]),
            parse_gamut(arguments["profile_a_gamut"]),
            parse_translation_field(arguments["profile_b_field"]),
            parse_gamut(arguments["profile_b_gamut"]),
            _grid_2d(arguments["canonical_grid"]),
            float(arguments["tau_obs"]),
        ))

    if name == "suggest_measurements":
        candidates = suggest_measurements(
            parse_translation_field(arguments["translation_field"]),
            parse_gamut(arguments["gamut"]),
            _grid_2d(arguments["canonical_grid"]),
            float(arguments["tau_obs"]),
            int(arguments.get("n", 5)),
        )
        return [_to_jsonable(c) for c in candidates]

    raise ValueError(f"unknown tool: {name!r}")


# ---------------------------------------------------------------------------
# MCP server wiring
# ---------------------------------------------------------------------------


def build_server() -> Server:
    """Build a fresh Server with the v3 tool surface registered.

    Factory function (not a module-level singleton) so tests can build
    an isolated server per call.
    """
    server: Server = Server("mpa-scale-solver", version=__version__)

    @server.list_tools()
    async def _list_tools() -> list[mcp_types.Tool]:
        return _tool_definitions()

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict) -> list[mcp_types.TextContent]:
        try:
            result = dispatch_tool(name, arguments or {})
            return [mcp_types.TextContent(type="text", text=json.dumps(result))]
        except Exception as exc:
            payload = {"error": type(exc).__name__, "message": str(exc)}
            return [mcp_types.TextContent(type="text", text=json.dumps(payload))]

    return server


async def _main_async() -> None:
    server = build_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream,
            server.create_initialization_options(),
        )


def main() -> None:
    """Console-script entry point."""
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
