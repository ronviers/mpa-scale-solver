"""Streaming / online inversion (v4 — BLOCK_IN §v4).

`forward_sweep_invert_stream` consumes substrate observations as an
iterator and yields canonical-state inversions per frame. State-local:
each frame is independent; no caching, smoothing, or cross-frame state
leaks. Consumers that want smoothing or cross-frame regularization
compose this stream with their own stateful aggregator.

The function is a thin generator over the v0 `forward_sweep_invert`
per frame — no new math. What it adds is the iterator shape, which
matches real-time experimental pipelines (live camera frames,
incremental observations, stdin pipelines) and the `InversionResult`
return shape (state + residual + tau_obs + frame_index).

Two thin source adapters are included:

  - `from_iterable(iterable)` — passthrough; useful for tests and
    in-memory consumers that want the same iterator shape as a live
    source.
  - `from_stdin(strict=True)` — yields `SubstrateState` parsed from
    JSON-per-line on stdin. The shape mirrors `_substrate_state` in
    `mcp_server.py`; consumers can pipe `python -m my_recorder | python
    -m my_inverter` without an in-process wiring step.

WebSocket and polling sources are deferred per the thin-discipline rule
(add them when a consumer asks); both fit the same iterator interface,
so a consumer can wrap their own source today without waiting on a
helper.

Wrapped-variant streaming (`Iterator[OperationOutput[CanonicalState]]`)
is not exposed here. Consumers that want validation + provenance per
frame call `forward_sweep_invert_wrapped` inside their own loop — the
streaming function's job is the iterator shape, not re-bundling the
existing wrapped contract.

A streaming MCP tool variant is deferred per BLOCK_IN §v4 "open / watch"
— the v3 MCP transport (one call per tool invocation) already covers
the common agentic workflow; streaming MCP lands when a consumer asks.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Optional, TextIO

import numpy as np

from .operations import forward_sweep_invert
from .types import (
    AnyTranslationField,
    CanonicalState,
    SubstrateState,
)


@dataclass(frozen=True)
class InversionResult:
    """Per-frame streaming inversion (v4 — BLOCK_IN §v4).

    Yielded by `forward_sweep_invert_stream` once per consumed
    observation. State-local: no reference to prior or subsequent
    frames. `frame_index` is the 0-based position in the consumed
    stream (useful for correlating against the source).

    Native-port note: this dataclass is intentionally flat — `state`,
    a float, a float, an int — so the v6 port reads it as a plain
    struct (no nested lists, no optional fields). v6 will produce
    the same per-frame shape under `Iterator<InversionResult>` in the
    target language.
    """

    state: CanonicalState
    residual: float
    tau_obs: float
    frame_index: int

    def _repr_html_(self) -> str:
        from .types import _html_table
        return _html_table("InversionResult", [
            ("frame_index", str(self.frame_index)),
            ("tau_obs", f"{self.tau_obs:.4g}"),
            ("state",
             f"chit={self.state.chit:.4g}, gamma_AB={self.state.gamma_AB:.4g}"),
            ("residual", f"{self.residual:.4g}"),
        ])


def forward_sweep_invert_stream(
    observations: Iterable[SubstrateState],
    field: AnyTranslationField,
    canonical_grid: np.ndarray,
    *,
    tau_obs: Optional[float] = None,
    score_fn: Optional[Callable[[SubstrateState, SubstrateState], float]] = None,
) -> Iterator[InversionResult]:
    """Stream substrate observations through forward_sweep_invert per frame.

    `tau_obs` is the constant observer-fact scale across the stream
    (the camera's τ_obs setting). When None, each observation's own
    `obs.tau_obs` field is used — useful for streams where the observer
    scale varies frame-to-frame (multi-camera scrubbing, RG-flow walks
    consumed as a stream).

    `canonical_grid` is the search grid used by the per-frame inversion.
    Required for every field shape (lookup_table / tangent_flow /
    learned) — the v5 streaming variant with gradient-based inversion
    will lift this requirement for differentiable fields per BLOCK_IN
    §v5. Until then, callers pass a grid sized to their accuracy /
    latency budget.

    `score_fn` is passed through to `forward_sweep_invert`; default is
    the L²-over-shared-numeric-keys score.

    Yields `InversionResult` per consumed observation. The generator is
    lazy: nothing is computed until the consumer pulls.
    """
    if canonical_grid.ndim != 2 or canonical_grid.shape[1] != 2:
        raise ValueError(
            f"canonical_grid must have shape (N, 2); got {canonical_grid.shape}"
        )

    for i, obs in enumerate(observations):
        frame_tau = float(tau_obs) if tau_obs is not None else float(obs.tau_obs)
        state, residual = forward_sweep_invert(
            obs, field, frame_tau, canonical_grid, score_fn=score_fn,
        )
        yield InversionResult(
            state=state,
            residual=float(residual),
            tau_obs=frame_tau,
            frame_index=i,
        )


# ---------------------------------------------------------------------------
# Thin source adapters
# ---------------------------------------------------------------------------


def from_iterable(iterable: Iterable[SubstrateState]) -> Iterator[SubstrateState]:
    """Trivial passthrough: in-memory iterable → SubstrateState iterator.

    Useful when a test or in-process consumer wants the same iterator
    shape a live source would produce.
    """
    return iter(iterable)


def from_stdin(stream: Optional[TextIO] = None, *, strict: bool = True) -> Iterator[SubstrateState]:
    """JSON-per-line stdin → SubstrateState iterator.

    Each non-empty line is parsed as JSON with the `SubstrateState`
    shape used elsewhere in the package (`mcp_server._substrate_state`):

        {"tau_obs": float,
         "label": str | null,
         "axes": {...},
         "observables": {...}}

    When `strict=True` (default), malformed JSON lines raise. When
    `strict=False`, malformed lines are skipped silently — useful for
    noisy log piping where occasional partial writes are expected.

    `stream` defaults to `sys.stdin`; injectable for tests.
    """
    src = stream if stream is not None else sys.stdin
    for line in src:
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            if strict:
                raise
            continue
        yield SubstrateState(
            tau_obs=float(d["tau_obs"]),
            label=d.get("label"),
            axes=dict(d.get("axes", {})),
            observables={k: float(v) for k, v in d.get("observables", {}).items()},
        )
