"""Byte-identical fixture regression (handoff §D.2).

Fixtures live under `tests/fixtures/` and are produced by other tests
(e.g. test_camera_migration writes `tests/fixtures/camera/frame_NNNN.json`).
This module locks the recorded values: a re-run of the operations against
the same inputs must produce the same outputs.

The first run of the camera test populates the fixtures. Subsequent runs
compare against them. Changing the math intentionally → bump
mpa_scale_solver.__version__ and regenerate the fixtures with a commit
note explaining the change.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mpa_scale_solver import CanonicalState
from mpa_scale_solver._test_fixtures import (
    AgingLogParams,
    analytical_canonical_chit,
    aging_log_forward,
)


def _load_frames(camera_dir: Path) -> list[dict]:
    frames = []
    for p in sorted(camera_dir.glob("frame_*.json")):
        frames.append(json.loads(p.read_text(encoding="utf-8")))
    return frames


def test_camera_frame_fixtures_match_analytical(fixtures_dir: Path):
    """Each recorded frame's analytical_chit reproduces from the closed form."""
    camera_dir = fixtures_dir / "camera"
    if not camera_dir.exists() or not list(camera_dir.glob("frame_*.json")):
        pytest.skip(
            "no camera fixtures yet; run test_camera_migration first to populate"
        )
    frames = _load_frames(camera_dir)
    assert len(frames) > 0

    # The camera test's parameters are pinned here for the regression.
    params = AgingLogParams(chit_aging_coeff=1.0, tau_aging=1.0, gamma_aging_coeff=0.0)
    # Reference substrate chit = CHIT_REF + a*log(1 + TAU_OBS_REF/tau_aging)
    substrate_chit = 2.0 + 1.0 * np.log1p(1.0 / 1.0)  # = 2.0 + log(2)

    for fx in frames:
        expected = analytical_canonical_chit(substrate_chit, fx["tau_obs"], params)
        assert fx["analytical_chit"] == pytest.approx(expected, abs=1e-12), (
            f"frame {fx['frame_index']}: recorded analytical {fx['analytical_chit']} "
            f"!= recomputed {expected}"
        )


def test_aging_log_forward_round_trip():
    """The fixture's forward map and analytical inverse compose to identity."""
    params = AgingLogParams(chit_aging_coeff=1.3, tau_aging=2.0)
    c = CanonicalState(chit=0.7, gamma_AB=-0.2)
    for tau in (0.01, 0.1, 1.0, 10.0, 100.0):
        s = aging_log_forward(c, tau, params)
        recovered = analytical_canonical_chit(
            s.observables["substrate_chit"], tau, params,
        )
        assert recovered == pytest.approx(c.chit, abs=1e-12)
