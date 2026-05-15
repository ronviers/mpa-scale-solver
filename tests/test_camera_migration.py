"""Camera migration visual test (handoff §D.2 / §E acceptance criterion).

Builds a synthetic substrate signal whose window-averaged observable, viewed
through a tau_obs camera sweep, traces a c -> s -> r migration in canonical
chit. The analytical truth comes from the aging_log closed form; the solver
recovers canonical chit at each frame via forward_sweep_invert and we plot
both curves overlaid.

Per handoff §C.2 step 3 the test exercises the production code path through
a lookup-form TranslationField sampled from the analytical aging_log.

Pass criterion: max |analytical - numerical| <= 0.05 across all frames.

Outputs (gitignored):
  tests/out/migration_compare.png
  tests/out/result.json
  tests/fixtures/camera/frame_NNNN.json  (per-frame canonical regression fixtures)
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from mpa_scale_solver import (
    CanonicalState,
    apply_translation,
    forward_sweep_invert,
    regime_at,
    regime_display_band,
)
from mpa_scale_solver._test_fixtures import (
    AgingLogParams,
    aging_log_forward,
    analytical_canonical_chit,
    make_aging_log_lookup_table,
)
from mpa_scale_solver.substrate_signal import (
    AgingSignalParams,
    window_average_at_tau_obs,
)


# Pass tolerance per handoff §E item 4: max |residual| <= 0.05
TOLERANCE = 0.05

# Reference operating point
CHIT_REF = 2.0
GAMMA_REF = -0.5
TAU_OBS_REF = 1.0

# tau_obs sweep
TAU_OBS_GRID = np.logspace(-2, 2, 80)

# aging_log parameters
PARAMS = AgingLogParams(chit_aging_coeff=1.0, tau_aging=1.0, gamma_aging_coeff=0.0)


def _build_test_assets():
    """Construct the canonical sweep + reference substrate + analytical truth."""
    # The synthetic substrate signal: at every frame the windowed observable
    # equals the aging_log forward-projection of the reference canonical at
    # the reference tau_obs.
    ref_substrate = aging_log_forward(
        CanonicalState(chit=CHIT_REF, gamma_AB=GAMMA_REF), TAU_OBS_REF, PARAMS,
    )
    signal = AgingSignalParams(
        substrate_chit=ref_substrate.observables["substrate_chit"],
        substrate_gamma_AB=ref_substrate.observables["substrate_gamma_AB"],
    )

    # Per-frame target substrates (handoff §C.1: window-average is per-frame)
    targets = [window_average_at_tau_obs(signal, float(t)) for t in TAU_OBS_GRID]

    # Analytical truth per frame
    analytical_chit = np.array([
        analytical_canonical_chit(signal.substrate_chit, float(t), PARAMS)
        for t in TAU_OBS_GRID
    ])
    return signal, targets, analytical_chit


def run_camera_test():
    """Return per-frame data + pass/fail. Pure compute; no file I/O."""
    signal, targets, analytical = _build_test_assets()

    # The "production-shape" lookup table — handoff §C.2 step 3.
    # Built dense in chit, with one tau slice per frame so each frame's
    # forward map has exact-match coverage. Step 0.025 → half-step 0.0125,
    # safely under the 0.05 acceptance tolerance.
    chit_grid_for_table = np.linspace(-3.0, 3.0, 241)  # step 0.025
    gamma_grid_for_table = np.array([GAMMA_REF])
    field = make_aging_log_lookup_table(
        chit_grid=chit_grid_for_table,
        gamma_AB_grid=gamma_grid_for_table,
        tau_obs_grid=TAU_OBS_GRID,
        params=PARAMS,
    )

    # Candidate canonical grid for the inversion — aligned to the table
    # so each table chit is itself a candidate; recovery is exact at the
    # table resolution.
    search_grid = np.column_stack([
        chit_grid_for_table, np.full_like(chit_grid_for_table, GAMMA_REF),
    ])

    # Inversion per frame
    numerical_chit = np.empty(len(TAU_OBS_GRID))
    residual_frames: list[np.ndarray] = []
    for i, tau in enumerate(TAU_OBS_GRID):
        rec, _, resf = forward_sweep_invert(
            targets[i], field, float(tau), search_grid,
            return_residual_field=True,
        )
        numerical_chit[i] = rec.chit
        residual_frames.append(resf)

    residuals = numerical_chit - analytical
    max_abs_residual = float(np.max(np.abs(residuals)))
    passes = max_abs_residual <= TOLERANCE

    regimes_5 = [regime_at(CanonicalState(chit=float(a), gamma_AB=GAMMA_REF), float(t)).regime
                 for a, t in zip(analytical, TAU_OBS_GRID)]
    regimes_3 = [regime_display_band(r) for r in regimes_5]

    return {
        "signal_substrate_chit": signal.substrate_chit,
        "tau_obs_grid": TAU_OBS_GRID.tolist(),
        "analytical_chit": analytical.tolist(),
        "numerical_chit": numerical_chit.tolist(),
        "residuals": residuals.tolist(),
        "max_abs_residual": max_abs_residual,
        "tolerance": TOLERANCE,
        "passes": passes,
        "regimes_5": regimes_5,
        "regimes_3": regimes_3,
    }


def _render_plot(result: dict, out_path: Path) -> None:
    """Static comparison plot. Skipped silently if matplotlib unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover
        return

    log_tau = np.log10(result["tau_obs_grid"])
    analytical = np.array(result["analytical_chit"])
    numerical = np.array(result["numerical_chit"])

    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=120)
    s_width = 0.2  # boundary between c_near_s/s_critical
    chit_min = min(analytical.min(), numerical.min(), -2.5)
    chit_max = max(analytical.max(), numerical.max(), 3.0)

    ax.axhspan(s_width, chit_max, color=(0.85, 1.00, 0.85), alpha=0.55, zorder=0)
    ax.axhspan(-s_width, s_width, color=(1.00, 0.95, 0.80), alpha=0.55, zorder=0)
    ax.axhspan(chit_min, -s_width, color=(1.00, 0.85, 0.85), alpha=0.55, zorder=0)

    ax.plot(log_tau, analytical, color=(0.20, 0.40, 0.85), linewidth=4.0,
            alpha=0.85, label="analytical (closed-form truth)", zorder=3)
    ax.plot(log_tau, numerical, color=(0.85, 0.20, 0.20), linewidth=1.5,
            linestyle="--", marker="o", markersize=4,
            label="numerical (forward_sweep_invert)", zorder=4)

    ax.set_xlim(log_tau.min(), log_tau.max())
    ax.set_ylim(chit_min, chit_max)
    ax.set_xlabel(r"$\log_{10}(\tau_{obs})$", fontsize=13)
    ax.set_ylabel(r"canonical $\chi$ (chit)", fontsize=13)
    ax.grid(True, alpha=0.25, linestyle=":", zorder=1)
    ax.axhline(0, color=(0.4, 0.4, 0.4), linewidth=0.8, alpha=0.6, zorder=1)
    status = "PASS" if result["passes"] else "FAIL"
    status_color = (0.10, 0.55, 0.20) if result["passes"] else (0.75, 0.15, 0.15)
    ax.set_title(
        f"mpa-scale-solver migration trace: c → s → r\n"
        f"max |residual| = {result['max_abs_residual']:.4f}   "
        f"tolerance = {TOLERANCE:.4f}   [{status}]",
        fontsize=13, color=status_color,
    )
    ax.legend(loc="upper right", fontsize=11, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _write_fixtures(result: dict, fixtures_dir: Path) -> None:
    """One JSON per frame for downstream byte-identical regression."""
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    for i in range(len(result["tau_obs_grid"])):
        fx = {
            "frame_index": i,
            "tau_obs": result["tau_obs_grid"][i],
            "analytical_chit": result["analytical_chit"][i],
            "numerical_chit": result["numerical_chit"][i],
            "residual": result["residuals"][i],
            "regime_5bucket": result["regimes_5"][i],
            "regime_display_band": result["regimes_3"][i],
        }
        (fixtures_dir / f"frame_{i:04d}.json").write_text(
            json.dumps(fx, indent=2) + "\n", encoding="utf-8",
        )


def test_camera_migration():
    """Canonical chit recovery overlays analytical truth within tolerance."""
    result = run_camera_test()
    # Save artifacts under tests/out/ (gitignored) so a human can inspect.
    out_dir = Path(__file__).parent / "out"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _render_plot(result, out_dir / "migration_compare.png")

    # Per-frame fixtures land under tests/fixtures/camera (tracked) but only
    # when the test passes — we don't want failing data to overwrite a known
    # good fixture set.
    if result["passes"]:
        _write_fixtures(result, Path(__file__).parent / "fixtures" / "camera")

    assert result["passes"], (
        f"camera migration max |residual| = {result['max_abs_residual']:.4f} "
        f"exceeds tolerance {TOLERANCE:.4f}"
    )


if __name__ == "__main__":
    sys.exit(0 if test_camera_migration() is None else 1)
