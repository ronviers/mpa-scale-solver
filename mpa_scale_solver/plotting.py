"""Default plot hooks for the seven operations (v4 — BLOCK_IN §v4).

Per north-star §Visualization-first: every operation has a default
viz. v4 ships four helpers covering the operations whose outputs are
visually load-bearing:

  - `plot_trajectory(trajectory, ...)` — canonical-space curve from
    `tau_obs_sweep`, regime-banded.
  - `plot_gamut(gamut, ...)` — gamut envelope with optional overlaid
    points (the `gamut_classify` / `intent_map` view).
  - `plot_residual_field(residuals, grid, ...)` — the
    `forward_sweep_invert` residual landscape with the recovered point.
  - `plot_posterior(posterior, ...)` — `forward_sweep_invert_posterior`
    output: MAP point + covariance ellipse.

Each helper takes ``backend="matplotlib"`` (default) or
``backend="plotly"``. Matplotlib imports lazily (not at module import);
plotly is fully optional — installed only when consumers ask. Helpers
return the backend's figure object so consumers can ``.show()``,
``.savefig()``, or compose further.

Lazy-eval / animation / scrubber UIs (the north-star §"Real-time τ_obs
scrubbing" entry) are deferred — they live in mpa-auditor's display
layer per the suite block-in. These helpers are for notebook
inspection, not embedded UI.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from .gfdr_model import vertex_regime
from .types import (
    CanonicalState,
    GamutSpec,
    Posterior,
)


# Five-bucket display colors (consistent with gfdr-model.js heuristics).
_REGIME_COLORS: dict[str, str] = {
    "deep_c":     "#1a5fb4",
    "c_near_s":   "#62a0ea",
    "s_critical": "#ffbe6f",
    "r_near_s":   "#e66100",
    "deep_r":     "#a51d2d",
}


def _lazy_matplotlib():
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for the matplotlib backend; install with "
            "`pip install matplotlib`"
        ) from exc
    return plt


def _lazy_plotly():
    try:
        import plotly.graph_objects as go  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "plotly is required for the plotly backend; install with "
            "`pip install plotly`"
        ) from exc
    return go


# ---------------------------------------------------------------------------
# plot_trajectory — tau_obs_sweep canonical trajectory
# ---------------------------------------------------------------------------


def plot_trajectory(
    trajectory: Sequence[CanonicalState],
    *,
    backend: str = "matplotlib",
    tau_obs_grid: Optional[Sequence[float]] = None,
    title: Optional[str] = None,
) -> Any:
    """Plot a canonical-space trajectory regime-banded by chit bucket.

    `tau_obs_grid` (optional) labels each point with its observer-scale.
    """
    chits = np.array([s.chit for s in trajectory], dtype=np.float64)
    gammas = np.array([s.gamma_AB for s in trajectory], dtype=np.float64)
    regimes = [vertex_regime(c) for c in chits]
    colors = [_REGIME_COLORS[r] for r in regimes]

    if backend == "matplotlib":
        plt = _lazy_matplotlib()
        fig, ax = plt.subplots()
        ax.plot(chits, gammas, color="0.6", linewidth=1.0, zorder=1)
        ax.scatter(chits, gammas, c=colors, s=24, zorder=2)
        ax.set_xlabel(r"$\chi_t$")
        ax.set_ylabel(r"$\gamma_{AB}$")
        if title:
            ax.set_title(title)
        return fig
    if backend == "plotly":
        go = _lazy_plotly()
        text = (
            [f"τ={t:g}" for t in tau_obs_grid]
            if tau_obs_grid is not None else None
        )
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=chits, y=gammas, mode="lines+markers",
            marker=dict(color=colors, size=8),
            line=dict(color="rgba(120,120,120,0.5)", width=1),
            text=text,
        ))
        fig.update_layout(
            xaxis_title="chit", yaxis_title="gamma_AB", title=title,
        )
        return fig
    raise ValueError(f"unsupported backend: {backend!r}")


# ---------------------------------------------------------------------------
# plot_gamut — gamut envelope with optional overlay
# ---------------------------------------------------------------------------


def plot_gamut(
    gamut: GamutSpec,
    *,
    points: Optional[Sequence[CanonicalState]] = None,
    backend: str = "matplotlib",
    title: Optional[str] = None,
) -> Any:
    """Plot the gamut rectangle in (chit, gamma_AB) with optional points.

    `points` overlays canonical states; in-gamut points are filled,
    out-of-gamut are outlined.
    """
    lo_c, hi_c = gamut.chit_range
    lo_g, hi_g = gamut.gamma_AB_range
    box_x = [lo_c, hi_c, hi_c, lo_c, lo_c]
    box_y = [lo_g, lo_g, hi_g, hi_g, lo_g]

    in_pts: list[CanonicalState] = []
    out_pts: list[CanonicalState] = []
    for p in points or ():
        if lo_c <= p.chit <= hi_c and lo_g <= p.gamma_AB <= hi_g:
            in_pts.append(p)
        else:
            out_pts.append(p)

    if backend == "matplotlib":
        plt = _lazy_matplotlib()
        fig, ax = plt.subplots()
        ax.plot(box_x, box_y, color="0.4")
        if in_pts:
            ax.scatter(
                [p.chit for p in in_pts], [p.gamma_AB for p in in_pts],
                c="tab:blue", s=24, label="in-gamut",
            )
        if out_pts:
            ax.scatter(
                [p.chit for p in out_pts], [p.gamma_AB for p in out_pts],
                facecolors="none", edgecolors="tab:red", s=24, label="out-of-gamut",
            )
        if in_pts or out_pts:
            ax.legend(loc="best")
        ax.set_xlabel(r"$\chi_t$")
        ax.set_ylabel(r"$\gamma_{AB}$")
        if title:
            ax.set_title(title)
        return fig
    if backend == "plotly":
        go = _lazy_plotly()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=box_x, y=box_y, mode="lines",
            line=dict(color="rgba(80,80,80,1.0)"), name="gamut",
        ))
        if in_pts:
            fig.add_trace(go.Scatter(
                x=[p.chit for p in in_pts], y=[p.gamma_AB for p in in_pts],
                mode="markers", marker=dict(color="royalblue", size=8),
                name="in-gamut",
            ))
        if out_pts:
            fig.add_trace(go.Scatter(
                x=[p.chit for p in out_pts], y=[p.gamma_AB for p in out_pts],
                mode="markers",
                marker=dict(color="rgba(0,0,0,0)", size=8,
                            line=dict(color="firebrick", width=2)),
                name="out-of-gamut",
            ))
        fig.update_layout(
            xaxis_title="chit", yaxis_title="gamma_AB", title=title,
        )
        return fig
    raise ValueError(f"unsupported backend: {backend!r}")


# ---------------------------------------------------------------------------
# plot_residual_field — forward_sweep_invert residual landscape
# ---------------------------------------------------------------------------


def plot_residual_field(
    residuals: np.ndarray,
    canonical_grid: np.ndarray,
    *,
    recovered: Optional[CanonicalState] = None,
    backend: str = "matplotlib",
    title: Optional[str] = None,
) -> Any:
    """Plot the per-candidate residual surface from forward_sweep_invert.

    Expects a flat-array residual + matching `canonical_grid` (N, 2).
    Renders as a scatter colored by residual; marks the recovered point.
    """
    if canonical_grid.shape[0] != residuals.shape[0]:
        raise ValueError("residuals and canonical_grid must have matching length")
    chits = canonical_grid[:, 0]
    gammas = canonical_grid[:, 1]

    if backend == "matplotlib":
        plt = _lazy_matplotlib()
        fig, ax = plt.subplots()
        sc = ax.scatter(chits, gammas, c=residuals, cmap="viridis", s=30)
        if recovered is not None:
            ax.scatter(
                [recovered.chit], [recovered.gamma_AB],
                marker="x", color="red", s=80, linewidths=2,
                label="recovered",
            )
            ax.legend(loc="best")
        fig.colorbar(sc, ax=ax, label="residual")
        ax.set_xlabel(r"$\chi_t$")
        ax.set_ylabel(r"$\gamma_{AB}$")
        if title:
            ax.set_title(title)
        return fig
    if backend == "plotly":
        go = _lazy_plotly()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=chits, y=gammas, mode="markers",
            marker=dict(color=residuals, colorscale="Viridis", showscale=True,
                        colorbar=dict(title="residual")),
        ))
        if recovered is not None:
            fig.add_trace(go.Scatter(
                x=[recovered.chit], y=[recovered.gamma_AB],
                mode="markers",
                marker=dict(symbol="x", color="red", size=14, line=dict(width=2)),
                name="recovered",
            ))
        fig.update_layout(
            xaxis_title="chit", yaxis_title="gamma_AB", title=title,
        )
        return fig
    raise ValueError(f"unsupported backend: {backend!r}")


# ---------------------------------------------------------------------------
# plot_posterior — forward_sweep_invert_posterior MAP + covariance ellipse
# ---------------------------------------------------------------------------


def plot_posterior(
    posterior: Posterior,
    *,
    backend: str = "matplotlib",
    n_sigma: float = 2.0,
    title: Optional[str] = None,
) -> Any:
    """Plot the MAP point with the n-sigma covariance ellipse.

    `posterior.covariance` is the 2x2 Laplace-approximation surface.
    The ellipse axes are eigenvectors scaled by `n_sigma * sqrt(eigval)`.
    Non-positive-definite covariance is handled by clipping eigenvalues
    to zero (the ellipse collapses).
    """
    mean = posterior.mean
    cov = np.array(posterior.covariance, dtype=np.float64)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 0.0, None)
    # Parametric ellipse: (x, y) = mean + R @ diag(sqrt(eigvals)) @ [cos, sin]
    theta = np.linspace(0.0, 2.0 * np.pi, 128)
    unit = np.stack([np.cos(theta), np.sin(theta)], axis=0)
    scaled = (eigvecs @ np.diag(n_sigma * np.sqrt(eigvals))) @ unit
    ex = mean.chit + scaled[0]
    ey = mean.gamma_AB + scaled[1]

    if backend == "matplotlib":
        plt = _lazy_matplotlib()
        fig, ax = plt.subplots()
        ax.plot(ex, ey, color="tab:blue", linewidth=1.5,
                label=f"{n_sigma}σ Laplace")
        ax.scatter([mean.chit], [mean.gamma_AB], marker="x",
                   color="tab:red", s=80, linewidths=2, label="MAP")
        ax.legend(loc="best")
        ax.set_xlabel(r"$\chi_t$")
        ax.set_ylabel(r"$\gamma_{AB}$")
        if title:
            ax.set_title(title)
        return fig
    if backend == "plotly":
        go = _lazy_plotly()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ex, y=ey, mode="lines", line=dict(color="royalblue"),
            name=f"{n_sigma}σ Laplace",
        ))
        fig.add_trace(go.Scatter(
            x=[mean.chit], y=[mean.gamma_AB], mode="markers",
            marker=dict(symbol="x", color="crimson", size=14, line=dict(width=2)),
            name="MAP",
        ))
        fig.update_layout(
            xaxis_title="chit", yaxis_title="gamma_AB", title=title,
        )
        return fig
    raise ValueError(f"unsupported backend: {backend!r}")
