"""Analytical gFDR forward model, ported from mpa-auditor/math/gfdr-model.js.

Five pure functions plus a residual scorer. The JS canonical version is
the source of truth; this is a byte-faithful port (modulo float vs np.float64
representation). When the JS version moves, this moves with it.

vertex_regime is the FIVE-bucket classifier (deep_c / c_near_s / s_critical /
r_near_s / deep_r). The three-bucket cut lives in operations.regime_display_band
as a display-only helper (handoff §C.4).

generate_locus depends on chit alone — the single-mode gFDR locus does not
constrain gamma_AB (per the auditor's RFC-S Appendix B item 4 thread).
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from .types import RegimeLabel


N_LOCUS_POINTS = 80


def vertex_regime(chit: float) -> RegimeLabel:
    """Five-bucket regime classifier. Canonical; matches gfdr-model.js."""
    if chit >= 0.7:
        return "deep_c"
    if chit >= 0.2:
        return "c_near_s"
    if chit > -0.2:
        return "s_critical"
    if chit > -0.7:
        return "r_near_s"
    return "deep_r"


def alpha_s(chit: float) -> float:
    """CK s-regime aging-diagonal slope. cdv1 §gFDR signatures."""
    return 0.5 + 0.3 * math.exp(-abs(chit) * 4.0)


def plateau_height(chit: float) -> float:
    """Plateau height of the s-critical locus."""
    return max(0.05, 1.0 - math.exp(-max(0.0, chit + 0.2) * 1.5))


def generate_locus(chit: float, regime: RegimeLabel) -> list[dict]:
    """Analytical gFDR locus chi(tau) / C(tau), log-spaced in tau.

    Returns N_LOCUS_POINTS rows {tau, chi, C}. Branch set matches the
    engines' continuous-mode generateLocus.
    """
    points: list[dict] = []
    tau_min, tau_max = 0.01, 1000.0
    for i in range(N_LOCUS_POINTS):
        t = i / (N_LOCUS_POINTS - 1)
        tau = tau_min * (tau_max / tau_min) ** t
        if regime in ("deep_c", "c_near_s"):
            depth = math.exp(-chit * 1.5)
            tau_c = 4.0 + 6.0 / max(0.1, chit)
            dC = 0.18 * depth * (1.0 - math.exp(-tau / tau_c))
            C = 1.0 - dC
            chi = (0.02 if regime == "deep_c" else 0.08) * dC
        elif regime == "s_critical":
            a = alpha_s(chit)
            P_s = plateau_height(chit)
            dC_short = (1.0 - P_s) * (1.0 - math.exp(-tau / 0.5))
            dC_long = P_s * (1.0 - (1.0 + tau / 50.0) ** (-a))
            dC = dC_short + dC_long
            C = 1.0 - dC
            chi = dC if dC <= (1.0 - P_s) else (1.0 - P_s) + a * (dC - (1.0 - P_s))
        else:  # r_near_s or deep_r
            tau_eq = max(0.5, 1.0 + 0.5 * math.exp(chit))
            dC = 1.0 - math.exp(-tau / tau_eq)
            C = 1.0 - dC
            chi = dC
        points.append({"tau": tau, "chi": chi, "C": C})
    return points


def interp_locus(model: list[dict], tau: float) -> dict:
    """Log-tau interpolation of a gFDR locus at an arbitrary tau."""
    if tau <= model[0]["tau"]:
        return {"C": model[0]["C"], "chi": model[0]["chi"]}
    last = model[-1]
    if tau >= last["tau"]:
        return {"C": last["C"], "chi": last["chi"]}
    for i in range(1, len(model)):
        if model[i]["tau"] >= tau:
            a, b = model[i - 1], model[i]
            f = (math.log(tau) - math.log(a["tau"])) / (math.log(b["tau"]) - math.log(a["tau"]))
            return {
                "C": a["C"] + f * (b["C"] - a["C"]),
                "chi": a["chi"] + f * (b["chi"] - a["chi"]),
            }
    return {"C": last["C"], "chi": last["chi"]}


def locus_residual(empirical_rows: list[dict], chit: float) -> float:
    """Mean squared residual between an empirical locus and the analytical at `chit`.

    empirical_rows: list of {tau, C, chi}. Returned value is the scoring
    function forward_sweep_invert minimizes when the empirical signal is a
    gFDR locus.
    """
    model = generate_locus(chit, vertex_regime(chit))
    sse = 0.0
    for row in empirical_rows:
        m = interp_locus(model, float(row["tau"]))
        dC = float(row["C"]) - m["C"]
        dChi = float(row["chi"]) - m["chi"]
        sse += dC * dC + dChi * dChi
    return sse / max(1, len(empirical_rows))


def locus_residual_array(empirical_rows: list[dict], chit_grid: np.ndarray) -> np.ndarray:
    """Vectorized scan of locus_residual over a chit candidate grid."""
    out = np.empty(chit_grid.shape[0], dtype=np.float64)
    for i, chit in enumerate(chit_grid):
        out[i] = locus_residual(empirical_rows, float(chit))
    return out
