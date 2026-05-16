"""Symbolic query interface (v4 — BLOCK_IN §v4).

Mathematica-style exploration: write a small string, get a closed-form
expression where one exists plus a numerical evaluation. Translates to
operation chains the seven-operation API already exposes — this module
adds zero new math, only a string surface for interactive use.

Five DSL patterns (the BLOCK_IN cap):

  1. ``regime at chit=A gamma=B [tau=T]``
        regime_at — five-bucket vertex regime; closed_form is the
        label (lookup is exact, no derivation needed).
  2. ``gamut at chit=A gamma=B [tau=T]`` (gamut from kwarg)
        gamut_classify — in-gamut diagnosis with per-axis distances.
  3. ``translate chit=A gamma=B at tau=T`` (field from kwarg)
        apply_translation — forward map. For tangent_flow fields the
        substituted closed-form scaling expression is returned.
  4. ``invert chit_obs=X gamma_obs=Y at tau=T`` (field + canonical_grid
     from kwargs)
        forward_sweep_invert — substrate→canonical. tangent_flow only
        for the v4 DSL (the closed-form algebraic inverse exists);
        lookup_table / learned consumers call the operation directly
        for now.
  5. ``tau where regime crosses B for chit=A gamma=B`` (field from
     kwarg; numerical needs tau_range kwarg for non-tangent fields)
        Root-find the tau at which the regime classifier crosses the
        boundary B (one of 0.7 / 0.2 / -0.2 / -0.7). tangent_flow
        with delta_chit ≠ 0 has a closed-form solution; other shapes
        bisect numerically over `tau_range`.

The DSL parses *intent* and *literal parameters*. Structural context
(the field, gamut, grid, tau range) rides as kwargs — that's the
Mathematica pattern too: the syntax names the operation, the kernel
holds the structures.

Patterns are matched by regex (one per pattern). The first match wins.
A query that matches no pattern raises ``QueryParseError`` with the
list of supported patterns.

This module is read-only — nothing here mutates the operations or the
field. Idempotent across calls.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field as _field
from typing import Any, Optional

import numpy as np

from .gfdr_model import vertex_regime
from .operations import (
    apply_translation,
    forward_sweep_invert,
    gamut_classify,
    regime_at,
)
from .types import (
    AnyTranslationField,
    CanonicalState,
    GamutSpec,
    SubstrateState,
    TangentFlowField,
)


# ---------------------------------------------------------------------------
# Result + error types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QueryResult:
    """Output of `query()` (v4 — BLOCK_IN §v4).

    `pattern` names the matched DSL pattern (the imperative verb).
    `numerical` is the typed answer (regime label, recovered state,
    in_gamut dict, tau float, ...). `closed_form` is a multi-line
    string with the substituted expression when one exists; None
    otherwise.
    """

    query: str
    pattern: str
    numerical: Any
    closed_form: Optional[str] = None
    notes: tuple[str, ...] = ()


class QueryParseError(ValueError):
    """The query string matched no supported DSL pattern."""


# The five regime boundaries the `tau where regime crosses` pattern accepts.
# Matches the boundary set used by `gfdr_model.vertex_regime`.
_REGIME_BOUNDARIES: tuple[float, ...] = (0.7, 0.2, -0.2, -0.7)

# Float pattern — accepts integer, decimal, scientific. Used inline below.
_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def _f(s: str) -> float:
    return float(s)


# ---------------------------------------------------------------------------
# Pattern regexes
# ---------------------------------------------------------------------------


_RX_REGIME_AT = re.compile(
    rf"^\s*regime\s+at\s+chit\s*=\s*(?P<chit>{_NUM})"
    rf"\s+gamma\s*=\s*(?P<gamma>{_NUM})"
    rf"(?:\s+tau\s*=\s*(?P<tau>{_NUM}))?\s*$",
    re.IGNORECASE,
)

_RX_GAMUT_AT = re.compile(
    rf"^\s*gamut\s+at\s+chit\s*=\s*(?P<chit>{_NUM})"
    rf"\s+gamma\s*=\s*(?P<gamma>{_NUM})"
    rf"(?:\s+tau\s*=\s*(?P<tau>{_NUM}))?\s*$",
    re.IGNORECASE,
)

_RX_TRANSLATE = re.compile(
    rf"^\s*translate\s+chit\s*=\s*(?P<chit>{_NUM})"
    rf"\s+gamma\s*=\s*(?P<gamma>{_NUM})"
    rf"\s+at\s+tau\s*=\s*(?P<tau>{_NUM})\s*$",
    re.IGNORECASE,
)

_RX_INVERT = re.compile(
    rf"^\s*invert\s+chit_obs\s*=\s*(?P<chit_obs>{_NUM})"
    rf"\s+gamma_obs\s*=\s*(?P<gamma_obs>{_NUM})"
    rf"\s+at\s+tau\s*=\s*(?P<tau>{_NUM})\s*$",
    re.IGNORECASE,
)

_RX_TAU_CROSSING = re.compile(
    rf"^\s*tau\s+where\s+regime\s+crosses\s+(?P<boundary>{_NUM})"
    rf"\s+for\s+chit\s*=\s*(?P<chit>{_NUM})"
    rf"\s+gamma\s*=\s*(?P<gamma>{_NUM})\s*$",
    re.IGNORECASE,
)


_PATTERN_NAMES = (
    "regime at chit=A gamma=B [tau=T]",
    "gamut at chit=A gamma=B [tau=T]                       (gamut= kwarg)",
    "translate chit=A gamma=B at tau=T                     (field= kwarg)",
    "invert chit_obs=X gamma_obs=Y at tau=T                (field= kwarg)",
    "tau where regime crosses B for chit=A gamma=B         (field= kwarg)",
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def query(query_str: str, **context: Any) -> QueryResult:
    """Parse `query_str`, dispatch to the matching operation, return a result.

    Recognized context kwargs:

      field        AnyTranslationField — required for translate / invert /
                   tau-crossing patterns.
      gamut        GamutSpec — required for the gamut pattern.
      canonical_grid numpy array (N, 2) — required for invert.
      tau_range    (lo, hi) tuple — required for tau-crossing on
                   lookup_table / learned fields (where no closed
                   form exists; bisection sweeps this range).
      bisection_max_iter int — caps the bisection (default 60).
      bisection_tol float — bisection abs tolerance on tau (default 1e-9).
    """
    m = _RX_REGIME_AT.match(query_str)
    if m:
        return _do_regime_at(query_str, m, context)
    m = _RX_GAMUT_AT.match(query_str)
    if m:
        return _do_gamut_at(query_str, m, context)
    m = _RX_TRANSLATE.match(query_str)
    if m:
        return _do_translate(query_str, m, context)
    m = _RX_INVERT.match(query_str)
    if m:
        return _do_invert(query_str, m, context)
    m = _RX_TAU_CROSSING.match(query_str)
    if m:
        return _do_tau_crossing(query_str, m, context)
    raise QueryParseError(
        "query matched no supported pattern. Supported patterns:\n  - "
        + "\n  - ".join(_PATTERN_NAMES)
    )


def supported_patterns() -> tuple[str, ...]:
    """Return the DSL patterns recognized by `query()`."""
    return _PATTERN_NAMES


# ---------------------------------------------------------------------------
# Pattern handlers
# ---------------------------------------------------------------------------


def _do_regime_at(q: str, m: re.Match, ctx: dict) -> QueryResult:
    chit = _f(m.group("chit"))
    gamma = _f(m.group("gamma"))
    tau = _f(m.group("tau")) if m.group("tau") else 1.0
    state = CanonicalState(chit=chit, gamma_AB=gamma)
    reading = regime_at(state, tau)
    return QueryResult(
        query=q,
        pattern="regime at",
        numerical=reading,
        closed_form=f"regime(chit={chit}) = {reading.regime}",
    )


def _do_gamut_at(q: str, m: re.Match, ctx: dict) -> QueryResult:
    gamut = ctx.get("gamut")
    if gamut is None:
        raise QueryParseError("gamut query requires `gamut=` kwarg (GamutSpec)")
    chit = _f(m.group("chit"))
    gamma = _f(m.group("gamma"))
    tau = _f(m.group("tau")) if m.group("tau") else 1.0
    state = CanonicalState(chit=chit, gamma_AB=gamma)
    result = gamut_classify(state, tau, gamut)
    closed = (
        f"in_gamut = ({gamut.chit_range[0]} <= {chit} <= {gamut.chit_range[1]})"
        f" AND ({gamut.gamma_AB_range[0]} <= {gamma} <= {gamut.gamma_AB_range[1]})"
        f" = {result['in_gamut']}"
    )
    return QueryResult(query=q, pattern="gamut at", numerical=result, closed_form=closed)


def _do_translate(q: str, m: re.Match, ctx: dict) -> QueryResult:
    field = ctx.get("field")
    if field is None:
        raise QueryParseError("translate query requires `field=` kwarg")
    chit = _f(m.group("chit"))
    gamma = _f(m.group("gamma"))
    tau = _f(m.group("tau"))
    state = CanonicalState(chit=chit, gamma_AB=gamma)
    substrate = apply_translation(state, field, tau)

    closed: Optional[str] = None
    if isinstance(field, TangentFlowField):
        rule = field.scaling
        ratio = tau / rule.tau_obs_ref if (tau > 0.0 and rule.tau_obs_ref > 0.0) else None
        if ratio is not None:
            log_term = math.log(ratio)
            s_chit = chit + rule.delta_chit * log_term
            s_gamma = gamma * (ratio ** rule.delta_gamma)
            closed = (
                f"substrate_chit     = chit + delta_chit * ln(tau/tau_ref)\n"
                f"                   = {chit} + {rule.delta_chit} * ln({tau}/{rule.tau_obs_ref})\n"
                f"                   = {s_chit}\n"
                f"substrate_gamma_AB = gamma * (tau/tau_ref)^delta_gamma\n"
                f"                   = {gamma} * ({tau}/{rule.tau_obs_ref})^{rule.delta_gamma}\n"
                f"                   = {s_gamma}"
            )
    return QueryResult(
        query=q, pattern="translate", numerical=substrate, closed_form=closed,
    )


def _do_invert(q: str, m: re.Match, ctx: dict) -> QueryResult:
    field = ctx.get("field")
    if field is None:
        raise QueryParseError("invert query requires `field=` kwarg")
    if not isinstance(field, TangentFlowField):
        raise QueryParseError(
            "invert pattern in the v4 DSL supports tangent_flow fields only "
            "(closed-form algebraic inverse). lookup_table and learned "
            "fields: call forward_sweep_invert directly."
        )
    chit_obs = _f(m.group("chit_obs"))
    gamma_obs = _f(m.group("gamma_obs"))
    tau = _f(m.group("tau"))

    rule = field.scaling
    if tau <= 0.0 or rule.tau_obs_ref <= 0.0:
        # apply_translation treats this as identity; the inverse is the
        # observation itself.
        recovered = CanonicalState(chit=chit_obs, gamma_AB=gamma_obs)
        closed = (
            f"tau <= 0 -> identity translation; canonical = (chit_obs, gamma_obs) "
            f"= ({chit_obs}, {gamma_obs})"
        )
    else:
        ratio = tau / rule.tau_obs_ref
        chit_canon = chit_obs - rule.delta_chit * math.log(ratio)
        if rule.delta_gamma == 0.0:
            gamma_canon = gamma_obs
        else:
            gamma_canon = gamma_obs / (ratio ** rule.delta_gamma)
        recovered = CanonicalState(chit=chit_canon, gamma_AB=gamma_canon)
        closed = (
            f"chit     = chit_obs - delta_chit * ln(tau/tau_ref)\n"
            f"         = {chit_obs} - {rule.delta_chit} * ln({tau}/{rule.tau_obs_ref})\n"
            f"         = {chit_canon}\n"
            f"gamma_AB = gamma_obs / (tau/tau_ref)^delta_gamma\n"
            f"         = {gamma_obs} / ({tau}/{rule.tau_obs_ref})^{rule.delta_gamma}\n"
            f"         = {gamma_canon}"
        )
    return QueryResult(
        query=q, pattern="invert", numerical=recovered, closed_form=closed,
    )


def _do_tau_crossing(q: str, m: re.Match, ctx: dict) -> QueryResult:
    field = ctx.get("field")
    if field is None:
        raise QueryParseError("tau-crossing query requires `field=` kwarg")
    boundary = _f(m.group("boundary"))
    if boundary not in _REGIME_BOUNDARIES:
        raise QueryParseError(
            f"regime boundary {boundary} is not one of the 5-bucket "
            f"thresholds {_REGIME_BOUNDARIES}"
        )
    chit0 = _f(m.group("chit"))
    gamma = _f(m.group("gamma"))

    if isinstance(field, TangentFlowField):
        rule = field.scaling
        if rule.delta_chit == 0.0:
            # chit(tau) ≡ chit0; never crosses (or always equals).
            crosses = (chit0 == boundary)
            return QueryResult(
                query=q, pattern="tau where regime crosses",
                numerical=None,
                closed_form=(
                    f"delta_chit = 0 -> chit(tau) = chit0 = {chit0} "
                    f"(constant); {'equals' if crosses else 'never crosses'} "
                    f"boundary {boundary}"
                ),
                notes=("constant-chit field; tau-crossing undefined",),
            )
        # chit(tau) = chit0 + delta_chit * ln(tau/tau_ref) = boundary
        # -> tau = tau_ref * exp((boundary - chit0) / delta_chit)
        tau_cross = rule.tau_obs_ref * math.exp((boundary - chit0) / rule.delta_chit)
        closed = (
            f"tau = tau_ref * exp((boundary - chit0) / delta_chit)\n"
            f"    = {rule.tau_obs_ref} * exp(({boundary} - {chit0}) / {rule.delta_chit})\n"
            f"    = {tau_cross}"
        )
        return QueryResult(
            query=q, pattern="tau where regime crosses",
            numerical=tau_cross, closed_form=closed,
        )

    # Numerical bisection for lookup_table / learned fields.
    tau_range = ctx.get("tau_range")
    if tau_range is None:
        raise QueryParseError(
            "tau-crossing on non-tangent-flow fields needs `tau_range=(lo, hi)` "
            "kwarg (the bisection bracket)"
        )
    lo, hi = float(tau_range[0]), float(tau_range[1])
    max_iter = int(ctx.get("bisection_max_iter", 60))
    tol = float(ctx.get("bisection_tol", 1e-9))

    def chit_at(tau: float) -> float:
        substrate = apply_translation(
            CanonicalState(chit=chit0, gamma_AB=gamma), field, tau,
        )
        # Pull a chit-like scalar from the substrate. Tangent-flow path is
        # handled above; lookup_table puts canonical chit in the matched
        # rule's `observables['canonical_chit']`; learned puts the
        # substrate-side chit in `observables['substrate_chit']`. We
        # treat the observable as the regime probe.
        for k in ("substrate_chit", "canonical_chit"):
            if k in substrate.observables:
                return float(substrate.observables[k])
        return chit0  # no probe; constant

    f_lo = chit_at(lo) - boundary
    f_hi = chit_at(hi) - boundary
    if f_lo * f_hi > 0.0:
        return QueryResult(
            query=q, pattern="tau where regime crosses",
            numerical=None, closed_form=None,
            notes=(
                f"no sign change of (chit(tau) - {boundary}) on [{lo}, {hi}]; "
                "boundary not crossed in bracket",
            ),
        )
    a, b, fa, fb = lo, hi, f_lo, f_hi
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        fm = chit_at(mid) - boundary
        if abs(fm) < tol or 0.5 * abs(b - a) < tol:
            return QueryResult(
                query=q, pattern="tau where regime crosses",
                numerical=mid, closed_form=None,
                notes=(f"bisection converged on [{lo}, {hi}] in <= {max_iter} iter",),
            )
        if fa * fm < 0.0:
            b, fb = mid, fm
        else:
            a, fa = mid, fm
    return QueryResult(
        query=q, pattern="tau where regime crosses",
        numerical=0.5 * (a + b), closed_form=None,
        notes=(f"bisection hit max_iter={max_iter} without abs-tol {tol}",),
    )
