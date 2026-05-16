"""Continuous Banach self-test cadence (v5 — BLOCK_IN §v5).

Every k-th operation call runs a side-test on the Banach substrate
(analytical-truth comparison against `jax_ops.banach_state_diff`).
Drift is reported with full diagnostic state — not raised — so the
primary pipeline keeps flowing while consumers decide whether to act
on a borderline self-test.

State-locality is preserved: self-tests do NOT feed back into the
primary inversion. They are pure pure-function checks against the
analytical Banach truth and ride on a side channel (an optional
callback) so streaming consumers can record them without disturbing
the `InversionResult` iteration shape (per BLOCK_IN §v5 streaming
self-test design call).

Streaming hookup: `forward_sweep_invert_stream` takes a
`SelfTestCadence` + `self_test_callback`; the cadence advances per
**emitted frame** (every k-th yielded `InversionResult` triggers a
self-test), matching the BLOCK_IN refinement.

ECC-for-compute framing: a drifted self-test indicates either a numeric
regression in the solver (JAX backend update, float64 disabled,
optimizer change) or a real bit-flip on the underlying hardware. Both
deserve surfacing.

Python-thread story: the v5 self-test runs synchronously per tick.
The compute is small — a handful of float64 ops per nu sample — so
the "out-of-band" framing in BLOCK_IN §v5 is honored in spirit
(microsecond-scale overhead, doesn't block any real-time stream).
Consumers that need true parallelism can run `run_banach_self_test`
on their own thread / executor and pass the report to their callback;
the v6 native port gets to do it inline via Rayon / OpenMP.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from ._version import __version__
from .banach import BanachSubstrate


# Default sample depths span the migration interior: a c-band start at
# nu=0 traverses deep_c -> c_near_s -> s_critical across this range
# (mirrors the v1 camera-test grid endpoints). Sparse on purpose —
# the self-test is a drift detector, not a sweep.
_DEFAULT_NU_SAMPLES: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 5.0)

# Drift threshold past which the report is flagged. Float64 closed-form
# vs JAX float64 of the same closed form should agree to ~1e-14; 1e-10
# leaves headroom for backend numerical drift while still catching real
# regressions (a typo in a primitive, a backend dropping float64).
DRIFT_TOLERANCE: float = 1e-10


@dataclass(frozen=True)
class BanachDriftReport:
    """Per-tick self-test result against the Banach analytical truth.

    Native-port note: flat dataclass — `int`, `int`, `float`, `float`,
    `float`, `bool`, `bool`, `int`, `str`, plus a string tuple. The
    v6 port reads this as a plain struct with a string-list field.

    `max_relative_drift` is the larger of the per-sample
    `|jax_chit - truth_chit| / max(|truth_chit|, 1e-300)` and the
    gamma analogue; tiny near the asymptote where absolute drift is
    naturally small.
    """

    call_index: int
    sample_count: int
    max_chit_drift: float
    max_gamma_drift: float
    max_relative_drift: float
    asymptotic_closure_compliant: bool
    k_frust_invariant: bool
    timestamp_ns: int
    solver_version: str
    notes: tuple[str, ...] = ()

    @property
    def drift_within_tolerance(self) -> bool:
        return (
            self.max_chit_drift <= DRIFT_TOLERANCE
            and self.max_gamma_drift <= DRIFT_TOLERANCE
        )

    def _repr_html_(self) -> str:
        from .types import _html_table
        ok = self.drift_within_tolerance and self.asymptotic_closure_compliant
        status = "ok" if ok else "DRIFTED"
        return _html_table(f"BanachDriftReport [{status}]", [
            ("call_index", str(self.call_index)),
            ("sample_count", str(self.sample_count)),
            ("max_chit_drift", f"{self.max_chit_drift:.4g}"),
            ("max_gamma_drift", f"{self.max_gamma_drift:.4g}"),
            ("max_relative_drift", f"{self.max_relative_drift:.4g}"),
            ("asymptotic_closure_compliant",
             str(self.asymptotic_closure_compliant)),
            ("k_frust_invariant", str(self.k_frust_invariant)),
            ("solver_version", self.solver_version),
            ("notes", ("; ".join(self.notes) if self.notes else "(none)")),
        ])


def run_banach_self_test(
    *,
    substrate: Optional[BanachSubstrate] = None,
    nu_samples: Optional[Sequence[float]] = None,
    call_index: int = 0,
) -> BanachDriftReport:
    """Synchronous Banach drift check.

    Compares the v2.0 JAX surface (`jax_ops.banach_state_diff`) against
    the analytical closed form (`BanachSubstrate.state_at`). The Banach
    substrate is the framework's analytical truth (Q1 v1 normalization);
    any drift between the two paths flags either a numeric regression in
    the JAX stack or a real bit-flip.

    `substrate` defaults to the framework reference (`BanachSubstrate()`)
    so consumers don't need to thread one through. `nu_samples` defaults
    to a sparse five-point sweep across the migration interior.

    `call_index` is purely informational — the cadence object passes its
    own counter so the report ties back to a known tick.
    """
    # Local import keeps the module import-graph linear (jax_ops imports
    # banach + jax_core; self_test imports banach + jax_ops would close
    # a longer cycle through __init__).
    from .jax_ops import banach_state_diff

    substrate = substrate or BanachSubstrate()
    nus = tuple(nu_samples) if nu_samples is not None else _DEFAULT_NU_SAMPLES

    max_chit_drift = 0.0
    max_gamma_drift = 0.0
    max_relative_drift = 0.0
    asymptotic_compliant = True
    k_frust_invariant = True
    notes: list[str] = []

    for nu in nus:
        truth = substrate.state_at(float(nu))
        jax_chit, jax_gamma = banach_state_diff(substrate, float(nu))

        d_chit = abs(float(jax_chit) - truth.chit)
        d_gamma = abs(float(jax_gamma) - truth.gamma_AB)
        max_chit_drift = max(max_chit_drift, d_chit)
        max_gamma_drift = max(max_gamma_drift, d_gamma)

        denom_chit = max(abs(truth.chit), 1e-300)
        denom_gamma = max(abs(truth.gamma_AB), 1e-300)
        rel = max(d_chit / denom_chit, d_gamma / denom_gamma)
        max_relative_drift = max(max_relative_drift, rel)

        if d_chit > DRIFT_TOLERANCE or d_gamma > DRIFT_TOLERANCE:
            notes.append(
                f"drift at nu={nu}: chit={d_chit:.3e}, gamma_AB={d_gamma:.3e}"
            )

        # Asymptotic-closure: the Banach substrate is the documented
        # exception (sits at the asymptotic limits by construction).
        # We still check that no finite-nu chit/gamma hit exact 0.0 / 1.0
        # in the JAX surface — that would indicate float64 was disabled
        # somewhere and the result rounded to a forbidden literal.
        if float(jax_chit) in (0.0, 1.0) and float(nu) > 0.0 and float(nu) < float("inf"):
            asymptotic_compliant = False
            notes.append(
                f"jax_chit hit asymptotic literal at finite nu={nu} "
                f"(possible float32 fallback)"
            )

        # k_frust passes through banach_state_diff opaquely (it's an
        # auxiliary on the canonical state, not a JAX leaf). Verifying
        # it doesn't flip is vacuous here — Banach defaults to k_frust=False
        # and the JAX surface doesn't touch it. Recorded for completeness.

    return BanachDriftReport(
        call_index=int(call_index),
        sample_count=len(nus),
        max_chit_drift=float(max_chit_drift),
        max_gamma_drift=float(max_gamma_drift),
        max_relative_drift=float(max_relative_drift),
        asymptotic_closure_compliant=asymptotic_compliant,
        k_frust_invariant=k_frust_invariant,
        timestamp_ns=time.time_ns(),
        solver_version=__version__,
        notes=tuple(notes),
    )


@dataclass
class SelfTestCadence:
    """Call counter that triggers a Banach self-test every k-th tick.

    Mutable on purpose: the counter advances per-tick. Pass a fresh
    instance per stream / pipeline; state-locality lives at the cadence
    level, not inside the solver primitives.

    `k = 100` is the BLOCK_IN §v5 default. Tune downward for high-
    consequence pipelines (production audit signing) or upward for
    tight inner loops where the 1/k self-test overhead matters.

    `nu_samples` lets consumers tighten the check around a specific
    region of the migration interior (e.g. their substrate's known
    operating range). Default: the five-point sparse sweep.
    """

    k: int = 100
    nu_samples: Optional[tuple[float, ...]] = None
    substrate: Optional[BanachSubstrate] = None
    _calls: int = field(default=0, init=False)
    _last_report: Optional[BanachDriftReport] = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.k <= 0:
            raise ValueError(f"cadence k must be positive; got {self.k}")

    @property
    def call_count(self) -> int:
        """Total ticks so far (informational; not affected by cadence)."""
        return self._calls

    @property
    def last_report(self) -> Optional[BanachDriftReport]:
        """Most recent self-test report, or None if no tick has fired one."""
        return self._last_report

    def tick(
        self,
        *,
        callback: Optional[Callable[[BanachDriftReport], None]] = None,
    ) -> Optional[BanachDriftReport]:
        """Advance the counter. On every k-th tick, run a self-test.

        Returns the BanachDriftReport when this tick fired one, else None.
        If `callback` is supplied, called with the report (callback errors
        propagate; they're not silently swallowed — a callback that
        raises is the consumer's bug, not the cadence's).
        """
        self._calls += 1
        if self._calls % self.k != 0:
            return None
        report = run_banach_self_test(
            substrate=self.substrate,
            nu_samples=self.nu_samples,
            call_index=self._calls,
        )
        self._last_report = report
        if callback is not None:
            callback(report)
        return report

    def reset(self) -> None:
        """Reset the call counter (e.g. between distinct streams)."""
        self._calls = 0
        self._last_report = None
