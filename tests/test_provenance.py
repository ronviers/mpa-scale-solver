"""Provenance trail tests (handoff §D.2 item 8).

Per-call provenance is correctly populated; downstream consumers can read
it from `OperationOutput.provenance`. The `provenance_hash` is stable
across runs for a given (version, operation, dispatch_path, table_version)
combination — timestamps and notes are excluded so the hash is
reproducible.
"""

from __future__ import annotations

import numpy as np
import pytest

from mpa_scale_solver import (
    BanachSubstrate,
    CanonicalState,
    DispatchPath,
    GamutSpec,
    Provenance,
    apply_translation_wrapped,
    forward_sweep_invert_wrapped,
    gamut_classify_wrapped,
    intent_map_wrapped,
    make_provenance,
    provenance_hash,
    regime_at_wrapped,
    tau_obs_sweep_wrapped,
    validate_driver_profile_wrapped,
    __version__,
)


def _trivial_canonical() -> CanonicalState:
    return CanonicalState(chit=0.5, gamma_AB=-0.3)


class TestMakeProvenance:
    def test_carries_solver_version(self):
        prov = make_provenance("test_op")
        assert prov.solver_version == __version__
        assert prov.operation == "test_op"
        assert prov.dispatch_path == DispatchPath.DIRECT_COMPUTE

    def test_table_hit_path(self):
        prov = make_provenance(
            "apply_translation",
            dispatch_path=DispatchPath.TABLE_HIT,
            table_version="banach-1.0.0",
        )
        assert prov.dispatch_path == DispatchPath.TABLE_HIT
        assert prov.table_version == "banach-1.0.0"

    def test_timestamp_monotonic(self):
        p1 = make_provenance("a")
        p2 = make_provenance("b")
        assert p2.timestamp_ns >= p1.timestamp_ns


class TestProvenanceHash:
    def test_hash_is_reproducible(self):
        p1 = make_provenance("apply_translation")
        p2 = make_provenance("apply_translation")  # different timestamp
        # Hash excludes timestamp + notes, so equivalent operations hash equal.
        assert provenance_hash(p1) == provenance_hash(p2)

    def test_hash_differs_by_operation(self):
        p1 = make_provenance("apply_translation")
        p2 = make_provenance("forward_sweep_invert")
        assert provenance_hash(p1) != provenance_hash(p2)

    def test_hash_differs_by_dispatch_path(self):
        p1 = make_provenance("apply_translation", dispatch_path=DispatchPath.DIRECT_COMPUTE)
        p2 = make_provenance("apply_translation", dispatch_path=DispatchPath.TABLE_HIT)
        assert provenance_hash(p1) != provenance_hash(p2)

    def test_hash_in_unit_interval(self):
        prov = make_provenance("apply_translation")
        h = provenance_hash(prov)
        assert 0.0 <= h < 1.0


class TestProvenanceOnEachWrappedOp:
    """Every wrapped op stamps a populated Provenance."""

    def test_apply_translation_wrapped(self):
        out = apply_translation_wrapped(
            _trivial_canonical(),
            BanachSubstrate().translation_field(),
            tau_obs=1.0,
        )
        assert out.provenance.operation == "apply_translation"
        assert out.provenance.solver_version == __version__

    def test_forward_sweep_invert_wrapped(self):
        substrate = BanachSubstrate()
        target = substrate.substrate_at(1.0)
        grid = np.array([[0.5, -0.2], [substrate.state_at(1.0).chit, substrate.state_at(1.0).gamma_AB]])
        out = forward_sweep_invert_wrapped(
            target, substrate.translation_field(), 1.0, grid,
        )
        assert out.provenance.operation == "forward_sweep_invert"

    def test_tau_obs_sweep_wrapped(self):
        substrate = BanachSubstrate()
        nu_grid = np.array([0.5, 1.0])
        targets = [substrate.substrate_at(float(nu)) for nu in nu_grid]
        truth_grid = np.array([
            [substrate.state_at(float(nu)).chit, substrate.state_at(float(nu)).gamma_AB]
            for nu in nu_grid
        ])
        out = tau_obs_sweep_wrapped(
            targets, substrate.translation_field(), nu_grid, truth_grid,
        )
        assert out.provenance.operation == "tau_obs_sweep"
        assert "frames:" in out.provenance.notes[0]

    def test_regime_at_wrapped(self):
        out = regime_at_wrapped(_trivial_canonical(), 1.0)
        assert out.provenance.operation == "regime_at"

    def test_gamut_classify_wrapped(self):
        out = gamut_classify_wrapped(
            _trivial_canonical(),
            1.0,
            GamutSpec(chit_range=(-1.0, 1.0), gamma_AB_range=(-1.0, 1.0)),
        )
        assert out.provenance.operation == "gamut_classify"

    def test_intent_map_wrapped(self):
        gamut = GamutSpec(chit_range=(-0.5, 0.5), gamma_AB_range=(-1.0, 1.0))
        out = intent_map_wrapped(
            CanonicalState(chit=2.0, gamma_AB=0.0), 1.0, gamut, "I5",
        )
        assert out.provenance.operation == "intent_map"

    def test_validate_driver_profile_wrapped(self):
        from mpa_scale_solver import TranslationField, TranslationRule, OperatingPoint, CanonicalPoint
        # Trivial one-rule field
        field = TranslationField(
            direction="forward", shape="lookup_table",
            rule=[TranslationRule(
                operating_point=OperatingPoint(label="o", gt="s"),
                xdot_choice="x",
                canonical=CanonicalPoint(chit=0.0, gamma_AB=0.0, k_frust=False, method="t"),
            )],
        )
        out = validate_driver_profile_wrapped(
            field,
            [{"canonical_state": CanonicalState(chit=0.0, gamma_AB=0.0), "tau_obs": 1.0}],
            np.array([[0.0, 0.0]]),
        )
        assert out.provenance.operation == "validate_driver_profile"


class TestProvenanceIsImmutable:
    def test_provenance_is_frozen(self):
        prov = make_provenance("x")
        with pytest.raises(Exception):
            prov.operation = "y"  # type: ignore[misc]
