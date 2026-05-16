"""Sidecar dispatch tests (handoff §D.2 item 6, §E item 5).

The sidecar is opt-in: `forward_sweep_invert_wrapped` works with or
without it; the result value is identical (table-hit equals compute
result for table-covered keys); the `provenance.dispatch_path` carries
the distinction.
"""

from __future__ import annotations

import numpy as np
import pytest

from mpa_scale_solver import (
    BanachSubstrate,
    CanonicalState,
    DispatchPath,
    InverseLookupSidecar,
    SubstrateState,
    apply_translation,
    apply_translation_wrapped,
    build_sidecar_for_banach,
    forward_sweep_invert,
    forward_sweep_invert_wrapped,
    lookup_forward,
    lookup_inverse,
    round_key,
)


TAU_OBS_GRID = np.array([0.5, 1.0, 2.0, 3.0])


def _banach_setup():
    substrate = BanachSubstrate(chit_0=1.5, gamma_AB_0=-0.5)
    field = substrate.translation_field()
    sidecar = substrate.build_sidecar(TAU_OBS_GRID)
    return substrate, field, sidecar


class TestSidecarLookup:
    def test_round_key_quantizes_consistently(self):
        k1 = round_key((1.1234567, 0.5, 2.0))
        k2 = round_key((1.1234568, 0.5, 2.0))
        # 6-decimal rounding folds the two values to the same key.
        assert k1[0] == k2[0]

    def test_forward_lookup_hits_recorded_canonicals(self):
        substrate, field, sidecar = _banach_setup()
        for nu in TAU_OBS_GRID:
            c = substrate.state_at(float(nu))
            hit = lookup_forward(sidecar, c, float(nu))
            assert hit is not None
            assert hit.observables["substrate_chit"] == pytest.approx(c.chit)

    def test_inverse_lookup_hits_recorded_substrates(self):
        substrate, field, sidecar = _banach_setup()
        for nu in TAU_OBS_GRID:
            s = substrate.substrate_at(float(nu))
            hit = lookup_inverse(sidecar, s, float(nu))
            assert hit is not None
            assert hit.chit == pytest.approx(substrate.state_at(float(nu)).chit)

    def test_inverse_lookup_misses_off_grid(self):
        substrate, field, sidecar = _banach_setup()
        # tau_obs not in grid -> miss.
        off_grid = substrate.substrate_at(7.0)
        assert lookup_inverse(sidecar, off_grid, 7.0) is None

    def test_inverse_lookup_misses_without_substrate_keys(self):
        _, _, sidecar = _banach_setup()
        # SubstrateState lacking substrate_chit/substrate_gamma_AB -> miss.
        bare = SubstrateState(tau_obs=1.0, observables={"other": 0.5})
        assert lookup_inverse(sidecar, bare, 1.0) is None


class TestSidecarDispatchEquivalence:
    """With + without sidecar produce identical values + correct dispatch."""

    def test_apply_translation_wrapped_with_sidecar_hits_table(self):
        substrate, field, sidecar = _banach_setup()
        c = substrate.state_at(1.0)

        out_no = apply_translation_wrapped(c, field, tau_obs=1.0)
        out_yes = apply_translation_wrapped(c, field, tau_obs=1.0, sidecar=sidecar)

        assert out_no.provenance.dispatch_path == DispatchPath.DIRECT_COMPUTE
        assert out_yes.provenance.dispatch_path == DispatchPath.TABLE_HIT

        # Same substrate value (identity translation; table records it).
        assert out_yes.value.observables["substrate_chit"] == pytest.approx(
            out_no.value.observables["substrate_chit"]
        )

    def test_apply_translation_wrapped_sidecar_miss_falls_back(self):
        substrate, field, sidecar = _banach_setup()
        c = CanonicalState(chit=0.123456, gamma_AB=-0.0654321)  # not in grid

        out = apply_translation_wrapped(c, field, tau_obs=1.0, sidecar=sidecar)
        assert out.provenance.dispatch_path == DispatchPath.COMPUTE_FALLBACK
        assert out.provenance.table_version == sidecar.version
        # Compute-fallback gives the identity-translation value.
        assert out.value.observables["substrate_chit"] == pytest.approx(c.chit)

    def test_forward_sweep_invert_wrapped_with_sidecar_hits(self):
        substrate, field, sidecar = _banach_setup()
        nu = 1.0
        target = substrate.substrate_at(nu)

        # Tiny search grid - the sidecar should bypass it.
        grid = np.array([[0.0, 0.0]])

        out_no = forward_sweep_invert_wrapped(target, field, nu, grid)
        out_yes = forward_sweep_invert_wrapped(target, field, nu, grid, sidecar=sidecar)

        assert out_no.provenance.dispatch_path == DispatchPath.DIRECT_COMPUTE
        assert out_yes.provenance.dispatch_path == DispatchPath.TABLE_HIT

        # The sidecar hit recovers the canonical exactly; the compute
        # path lands on the closest grid candidate (0,0) which is wrong.
        truth = substrate.state_at(nu)
        assert out_yes.value.chit == pytest.approx(truth.chit)
        assert out_yes.value.gamma_AB == pytest.approx(truth.gamma_AB)

    def test_forward_sweep_invert_wrapped_miss_falls_back(self):
        substrate, field, sidecar = _banach_setup()
        # An off-grid substrate => sidecar misses, compute path runs.
        bare = SubstrateState(tau_obs=1.0, observables={"unrelated": 0.0})
        grid = np.array([[0.0, 0.0]])

        out = forward_sweep_invert_wrapped(bare, field, 1.0, grid, sidecar=sidecar)
        assert out.provenance.dispatch_path == DispatchPath.COMPUTE_FALLBACK


class TestSidecarStructure:
    def test_sidecar_carries_grids_and_version(self):
        substrate, _, sidecar = _banach_setup()
        assert isinstance(sidecar, InverseLookupSidecar)
        assert sidecar.driver_profile_id == "banach"
        assert sidecar.version
        assert len(sidecar.tau_obs_grid) == len(TAU_OBS_GRID)
        assert len(sidecar.canonical_grid) == len(TAU_OBS_GRID)
        assert len(sidecar.substrate_grid) == len(TAU_OBS_GRID)

    def test_build_sidecar_convenience(self):
        sidecar = build_sidecar_for_banach()
        assert isinstance(sidecar, InverseLookupSidecar)
        assert len(sidecar.tau_obs_grid) == 80  # default logspace(-2, 4, 80)
