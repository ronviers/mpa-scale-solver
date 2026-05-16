"""v2.3 — I1–I5 intents + composition algebra (BLOCK_IN cut d, RFC-S §3).

Acceptance criteria from BLOCK_IN §v2.3:
  - Each intent's invariance check fires (or doesn't) per its named
    invariant on the sacrifice dict.
  - Composition algebra holds: independent intents compose; I2 (drive-
    faithful) does not compose with adjusting intents.

v1's I5 sacrifice-dict back-compat is exercised by `test_operations.py
:: TestIntentMap`, which stays green; this module covers the new I1–I4
surface and the composition algebra.
"""

from __future__ import annotations

import pytest

from mpa_scale_solver import (
    CanonicalState,
    GamutSpec,
    intent_compose,
    intent_compose_wrapped,
    intent_map,
    intent_map_wrapped,
)


def _gamut(chit=(-1.0, 1.0), gamma=(-1.0, 1.0)) -> GamutSpec:
    return GamutSpec(chit_range=chit, gamma_AB_range=gamma)


# ---------------------------------------------------------------------------
# I1 regime-preserving — regime ∧ sign(gamma_AB) ∧ k_frust
# ---------------------------------------------------------------------------

class TestI1RegimePreserving:
    def test_clamp_preserves_regime_when_possible(self):
        # chit=2.0 is deep_c (>=0.7); gamut chit_max=1.0 is still deep_c.
        mapped, sac = intent_map(
            CanonicalState(chit=2.0, gamma_AB=0.3), 1.0, _gamut(), "I1",
        )
        assert sac["invariant_preserved"] is True
        assert sac["regime_preserved"] is True
        assert sac["original_regime"] == "deep_c"
        assert sac["mapped_regime"] == "deep_c"
        assert mapped.chit == 1.0  # clamped to gamut max

    def test_regime_unreachable_flags_sacrifice(self):
        # deep_c (chit>=0.7) cannot live in gamut chit in [-0.5, 0.5].
        mapped, sac = intent_map(
            CanonicalState(chit=2.0, gamma_AB=0.0), 1.0,
            _gamut(chit=(-0.5, 0.5)), "I1",
        )
        assert sac["invariant_preserved"] is False
        assert sac["regime_preserved"] is False
        assert sac["original_regime"] == "deep_c"
        assert sac["mapped_regime"] != "deep_c"
        # naive clamp falls back to gamut max
        assert mapped.chit == 0.5

    def test_in_gamut_no_change(self):
        mapped, sac = intent_map(
            CanonicalState(chit=0.3, gamma_AB=-0.2), 1.0, _gamut(), "I1",
        )
        assert sac["invariant_preserved"] is True
        assert mapped.chit == 0.3
        assert mapped.gamma_AB == -0.2

    def test_gamma_sign_flip_flags_sacrifice(self):
        # gamma=+0.5 (positive), gamut gamma in [-1.0, -0.1] (negative only).
        mapped, sac = intent_map(
            CanonicalState(chit=0.0, gamma_AB=0.5), 1.0,
            _gamut(gamma=(-1.0, -0.1)), "I1",
        )
        assert sac["invariant_preserved"] is False
        assert sac["gamma_AB_sign_preserved"] is False
        assert sac["original_gamma_AB_sign"] == 1
        assert sac["mapped_gamma_AB_sign"] == -1

    def test_k_frust_propagated(self):
        mapped, sac = intent_map(
            CanonicalState(chit=2.0, gamma_AB=0.0, k_frust=True), 1.0, _gamut(), "I1",
        )
        assert mapped.k_frust is True
        assert sac["k_frust_preserved"] is True


# ---------------------------------------------------------------------------
# I2 drive-faithful — no adjustment; completeness sacrifice on OOG
# ---------------------------------------------------------------------------

class TestI2DriveFaithful:
    def test_in_gamut_passthrough(self):
        state = CanonicalState(chit=0.3, gamma_AB=-0.2)
        mapped, sac = intent_map(state, 1.0, _gamut(), "I2")
        assert mapped == state
        assert sac["invariant_preserved"] is True
        assert sac["out_of_gamut_rejected"] is False

    def test_out_of_gamut_unchanged_and_flagged(self):
        state = CanonicalState(chit=2.0, gamma_AB=0.0)
        mapped, sac = intent_map(state, 1.0, _gamut(), "I2")
        assert mapped == state  # NOT clamped
        assert sac["invariant_preserved"] is False
        assert sac["out_of_gamut_rejected"] is True
        assert "chit" in sac["out_of_gamut_axes"]
        assert sac["delta_chit"] == 0.0

    def test_both_axes_oog_listed(self):
        state = CanonicalState(chit=2.0, gamma_AB=2.0)
        _, sac = intent_map(state, 1.0, _gamut(), "I2")
        assert "chit" in sac["out_of_gamut_axes"]
        assert "gamma_AB" in sac["out_of_gamut_axes"]


# ---------------------------------------------------------------------------
# I3 capacity-preserving — capacity class ∧ k_frust
# ---------------------------------------------------------------------------

class TestI3CapacityPreserving:
    def test_deep_state_preserves_capacity_class(self):
        # chit=0.9 is deep; gamut chit_max=1.0 keeps it deep.
        mapped, sac = intent_map(
            CanonicalState(chit=0.9, gamma_AB=0.0), 1.0, _gamut(), "I3",
        )
        assert sac["invariant_preserved"] is True
        assert sac["capacity_class"] == "deep"
        assert sac["mapped_capacity_class"] == "deep"
        assert mapped.chit == 0.9

    def test_deep_state_demoted_to_shallow_flags(self):
        # gamut chit_max=0.5 forces deep->shallow.
        mapped, sac = intent_map(
            CanonicalState(chit=0.9, gamma_AB=0.0), 1.0,
            _gamut(chit=(-0.5, 0.5)), "I3",
        )
        assert sac["invariant_preserved"] is False
        assert sac["capacity_class"] == "deep"
        assert sac["mapped_capacity_class"] == "shallow"
        assert mapped.chit == 0.5

    def test_deep_state_clamped_to_gamut_edge_when_still_deep(self):
        # chit=2.0 deep_c, gamut [-1.0, 1.0] -> clamp to 1.0 (still deep).
        mapped, sac = intent_map(
            CanonicalState(chit=2.0, gamma_AB=0.0), 1.0, _gamut(), "I3",
        )
        assert sac["invariant_preserved"] is True
        assert mapped.chit == 1.0
        assert sac["capacity_class"] == "deep"
        assert sac["mapped_capacity_class"] == "deep"

    def test_k_frust_propagated(self):
        mapped, sac = intent_map(
            CanonicalState(chit=0.9, gamma_AB=0.0, k_frust=True), 1.0, _gamut(), "I3",
        )
        assert mapped.k_frust is True
        assert sac["k_frust"] is True


# ---------------------------------------------------------------------------
# I4 persistence-preserving — sign(gamma_AB)
# ---------------------------------------------------------------------------

class TestI4PersistencePreserving:
    def test_positive_gamma_kept_positive(self):
        mapped, sac = intent_map(
            CanonicalState(chit=0.0, gamma_AB=2.0), 1.0, _gamut(), "I4",
        )
        assert sac["invariant_preserved"] is True
        assert sac["original_gamma_AB_sign"] == 1
        assert sac["mapped_gamma_AB_sign"] == 1
        assert mapped.gamma_AB == 1.0  # clamped to gamut max but still positive

    def test_sign_flip_flagged_when_gamut_excludes_sign(self):
        # gamma=+0.5, gamut gamma in [-1.0, -0.1] (negative only).
        mapped, sac = intent_map(
            CanonicalState(chit=0.0, gamma_AB=0.5), 1.0,
            _gamut(gamma=(-1.0, -0.1)), "I4",
        )
        assert sac["invariant_preserved"] is False
        assert sac["original_gamma_AB_sign"] == 1
        assert sac["mapped_gamma_AB_sign"] == -1

    def test_zero_gamma_treated_as_signless(self):
        mapped, sac = intent_map(
            CanonicalState(chit=0.0, gamma_AB=0.0), 1.0, _gamut(), "I4",
        )
        assert sac["invariant_preserved"] is True
        assert mapped.gamma_AB == 0.0


# ---------------------------------------------------------------------------
# I5 — v2.3 adds uniform keys; v1 keys preserved (covered in test_operations.py)
# ---------------------------------------------------------------------------

class TestI5UniformKeys:
    def test_carries_v23_invariant_keys(self):
        gamut = _gamut()
        _, sac = intent_map(
            CanonicalState(chit=2.0, gamma_AB=0.0), 1.0, gamut, "I5",
        )
        assert sac["preserved_invariant"] == "regime_label"
        assert "invariant_preserved" in sac
        # v1 keys must still be present (regression):
        assert "regime_preserved" in sac
        assert sac["invariant_preserved"] == sac["regime_preserved"]


# ---------------------------------------------------------------------------
# Composition algebra (RFC-S §3)
# ---------------------------------------------------------------------------

class TestComposition:
    def test_single_intent_compose_equals_intent_map(self):
        state = CanonicalState(chit=2.0, gamma_AB=0.5)
        mapped_a, sacs = intent_compose(state, 1.0, _gamut(), ["I5"])
        mapped_b, sac = intent_map(state, 1.0, _gamut(), "I5")
        assert mapped_a == mapped_b
        assert len(sacs) == 1
        assert sacs[0]["regime_preserved"] == sac["regime_preserved"]

    def test_idempotent_under_same_intent(self):
        # Applying I3 twice = applying I3 once (since the first lands in-gamut).
        state = CanonicalState(chit=2.0, gamma_AB=0.0)
        mapped_once, _ = intent_compose(state, 1.0, _gamut(), ["I3"])
        mapped_twice, _ = intent_compose(state, 1.0, _gamut(), ["I3", "I3"])
        assert mapped_once == mapped_twice

    def test_i1_then_i3_composable(self):
        # Independent invariants — both should hold on the output.
        state = CanonicalState(chit=2.0, gamma_AB=0.3)
        mapped, sacs = intent_compose(state, 1.0, _gamut(), ["I1", "I3"])
        assert len(sacs) == 2
        # Both intents preserved their invariants on a benign gamut.
        assert sacs[0]["invariant_preserved"] is True
        assert sacs[1]["invariant_preserved"] is True
        # Final state is in gamut.
        assert -1.0 <= mapped.chit <= 1.0
        assert -1.0 <= mapped.gamma_AB <= 1.0

    def test_i3_then_i4_composable(self):
        state = CanonicalState(chit=0.9, gamma_AB=2.0)
        mapped, sacs = intent_compose(state, 1.0, _gamut(), ["I3", "I4"])
        assert sacs[0]["invariant_preserved"] is True  # capacity preserved
        assert sacs[1]["invariant_preserved"] is True  # sign preserved
        assert mapped.chit == 0.9
        assert mapped.gamma_AB == 1.0  # positive, in gamut

    def test_i2_does_not_compose(self):
        gamut = _gamut()
        with pytest.raises(ValueError, match="does not compose"):
            intent_compose(
                CanonicalState(chit=0.0, gamma_AB=0.0), 1.0, gamut, ["I1", "I2"],
            )
        with pytest.raises(ValueError, match="does not compose"):
            intent_compose(
                CanonicalState(chit=0.0, gamma_AB=0.0), 1.0, gamut, ["I2", "I1"],
            )

    def test_i2_alone_is_legal(self):
        state = CanonicalState(chit=0.3, gamma_AB=-0.2)
        mapped, sacs = intent_compose(state, 1.0, _gamut(), ["I2"])
        assert mapped == state
        assert len(sacs) == 1
        assert sacs[0]["intent"] == "I2"

    def test_empty_intents_raises(self):
        with pytest.raises(ValueError, match="requires at least one"):
            intent_compose(
                CanonicalState(chit=0.0, gamma_AB=0.0), 1.0, _gamut(), [],
            )

    def test_unknown_intent_raises(self):
        with pytest.raises(ValueError, match="unknown intent"):
            intent_compose(
                CanonicalState(chit=0.0, gamma_AB=0.0), 1.0, _gamut(), ["I1", "I99"],
            )

    def test_conflict_surfaces_in_sacrifice_trace(self):
        # I1 on a deep_c state in a chit=[-0.5, 0.5] gamut breaks regime.
        # Composing I1 after that exposes the I1 invariant break.
        gamut = _gamut(chit=(-0.5, 0.5))
        state = CanonicalState(chit=2.0, gamma_AB=0.0)
        _, sacs = intent_compose(state, 1.0, gamut, ["I1", "I3"])
        # I1 should fail (regime can't survive); I3 should pass on a now
        # in-gamut shallow state.
        assert sacs[0]["invariant_preserved"] is False
        assert sacs[1]["invariant_preserved"] is True


# ---------------------------------------------------------------------------
# Validation: intent invariants surface through OperationOutput
# ---------------------------------------------------------------------------

class TestValidation:
    def test_intent_map_wrapped_reports_invariant(self):
        # I3 demoting deep->shallow flags k_frust_invariant=False in the
        # wrapped report (the field is repurposed as the intent invariant).
        out = intent_map_wrapped(
            CanonicalState(chit=0.9, gamma_AB=0.0), 1.0,
            _gamut(chit=(-0.5, 0.5)), "I3",
        )
        assert out.validation.k_frust_invariant is False
        assert any("I3" in n for n in out.validation.notes)

    def test_intent_compose_wrapped_aggregates(self):
        out = intent_compose_wrapped(
            CanonicalState(chit=0.9, gamma_AB=2.0), 1.0, _gamut(), ["I3", "I4"],
        )
        assert out.validation.k_frust_invariant is True
        assert out.provenance.operation == "intent_compose"
        assert any("I3" in n or "I4" in n or "intents=" in n
                   for n in out.provenance.notes)

    def test_intent_compose_wrapped_failure_flagged(self):
        out = intent_compose_wrapped(
            CanonicalState(chit=2.0, gamma_AB=0.5), 1.0,
            _gamut(chit=(-0.5, 0.5), gamma=(-1.0, -0.1)),
            ["I1", "I3"],
        )
        # I1 cannot preserve regime AND sign; I3 cannot preserve deep.
        # Both fail -> aggregate False.
        assert out.validation.k_frust_invariant is False
