"""Integration test: the three v2.0 seed-corpus driver profiles parse,
expose the lookup_table forward map, and pass an I5 round-trip on their
own canonical entries (handoff §D.2 / §E item 5).

Profiles live in H:/mpa-conform/output/seed-corpus/:
  - neural-population/driver-profile.json
  - ck-glassy/driver-profile.json
  - surface-code-qec/driver-profile.json

Skipped at collection time if the seed corpus is not present (e.g. someone
clones mpa-scale-solver standalone). The tests are pure consumers — they
do not write into mpa-conform.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mpa_scale_solver import (
    CanonicalState,
    apply_translation,
    forward_sweep_invert,
    parse_gamut,
    parse_translation_field,
    regime_at,
    validate_driver_profile,
)


SEED_PROFILES = ("neural-population", "ck-glassy", "surface-code-qec")


def _seed_path(seed_corpus_root: Path, profile: str) -> Path:
    return seed_corpus_root / profile / "driver-profile.json"


def _load_profile(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def _check_corpus(seed_corpus_root: Path):
    if not seed_corpus_root.exists():
        pytest.skip(f"seed corpus not present at {seed_corpus_root}")
    missing = [
        p for p in SEED_PROFILES if not _seed_path(seed_corpus_root, p).exists()
    ]
    if missing:
        pytest.skip(f"seed corpus missing profiles: {missing}")
    return seed_corpus_root


@pytest.mark.parametrize("profile", SEED_PROFILES)
class TestSeedCorpusProfiles:
    def test_parses(self, _check_corpus, profile):
        d = _load_profile(_seed_path(_check_corpus, profile))
        field = parse_translation_field(d["translation_field"])
        gamut = parse_gamut(d["gamut"])
        assert field.direction == "forward"
        assert field.shape == "lookup_table"
        assert len(field.rule) >= 1
        # gamut shape sanity
        assert gamut.chit_range[0] <= gamut.chit_range[1]

    def test_apply_translation_exact_match(self, _check_corpus, profile):
        """Every rule's canonical point maps to its own substrate cell."""
        d = _load_profile(_seed_path(_check_corpus, profile))
        field = parse_translation_field(d["translation_field"])
        for rule in field.rule:
            c = CanonicalState(
                chit=rule.canonical.chit,
                gamma_AB=rule.canonical.gamma_AB,
                k_frust=rule.canonical.k_frust,
            )
            sub = apply_translation(c, field, tau_obs=1.0)
            # Nearest rule for an exact canonical match should be a rule
            # with the same canonical. Multiple rules may share a canonical
            # (one per xdot_choice); any of them is acceptable.
            assert sub.observables["canonical_chit"] == pytest.approx(rule.canonical.chit)
            assert sub.observables["canonical_gamma_AB"] == pytest.approx(rule.canonical.gamma_AB)

    def test_round_trip_substrate_closure(self, _check_corpus, profile):
        """RFC-S §5 round-trip: recovered canonical's forward-projection
        lands on the same substrate cell as the truth's forward-projection.

        Discrete v0 lookup_table fields partition canonical space into
        Voronoi cells; multiple canonicals may share a cell. The cell
        (operating_point label) is the meaningful unit of round-trip
        closure — not the canonical scalar pair. Exact-canonical recovery
        is a v1 concern (residual_field + adaptive refinement).
        """
        from mpa_scale_solver import apply_translation, forward_sweep_invert

        d = _load_profile(_seed_path(_check_corpus, profile))
        field = parse_translation_field(d["translation_field"])

        seen_chits = sorted({rule.canonical.chit for rule in field.rule})
        seen_gammas = sorted({rule.canonical.gamma_AB for rule in field.rule})
        cg, gg = np.meshgrid(seen_chits, seen_gammas, indexing="ij")
        grid = np.column_stack([cg.ravel(), gg.ravel()])

        # Unique (chit, gamma) canonicals — k_frust does not enter the
        # nearest-neighbor lookup in v0.
        seen_canonicals = {(r.canonical.chit, r.canonical.gamma_AB)
                           for r in field.rule}

        closures = 0
        regime_matches = 0
        for chit, gamma in seen_canonicals:
            truth = CanonicalState(chit=chit, gamma_AB=gamma)
            predicted = apply_translation(truth, field, 1.0)
            recovered, _ = forward_sweep_invert(predicted, field, 1.0, grid)
            recovered_predicted = apply_translation(recovered, field, 1.0)
            # Substrate-cell closure: same operating_point label
            if recovered_predicted.label == predicted.label:
                closures += 1
            # I5 regime closure: 5-bucket regime agrees
            from mpa_scale_solver import regime_at
            if regime_at(truth, 1.0).regime == regime_at(recovered, 1.0).regime:
                regime_matches += 1

        n = len(seen_canonicals)
        assert closures == n, f"{closures}/{n} canonicals close the round trip"
        assert regime_matches == n, f"{regime_matches}/{n} regime agreements"
