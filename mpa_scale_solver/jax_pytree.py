"""PyTree registrations for the canonical dataclasses.

JAX traces through PyTrees — nested containers of leaves (arrays / scalars).
A frozen dataclass is opaque to JAX unless explicitly registered: without
this, `jax.grad(f)(CanonicalState(...))` raises because JAX cannot find
the differentiable leaves.

Registration policy:
  - `CanonicalState`: leaves = `(chit, gamma_AB)` (the differentiable
    canonical-frame coordinates). `k_frust` is aux (a topological bool;
    not differentiable — the framework's scale-relativity invariant).

`SubstrateState` is intentionally NOT registered as a PyTree. Its
`observables` and `axes` dicts hold mixed types (strings, ints, floats)
and have no fixed leaf schema; differentiable consumers go through
`jax_ops`' tuple-returning surfaces instead of round-tripping
`SubstrateState`.

Idempotent: importing this module multiple times is safe (JAX raises
on duplicate registration; we catch and ignore).
"""

from __future__ import annotations

from typing import Tuple

import jax

from .types import CanonicalState


def _canonical_state_flatten(state: CanonicalState) -> Tuple[tuple, tuple]:
    """Leaves = (chit, gamma_AB); aux = (k_frust,)."""
    leaves = (state.chit, state.gamma_AB)
    aux = (state.k_frust,)
    return leaves, aux


def _canonical_state_unflatten(aux: tuple, leaves: tuple) -> CanonicalState:
    (k_frust,) = aux
    chit, gamma_AB = leaves
    return CanonicalState(chit=chit, gamma_AB=gamma_AB, k_frust=k_frust)


try:
    jax.tree_util.register_pytree_node(
        CanonicalState,
        _canonical_state_flatten,
        _canonical_state_unflatten,
    )
except ValueError:
    # Already registered (e.g., module reloaded). No-op.
    pass
