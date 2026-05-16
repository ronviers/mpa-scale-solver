"""Per-call provenance builder (handoff §C.6).

Stateless. Operations build a `Provenance` at the end of each call and
stamp it onto the `OperationOutput`. mpa-conform extracts these records
into the bundle's audit trail; mpa-auditor's display layer reads them
directly.
"""

from __future__ import annotations

import hashlib
import time
from typing import Iterable, Optional

from ._version import __version__
from .types import DispatchPath, Provenance


def make_provenance(
    operation: str,
    *,
    dispatch_path: DispatchPath = DispatchPath.DIRECT_COMPUTE,
    table_version: Optional[str] = None,
    notes: Iterable[str] = (),
) -> Provenance:
    """Stamp a `Provenance` for the current call.

    `timestamp_ns` is `time.monotonic_ns()` — monotonic, intra-process
    ordering. Reproducibility-sensitive consumers should ignore timestamps
    when comparing runs.
    """
    return Provenance(
        solver_version=__version__,
        operation=operation,
        timestamp_ns=time.monotonic_ns(),
        dispatch_path=dispatch_path,
        table_version=table_version,
        notes=tuple(notes),
    )


def provenance_hash(prov: Provenance) -> float:
    """Stable float32-encoded hash of a provenance record (handoff §A.5).

    Used by the EXR channel `provenance_hash` so frame-level dispatch
    fingerprints are queryable without unpacking the full provenance
    payload. Hash inputs are the version + operation + dispatch_path +
    table_version (timestamps and notes are excluded so the hash is
    reproducible across runs).
    """
    payload = "|".join((
        prov.solver_version,
        prov.operation,
        prov.dispatch_path.value,
        prov.table_version or "",
    )).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=4).digest()
    # Interpret 4 bytes as an unsigned int, then map onto [0, 1) for
    # safe float32 encoding in the EXR channel.
    return int.from_bytes(digest, "big") / 2**32
