"""Canonical JSON serialization for run and dataset manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from cog_surp.domain.datasets import DatasetManifest


def canonical_json_bytes(value: object) -> bytes:
    """Serialize a value in the canonical form used for artifact hashes."""
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()


def write_dataset_manifest(manifest: DatasetManifest, path: Path) -> str:
    """Atomically write a dataset manifest and return its SHA-256 digest."""
    payload = canonical_json_bytes(asdict(manifest))
    digest = hashlib.sha256(payload).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return digest
