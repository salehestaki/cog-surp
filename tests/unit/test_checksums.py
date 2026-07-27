from __future__ import annotations

import hashlib
from pathlib import Path

from cog_surp.provenance.checksums import sha256_file


def test_sha256_file(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"cog-surp")

    assert sha256_file(artifact) == hashlib.sha256(b"cog-surp").hexdigest()
