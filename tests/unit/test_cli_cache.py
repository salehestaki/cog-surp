from __future__ import annotations

import json
from pathlib import Path

from cog_surp.cli.app import _valid_artifact_manifest
from cog_surp.provenance.checksums import sha256_file


def test_valid_artifact_manifest_supports_single_and_multiple_outputs(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifacts": [
                    {"path": first.name, "sha256": sha256_file(first)},
                    {"path": second.name, "sha256": sha256_file(second)},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert _valid_artifact_manifest(tmp_path, manifest)

    manifest.write_text(
        json.dumps({"artifact": {"path": first.name, "sha256": sha256_file(first)}}),
        encoding="utf-8",
    )
    assert _valid_artifact_manifest(tmp_path, manifest)

    first.write_text("changed", encoding="utf-8")
    assert not _valid_artifact_manifest(tmp_path, manifest)
