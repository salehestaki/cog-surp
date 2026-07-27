from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from cog_surp.datasets.erp_core import BIDS_FOLDER, OSF_ROOT, ERPCoreN400Adapter
from cog_surp.datasets.osf import OSFClient
from cog_surp.domain.datasets import DatasetConfig


def _page(data: list[dict[str, Any]], next_url: str | None = None) -> bytes:
    return json.dumps({"data": data, "links": {"next": next_url}}).encode()


def _folder(name: str, related: str) -> dict[str, Any]:
    return {
        "id": "folder",
        "attributes": {
            "kind": "folder",
            "name": name,
            "materialized_path": f"/{BIDS_FOLDER}/{name}/",
        },
        "relationships": {"files": {"links": {"related": {"href": related}}}},
        "links": {},
    }


def _file(name: str, path: str, content: bytes) -> dict[str, Any]:
    return {
        "id": name,
        "attributes": {
            "kind": "file",
            "name": name,
            "materialized_path": path,
            "size": len(content),
            "extra": {"hashes": {"sha256": hashlib.sha256(content).hexdigest()}},
        },
        "relationships": {},
        "links": {"download": f"https://download.test/{name}"},
    }


def test_metadata_fetch_is_checksummed_and_excludes_signals(tmp_path: Path) -> None:
    folder_url = "https://api.test/bids"
    subject_url = "https://api.test/sub-001"
    metadata = b"onset\tduration\tvalue\n0\t0\t211\n"
    signal = b"large signal fixture"
    routes = {
        OSF_ROOT: _page([_folder(BIDS_FOLDER, folder_url)]),
        folder_url: _page([_folder("sub-001", subject_url)]),
        subject_url: _page(
            [
                _file(
                    "sub-001_task-N400_events.tsv",
                    f"/{BIDS_FOLDER}/sub-001/eeg/sub-001_task-N400_events.tsv",
                    metadata,
                ),
                _file(
                    "sub-001_task-N400_eeg.set",
                    f"/{BIDS_FOLDER}/sub-001/eeg/sub-001_task-N400_eeg.set",
                    signal,
                ),
            ]
        ),
        "https://download.test/sub-001_task-N400_events.tsv": metadata,
        "https://download.test/sub-001_task-N400_eeg.set": signal,
    }

    def transport(url: str) -> bytes:
        return routes[url]

    def downloader(url: str, destination: Path) -> None:
        destination.write_bytes(routes[url])

    adapter = ERPCoreN400Adapter(OSFClient(transport, downloader))
    manifest = adapter.fetch(
        DatasetConfig(
            dataset_id="erp-core-n400",
            destination=tmp_path,
            subjects=("001",),
            metadata_only=True,
        )
    )

    assert manifest.subjects == ("001",)
    assert len(manifest.artifacts) == 1
    assert manifest.artifacts[0].sha256 == hashlib.sha256(metadata).hexdigest()
    assert not list(tmp_path.rglob("*.set"))


def test_adapter_rejects_unknown_dataset(tmp_path: Path) -> None:
    adapter = ERPCoreN400Adapter()

    try:
        adapter.fetch(DatasetConfig("unknown", tmp_path))
    except ValueError as error:
        assert "unsupported dataset_id" in str(error)
    else:
        raise AssertionError("unknown dataset must be rejected")


def test_subject_filter_does_not_traverse_unselected_folder(tmp_path: Path) -> None:
    folder_url = "https://api.test/bids"
    selected_url = "https://api.test/sub-001"
    routes = {
        OSF_ROOT: _page([_folder(BIDS_FOLDER, folder_url)]),
        folder_url: _page(
            [
                _folder("sub-001", selected_url),
                _folder("sub-002", "https://api.test/must-not-be-called"),
            ]
        ),
        selected_url: _page(
            [
                _file(
                    "sub-001_task-N400_events.tsv",
                    f"/{BIDS_FOLDER}/sub-001/eeg/sub-001_task-N400_events.tsv",
                    b"events",
                )
            ]
        ),
        "https://download.test/sub-001_task-N400_events.tsv": b"events",
    }

    def transport(url: str) -> bytes:
        return routes[url]

    def downloader(url: str, destination: Path) -> None:
        destination.write_bytes(routes[url])

    manifest = ERPCoreN400Adapter(OSFClient(transport, downloader)).fetch(
        DatasetConfig(
            "erp-core-n400",
            tmp_path,
            subjects=("001",),
            metadata_only=True,
        )
    )

    assert manifest.subjects == ("001",)
