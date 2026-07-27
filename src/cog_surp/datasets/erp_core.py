"""ERP CORE N400 dataset adapter."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from cog_surp.domain.datasets import (
    ArtifactRecord,
    DatasetConfig,
    DatasetManifest,
    EventTable,
)
from cog_surp.provenance.checksums import sha256_file
from cog_surp.stimuli.erp_core import load_erp_core_stimuli

from .osf import OSFClient, OSFFile

if TYPE_CHECKING:
    import mne
    import pandas as pd

OSF_ROOT = "https://api.osf.io/v2/nodes/29xpq/files/osfstorage/"
BIDS_FOLDER = "N400 Raw Data BIDS-Compatible"
PROJECT_URL = "https://erpinfo.org/erp-core"
RESOURCE_URL = "https://osf.io/29xpq/"


def _subject_from_path(path: str) -> str | None:
    for part in PurePosixPath(path).parts:
        if part.startswith("sub-"):
            return part.removeprefix("sub-")
    return None


def _selected(file: OSFFile, config: DatasetConfig) -> bool:
    subject = _subject_from_path(file.materialized_path)
    if subject and config.subjects and subject not in config.subjects:
        return False
    if config.metadata_only and file.name.endswith((".fdt", ".set")):
        return False
    return True


def _descend(path: str, subjects: tuple[str, ...]) -> bool:
    subject = _subject_from_path(path)
    return not subject or not subjects or subject in subjects


class ERPCoreN400Adapter:
    """Adapter for the public ERP CORE N400 BIDS-compatible release."""

    def __init__(self, client: OSFClient | None = None) -> None:
        self._client = client or OSFClient()
        self._root: Path | None = None

    def fetch(self, config: DatasetConfig) -> DatasetManifest:
        """Fetch a bounded or complete immutable ERP CORE N400 snapshot."""
        if config.dataset_id != "erp-core-n400":
            raise ValueError(f"unsupported dataset_id: {config.dataset_id}")
        dataset_root = config.destination / config.dataset_id
        dataset_root.mkdir(parents=True, exist_ok=True)
        folder_url = self._client.find_folder(OSF_ROOT, BIDS_FOLDER)
        artifacts: list[ArtifactRecord] = []
        found_subjects: set[str] = set()

        for remote in self._client.walk_files(
            folder_url,
            descend=lambda path: _descend(path, config.subjects),
        ):
            if not _selected(remote, config):
                continue
            relative = _relative_bids_path(remote.materialized_path)
            target = dataset_root / Path(*PurePosixPath(relative).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or (
                remote.sha256 and sha256_file(target) != remote.sha256
            ):
                temporary = target.with_suffix(target.suffix + ".part")
                self._client.download(remote.download_url, temporary)
                temporary.replace(target)
            digest = sha256_file(target)
            if remote.sha256 and digest != remote.sha256:
                raise OSError(
                    f"checksum mismatch for {relative}: {digest} != {remote.sha256}"
                )
            subject = _subject_from_path(remote.materialized_path)
            if subject:
                found_subjects.add(subject)
            artifacts.append(
                ArtifactRecord(
                    relative_path=relative,
                    source_url=remote.download_url,
                    size_bytes=target.stat().st_size,
                    sha256=digest,
                    source_sha256=remote.sha256,
                )
            )

        requested = set(config.subjects)
        if requested and requested - found_subjects:
            missing = ", ".join(sorted(requested - found_subjects))
            raise LookupError(f"subjects not present in ERP CORE N400: {missing}")
        self._root = dataset_root
        return DatasetManifest(
            schema_version=1,
            dataset_id=config.dataset_id,
            source_project=PROJECT_URL,
            source_resource=RESOURCE_URL,
            license="CC-BY-SA-4.0",
            retrieved_at_utc=datetime.now(UTC).isoformat(),
            subjects=tuple(sorted(found_subjects)),
            metadata_only=config.metadata_only,
            artifacts=tuple(sorted(artifacts, key=lambda item: item.relative_path)),
        )

    def load_raw(self, subject: str, run: str | None = None) -> mne.io.BaseRaw:
        """Load the EEGLAB recording for a fetched subject."""
        del run
        if self._root is None:
            raise RuntimeError("fetch must be called before load_raw")
        try:
            import mne
        except ImportError as error:
            raise RuntimeError(
                "EEG support is not installed; run `uv sync --extra eeg`"
            ) from error
        source = (
            self._root / f"sub-{subject}" / "eeg" / f"sub-{subject}_task-N400_eeg.set"
        )
        if not source.exists():
            raise FileNotFoundError(source)
        return mne.io.read_raw_eeglab(source, preload=False)

    def events(self, raw: mne.io.BaseRaw) -> EventTable:
        """Convert MNE annotations to stable event records."""
        rows = tuple(
            {
                "onset_s": float(onset),
                "duration_s": float(duration),
                "description": str(description),
            }
            for onset, duration, description in zip(
                raw.annotations.onset,
                raw.annotations.duration,
                raw.annotations.description,
                strict=True,
            )
        )
        return EventTable(rows)

    def stimulus_table(self) -> pd.DataFrame:
        """Load the two publisher-supplied English stimulus lists."""
        if self._root is None:
            raise RuntimeError("fetch must be called before stimulus_table")
        return load_erp_core_stimuli(self._root)


def _relative_bids_path(materialized_path: str) -> str:
    marker = f"/{BIDS_FOLDER}/"
    if marker not in materialized_path:
        raise ValueError(f"unexpected ERP CORE path: {materialized_path}")
    return materialized_path.split(marker, maxsplit=1)[1]
