"""Dataset ports shared by concrete EEG adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import mne
    import pandas as pd


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """Resolved dataset retrieval settings."""

    dataset_id: str
    destination: Path
    subjects: tuple[str, ...] = ()
    runs: tuple[str, ...] = ()
    metadata_only: bool = False


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    """One immutable downloaded dataset artifact."""

    relative_path: str
    source_url: str
    size_bytes: int
    sha256: str
    source_sha256: str | None


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Checksummed evidence describing a concrete dataset retrieval."""

    schema_version: int
    dataset_id: str
    source_project: str
    source_resource: str
    license: str
    retrieved_at_utc: str
    subjects: tuple[str, ...]
    metadata_only: bool
    artifacts: tuple[ArtifactRecord, ...]


@dataclass(frozen=True, slots=True)
class EventTable:
    """Dataset-neutral event records."""

    rows: tuple[dict[str, Any], ...]


class EEGDatasetAdapter(Protocol):
    """Port that prevents dataset details leaking into scientific layers."""

    def fetch(self, config: DatasetConfig) -> DatasetManifest:
        """Fetch immutable source files and return a checksummed manifest."""

    def load_raw(self, subject: str, run: str | None = None) -> mne.io.BaseRaw:
        """Load one raw EEG recording."""

    def events(self, raw: mne.io.BaseRaw) -> EventTable:
        """Map raw annotations to a dataset-neutral event table."""

    def stimulus_table(self) -> pd.DataFrame:
        """Return validated stimulus metadata."""
