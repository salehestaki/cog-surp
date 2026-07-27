"""Bounded adapter for the public DERCo word-aligned reading corpus."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from cog_surp.domain.datasets import (
    ArtifactRecord,
    DatasetConfig,
    DatasetManifest,
    EventTable,
)
from cog_surp.provenance.checksums import sha256_file

from .osf import OSFClient, OSFFile

if TYPE_CHECKING:
    import mne
    import pandas as pd

OSF_ROOT = "https://api.osf.io/v2/nodes/rkqbu/files/osfstorage/"
PROJECT_URL = "https://github.com/Tayerquach/DERCo"
RESOURCE_URL = "https://osf.io/rkqbu/"
DATASET_LICENSE = (
    "NOASSERTION: public research dataset; OSF dataset license is not specified"
)


class DERCoAdapter:
    """Adapter for preprocessed DERCo word epochs and human predictions."""

    def __init__(self, client: OSFClient | None = None) -> None:
        self._client = client or OSFClient()
        self._root: Path | None = None

    def fetch(self, config: DatasetConfig) -> DatasetManifest:
        """Fetch selected subject/article epochs and corresponding predictions."""
        if config.dataset_id != "derco":
            raise ValueError(f"unsupported dataset_id: {config.dataset_id}")
        if not config.subjects:
            raise ValueError("DERCo fetch requires at least one explicit subject")
        articles = config.runs or ("article_0",)
        invalid = [name for name in articles if name not in _article_names()]
        if invalid:
            raise ValueError(f"invalid DERCo article IDs: {invalid}")
        dataset_root = config.destination / config.dataset_id
        dataset_root.mkdir(parents=True, exist_ok=True)

        eeg_root = self._client.find_folder(OSF_ROOT, "EEG-based Reading Experiment")
        eeg_data = self._client.find_folder(eeg_root, "EEG_data")
        preprocessed = self._client.find_folder(eeg_data, "preprocessed")
        behavioral = self._client.find_folder(
            OSF_ROOT, "Behavioural Word-Prediction Experiment"
        )
        prediction = self._client.find_folder(behavioral, "prediction")
        remotes: list[tuple[OSFFile, str]] = []

        prediction_files = {
            item.name: item for item in self._client.walk_files(prediction)
        }
        for article in articles:
            filename = f"human_prediction_{article}.csv"
            if filename not in prediction_files:
                raise LookupError(f"DERCo prediction file missing: {filename}")
            remotes.append((prediction_files[filename], f"prediction/{filename}"))

        found_subjects: set[str] = set()
        for subject in config.subjects:
            subject_folder = self._client.find_folder(preprocessed, subject)
            for article in articles:
                article_folder = self._client.find_folder(subject_folder, article)
                files = list(self._client.walk_files(article_folder))
                if len(files) != 1 or files[0].name != "preprocessed_epoch.fif":
                    raise LookupError(
                        f"unexpected DERCo files for {subject}/{article}: "
                        f"{[item.name for item in files]}"
                    )
                remotes.append(
                    (
                        files[0],
                        (
                            f"EEG_data/preprocessed/{subject}/{article}/"
                            "preprocessed_epoch.fif"
                        ),
                    )
                )
                found_subjects.add(subject)

        artifacts = tuple(
            sorted(
                (
                    self._download(remote, relative, dataset_root)
                    for remote, relative in remotes
                ),
                key=lambda item: item.relative_path,
            )
        )
        self._root = dataset_root
        return DatasetManifest(
            schema_version=1,
            dataset_id="derco",
            source_project=PROJECT_URL,
            source_resource=RESOURCE_URL,
            license=DATASET_LICENSE,
            retrieved_at_utc=datetime.now(UTC).isoformat(),
            subjects=tuple(sorted(found_subjects)),
            metadata_only=False,
            artifacts=artifacts,
        )

    def _download(
        self,
        remote: OSFFile,
        relative: str,
        dataset_root: Path,
    ) -> ArtifactRecord:
        target = dataset_root / Path(*relative.split("/"))
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
        return ArtifactRecord(
            relative_path=relative,
            source_url=remote.download_url,
            size_bytes=target.stat().st_size,
            sha256=digest,
            source_sha256=remote.sha256,
        )

    def load_epochs(self, subject: str, article: str) -> mne.Epochs:
        """Load one publisher-preprocessed, word-aligned epoch file."""
        if self._root is None:
            raise RuntimeError("fetch must be called before load_epochs")
        import mne

        source = (
            self._root
            / "EEG_data"
            / "preprocessed"
            / subject
            / article
            / "preprocessed_epoch.fif"
        )
        return mne.read_epochs(source, preload=False, verbose="ERROR")

    def load_raw(self, subject: str, run: str | None = None) -> mne.io.BaseRaw:
        """Explain why DERCo's released units cannot be treated as continuous raw."""
        del subject, run
        raise NotImplementedError(
            "DERCo releases article-level Epochs FIF, not continuous MNE Raw; "
            "use load_epochs(subject, article)"
        )

    def events(self, raw: mne.io.BaseRaw) -> EventTable:
        """DERCo event metadata are stored on Epochs rather than Raw."""
        del raw
        raise NotImplementedError("use the metadata attached to load_epochs()")

    def stimulus_table(self) -> pd.DataFrame:
        """Load and concatenate downloaded human prediction records."""
        if self._root is None:
            raise RuntimeError("fetch must be called before stimulus_table")
        import pandas as pd

        paths = sorted(
            (self._root / "prediction").glob("human_prediction_article_*.csv")
        )
        return pd.concat(
            [pd.read_csv(path).assign(source_file=path.name) for path in paths],
            ignore_index=True,
        )


def _article_names() -> frozenset[str]:
    return frozenset(f"article_{index}" for index in range(5))
