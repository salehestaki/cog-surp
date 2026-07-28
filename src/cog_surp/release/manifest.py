"""Authoritative, self-contained release manifests for reports and dashboards."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from cog_surp import __version__
from cog_surp.provenance.checksums import sha256_file
from cog_surp.provenance.manifests import canonical_json_bytes
from cog_surp.provenance.runtime import collect_runtime_provenance

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$"),
]


class ManifestValidationError(ValueError):
    """A release manifest is unsafe, corrupt, or internally inconsistent."""


class DataStatus(StrEnum):
    """Scientific evidence status carried by a release or panel artifact."""

    REAL = "real"
    SYNTHETIC = "synthetic"
    MIXED = "mixed"


class ArtifactType(StrEnum):
    """Known release artifact roles; unknown roles are rejected by Pydantic."""

    FEATURES = "features"
    LM_SURPRISAL = "lm-surprisal"
    PREDICTIVE_SUMMARY = "predictive-summary"
    POSTERIOR_SUMMARY = "posterior-summary"
    DIAGNOSTICS = "diagnostics"
    ROBUSTNESS = "robustness"
    H1_EFFECT = "h1-effect"
    H1_CONDITION_ERP = "h1-condition-erp"
    H1_DIFFERENCE_WAVE = "h1-difference-wave"
    H1_TOPOMAP = "h1-topomap"
    H1_PARTICIPANT_QC = "h1-participant-qc"
    H2_EFFECT = "h2-effect"
    CAUSAL_AUDIT = "causal-audit"
    CAUSAL_GRAPH = "causal-graph"
    CLUSTER_METADATA = "cluster-metadata"
    CLUSTER_SUMMARY = "cluster-summary"
    CLUSTER_FIGURE = "cluster-figure"
    REPORT = "report"


DASHBOARD_REQUIRED_ARTIFACT_TYPES = frozenset(
    {
        ArtifactType.FEATURES,
        ArtifactType.PREDICTIVE_SUMMARY,
        ArtifactType.POSTERIOR_SUMMARY,
        ArtifactType.DIAGNOSTICS,
        ArtifactType.ROBUSTNESS,
        ArtifactType.H1_EFFECT,
        ArtifactType.H1_CONDITION_ERP,
        ArtifactType.H1_DIFFERENCE_WAVE,
        ArtifactType.H1_TOPOMAP,
        ArtifactType.H1_PARTICIPANT_QC,
        ArtifactType.H2_EFFECT,
        ArtifactType.CAUSAL_AUDIT,
        ArtifactType.CAUSAL_GRAPH,
        ArtifactType.CLUSTER_METADATA,
        ArtifactType.CLUSTER_SUMMARY,
        ArtifactType.CLUSTER_FIGURE,
        ArtifactType.REPORT,
    }
)

_FEATURE_ANALYSES = {
    ArtifactType.PREDICTIVE_SUMMARY,
    ArtifactType.POSTERIOR_SUMMARY,
    ArtifactType.DIAGNOSTICS,
    ArtifactType.ROBUSTNESS,
}
_H1_ARTIFACTS = {
    ArtifactType.H1_EFFECT,
    ArtifactType.H1_CONDITION_ERP,
    ArtifactType.H1_DIFFERENCE_WAVE,
    ArtifactType.H1_TOPOMAP,
    ArtifactType.H1_PARTICIPANT_QC,
}


class DatasetReference(BaseModel):
    """Immutable dataset identity and license-aware citation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: Identifier
    version: str
    sha256: Sha256
    data_status: DataStatus
    citation: str


class ModelReference(BaseModel):
    """Pinned model/tokenizer identity represented in the release."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str
    revision: str
    tokenizer_revision: str
    scoring_run_ids: list[Identifier] = Field(min_length=1)


class RunLineage(BaseModel):
    """Run IDs that define the coherent analysis family."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    preprocessing_run_id: Identifier
    feature_run_id: Identifier
    lm_scoring_run_ids: list[Identifier] = Field(min_length=1)
    statistical_analysis_run_ids: list[Identifier] = Field(min_length=1)
    causal_analysis_run_ids: list[Identifier] = Field(min_length=1)
    report_run_id: Identifier


class ReleaseArtifact(BaseModel):
    """One checksummed file and its panel-level scientific lineage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_id: Identifier
    artifact_type: ArtifactType
    path: str
    sha256: Sha256
    schema_version: int = Field(default=1, ge=1, le=1)
    data_status: DataStatus
    source_run_id: Identifier
    parent_artifact_ids: list[Identifier] = Field(default_factory=list)
    parent_run_ids: list[Identifier] = Field(default_factory=list)
    dataset_id: Identifier | None = None
    model_id: str | None = None
    label: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_safe_relative_path(self) -> ReleaseArtifact:
        path = PurePosixPath(self.path)
        if path.is_absolute() or ".." in path.parts or "\\" in self.path:
            raise ValueError(
                "artifact paths must be POSIX-style, relative, and cannot contain '..'"
            )
        if not path.parts or path.parts[0] == ".":
            raise ValueError("artifact path must name a file inside the bundle")
        if self.data_status is DataStatus.MIXED:
            raise ValueError(
                "individual artifacts must be labeled real or synthetic, not mixed"
            )
        return self


class ReleaseManifest(BaseModel):
    """Validated source of truth for one report/dashboard release bundle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_schema_version: Literal[1] = 1
    project_version: str
    release_id: Identifier
    label: str = Field(min_length=1)
    created_at_utc: datetime
    git_commit: str
    git_dirty: bool
    source_implementation_sha256: Sha256
    resolved_configuration_hashes: dict[Identifier, Sha256]
    data_status: DataStatus
    datasets: list[DatasetReference] = Field(min_length=1)
    models: list[ModelReference] = Field(min_length=1)
    runs: RunLineage
    parent_artifact_ids: list[Identifier]
    artifacts: list[ReleaseArtifact] = Field(min_length=1)
    citations: list[str] = Field(min_length=1)
    known_limitations: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_release_consistency(self) -> ReleaseManifest:
        if self.project_version != __version__:
            raise ValueError(
                f"manifest project version {self.project_version!r} does not "
                f"match installed Cog-Surp {__version__!r}"
            )
        artifact_by_id = {artifact.artifact_id: artifact for artifact in self.artifacts}
        if len(artifact_by_id) != len(self.artifacts):
            raise ValueError("artifact IDs must be unique")
        known_ids = set(artifact_by_id)
        for artifact in self.artifacts:
            unknown_parents = set(artifact.parent_artifact_ids) - known_ids
            if unknown_parents:
                raise ValueError(
                    f"artifact {artifact.artifact_id!r} has unknown parents: "
                    f"{sorted(unknown_parents)}"
                )
            if artifact.artifact_id in artifact.parent_artifact_ids:
                raise ValueError(
                    f"artifact {artifact.artifact_id!r} cannot parent itself"
                )
        if set(self.parent_artifact_ids) - known_ids:
            raise ValueError("release parent_artifact_ids contain unknown artifacts")

        dataset_ids = {dataset.dataset_id for dataset in self.datasets}
        if len(dataset_ids) != len(self.datasets):
            raise ValueError("dataset IDs must be unique")
        model_ids = {model.model_id for model in self.models}
        if len(model_ids) != len(self.models):
            raise ValueError("model IDs must be unique")
        configured_lm_runs = set(self.runs.lm_scoring_run_ids)
        for model in self.models:
            unknown_runs = set(model.scoring_run_ids) - configured_lm_runs
            if unknown_runs:
                raise ValueError(
                    f"model {model.model_id!r} has unregistered scoring runs: "
                    f"{sorted(unknown_runs)}"
                )
        for artifact in self.artifacts:
            if (
                artifact.dataset_id is not None
                and artifact.dataset_id not in dataset_ids
            ):
                raise ValueError(
                    f"artifact {artifact.artifact_id!r} references unknown dataset "
                    f"{artifact.dataset_id!r}"
                )
            if artifact.model_id is not None and artifact.model_id not in model_ids:
                raise ValueError(
                    f"artifact {artifact.artifact_id!r} references unknown model "
                    f"{artifact.model_id!r}"
                )

        statuses = {artifact.data_status for artifact in self.artifacts}
        if self.data_status is DataStatus.REAL and statuses != {DataStatus.REAL}:
            raise ValueError("a real release can contain only real artifacts")
        if self.data_status is DataStatus.SYNTHETIC and statuses != {
            DataStatus.SYNTHETIC
        }:
            raise ValueError("a synthetic release can contain only synthetic artifacts")
        if self.data_status is DataStatus.MIXED and not {
            DataStatus.REAL,
            DataStatus.SYNTHETIC,
        }.issubset(statuses):
            raise ValueError(
                "a mixed release must contain labeled real and synthetic artifacts"
            )

        type_counts: dict[ArtifactType, int] = {}
        for artifact in self.artifacts:
            type_counts[artifact.artifact_type] = (
                type_counts.get(artifact.artifact_type, 0) + 1
            )
        missing_types = DASHBOARD_REQUIRED_ARTIFACT_TYPES - set(type_counts)
        if missing_types:
            raise ValueError(
                "release lacks dashboard artifacts: "
                f"{sorted(value.value for value in missing_types)}"
            )
        repeated_required = sorted(
            artifact_type.value
            for artifact_type in DASHBOARD_REQUIRED_ARTIFACT_TYPES
            if type_counts.get(artifact_type) != 1
        )
        if repeated_required:
            raise ValueError(
                f"dashboard artifact types must occur exactly once: {repeated_required}"
            )

        self._validate_run_lineage()
        return self

    def _validate_run_lineage(self) -> None:
        artifacts = self.artifacts
        runs = self.runs
        lm_runs = set(runs.lm_scoring_run_ids)
        statistical_runs = set(runs.statistical_analysis_run_ids)
        causal_runs = set(runs.causal_analysis_run_ids)

        feature = next(
            artifact
            for artifact in artifacts
            if artifact.artifact_type is ArtifactType.FEATURES
        )
        if feature.source_run_id != runs.feature_run_id:
            raise ValueError("feature artifact does not match feature_run_id")
        if not lm_runs.intersection(feature.parent_run_ids):
            raise ValueError(
                "feature artifact must name a configured LM scoring parent run"
            )

        surprisal_ids = {
            artifact.artifact_id
            for artifact in artifacts
            if artifact.artifact_type is ArtifactType.LM_SURPRISAL
        }
        if surprisal_ids and not surprisal_ids.intersection(
            feature.parent_artifact_ids
        ):
            raise ValueError(
                "feature artifact must depend on an included LM surprisal artifact"
            )

        for artifact in artifacts:
            if artifact.artifact_type in _FEATURE_ANALYSES:
                if artifact.source_run_id not in statistical_runs:
                    raise ValueError(
                        f"{artifact.artifact_id!r} has an unregistered statistical run"
                    )
                if runs.feature_run_id not in artifact.parent_run_ids:
                    raise ValueError(
                        f"{artifact.artifact_id!r} does not descend from feature run "
                        f"{runs.feature_run_id!r}"
                    )
            if artifact.artifact_type in _H1_ARTIFACTS:
                if artifact.source_run_id != runs.preprocessing_run_id:
                    raise ValueError(
                        f"{artifact.artifact_id!r} does not match preprocessing run"
                    )
            if artifact.artifact_type is ArtifactType.H2_EFFECT:
                if artifact.source_run_id not in statistical_runs:
                    raise ValueError("H2 effect has an unregistered analysis run")
                h2_lm_parents = lm_runs.intersection(artifact.parent_run_ids)
                if not h2_lm_parents:
                    raise ValueError("H2 effect lacks a configured LM scoring parent")
                if set(artifact.parent_run_ids) - lm_runs:
                    raise ValueError("H2 effect has an unregistered LM scoring parent")
            if artifact.artifact_type in {
                ArtifactType.CAUSAL_AUDIT,
                ArtifactType.CAUSAL_GRAPH,
            }:
                if artifact.source_run_id not in causal_runs:
                    raise ValueError("causal artifact has an unregistered causal run")
                parents = set(artifact.parent_run_ids)
                if runs.preprocessing_run_id not in parents or not lm_runs.intersection(
                    parents
                ):
                    raise ValueError(
                        "causal artifact lacks preprocessing or LM scoring parents"
                    )
                unknown = parents - {runs.preprocessing_run_id, *lm_runs}
                if unknown:
                    raise ValueError(
                        "causal artifact has unregistered parent runs: "
                        f"{sorted(unknown)}"
                    )
            if artifact.artifact_type in {
                ArtifactType.CLUSTER_METADATA,
                ArtifactType.CLUSTER_SUMMARY,
                ArtifactType.CLUSTER_FIGURE,
            }:
                if artifact.source_run_id not in statistical_runs:
                    raise ValueError(
                        "cluster artifact has an unregistered analysis run"
                    )
                if runs.preprocessing_run_id not in artifact.parent_run_ids:
                    raise ValueError(
                        "cluster artifact does not descend from preprocessing run"
                    )
            if artifact.artifact_type is ArtifactType.REPORT:
                if artifact.source_run_id != runs.report_run_id:
                    raise ValueError("report artifact does not match report_run_id")
                if not set(self.parent_artifact_ids).issubset(
                    artifact.parent_artifact_ids
                ):
                    raise ValueError(
                        "report artifact does not include all release parent artifacts"
                    )

    def artifact(self, artifact_type: ArtifactType) -> ReleaseArtifact:
        """Return the unique artifact for a required dashboard role."""
        matches = [
            artifact
            for artifact in self.artifacts
            if artifact.artifact_type is artifact_type
        ]
        if len(matches) != 1:
            raise ManifestValidationError(
                f"expected one {artifact_type.value!r} artifact; found {len(matches)}"
            )
        return matches[0]


class ArtifactInput(BaseModel):
    """Source file and lineage used to construct a release bundle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_id: Identifier
    artifact_type: ArtifactType
    source_path: Path
    schema_version: int = Field(default=1, ge=1, le=1)
    data_status: DataStatus
    source_run_id: Identifier
    parent_artifact_ids: list[Identifier] = Field(default_factory=list)
    parent_run_ids: list[Identifier] = Field(default_factory=list)
    dataset_id: Identifier | None = None
    model_id: str | None = None
    label: str = Field(min_length=1)


class ReleaseBuildConfig(BaseModel):
    """Versioned YAML input for deterministic release-bundle construction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    project_version: str
    label: str = Field(min_length=1)
    data_status: DataStatus
    datasets: list[DatasetReference] = Field(min_length=1)
    models: list[ModelReference] = Field(min_length=1)
    runs: RunLineage
    parent_artifact_ids: list[Identifier]
    artifacts: list[ArtifactInput] = Field(min_length=1)
    resolved_configuration_paths: dict[Identifier, Path]
    citations: list[str] = Field(min_length=1)
    known_limitations: list[str] = Field(min_length=1)

    @classmethod
    def from_yaml(cls, path: Path) -> ReleaseBuildConfig:
        """Load a strict release configuration from YAML."""
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"{path} must contain a YAML object")
        return cls.model_validate(loaded)


def _manifest_identity(
    config: ReleaseBuildConfig,
    *,
    artifact_hashes: dict[str, str],
    configuration_hashes: dict[str, str],
    source_implementation_sha256: str,
    git_commit: str,
) -> dict[str, Any]:
    artifacts = []
    for artifact in config.artifacts:
        record = artifact.model_dump(mode="json")
        record.pop("source_path")
        record["sha256"] = artifact_hashes[artifact.artifact_id]
        artifacts.append(record)
    return {
        "manifest_schema_version": 1,
        "project_version": config.project_version,
        "label": config.label,
        "data_status": config.data_status,
        "datasets": [value.model_dump(mode="json") for value in config.datasets],
        "models": [value.model_dump(mode="json") for value in config.models],
        "runs": config.runs.model_dump(mode="json"),
        "parent_artifact_ids": config.parent_artifact_ids,
        "artifacts": artifacts,
        "resolved_configuration_hashes": configuration_hashes,
        "citations": config.citations,
        "known_limitations": config.known_limitations,
        "source_implementation_sha256": source_implementation_sha256,
        "git_commit": git_commit,
    }


def build_release_bundle(
    *,
    config: ReleaseBuildConfig,
    output_root: Path,
    project_root: Path,
) -> tuple[Path, ReleaseManifest]:
    """Copy a coherent artifact family into an immutable release bundle."""
    if config.project_version != __version__:
        raise ManifestValidationError(
            f"release config version {config.project_version!r} does not match "
            f"package version {__version__!r}"
        )
    artifact_hashes: dict[str, str] = {}
    for artifact in config.artifacts:
        source = (project_root / artifact.source_path).resolve()
        if not source.is_file():
            raise ManifestValidationError(
                f"release source artifact is missing: {artifact.source_path}"
            )
        artifact_hashes[artifact.artifact_id] = sha256_file(source)
    configuration_hashes: dict[str, str] = {}
    for name, relative_path in config.resolved_configuration_paths.items():
        source = (project_root / relative_path).resolve()
        if not source.is_file():
            raise ManifestValidationError(
                f"resolved configuration is missing: {relative_path}"
            )
        configuration_hashes[name] = sha256_file(source)

    environment = collect_runtime_provenance(project_root)
    code = environment["code"]
    git_commit = str(code["git_revision"] or "uncommitted")
    implementation_hash = str(code["code_tree_sha256"])
    identity = _manifest_identity(
        config,
        artifact_hashes=artifact_hashes,
        configuration_hashes=configuration_hashes,
        source_implementation_sha256=implementation_hash,
        git_commit=git_commit,
    )
    digest = hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
    release_id = f"release-{digest[:12]}"
    bundle_root = output_root / release_id
    manifest_path = bundle_root / "release-manifest.json"
    if manifest_path.is_file():
        manifest = load_release_manifest(manifest_path)
        return manifest_path, manifest
    if bundle_root.exists():
        raise ManifestValidationError(
            f"incomplete release directory already exists: {bundle_root}"
        )

    temporary = output_root / f".{release_id}.tmp"
    if temporary.exists():
        raise ManifestValidationError(
            f"temporary release directory already exists: {temporary}"
        )
    artifacts_dir = temporary / "artifacts"
    artifacts_dir.mkdir(parents=True)
    release_artifacts: list[ReleaseArtifact] = []
    try:
        for artifact in config.artifacts:
            source = (project_root / artifact.source_path).resolve()
            destination_name = f"{artifact.artifact_id}{source.suffix.lower()}"
            destination = artifacts_dir / destination_name
            shutil.copy2(source, destination)
            release_artifacts.append(
                ReleaseArtifact(
                    **artifact.model_dump(exclude={"source_path"}),
                    path=f"artifacts/{destination_name}",
                    sha256=artifact_hashes[artifact.artifact_id],
                )
            )
        manifest = ReleaseManifest(
            manifest_schema_version=1,
            project_version=config.project_version,
            release_id=release_id,
            label=config.label,
            created_at_utc=datetime.now(UTC),
            git_commit=git_commit,
            git_dirty=bool(code["git_dirty"]),
            source_implementation_sha256=implementation_hash,
            resolved_configuration_hashes=configuration_hashes,
            data_status=config.data_status,
            datasets=config.datasets,
            models=config.models,
            runs=config.runs,
            parent_artifact_ids=config.parent_artifact_ids,
            artifacts=release_artifacts,
            citations=config.citations,
            known_limitations=config.known_limitations,
        )
        (temporary / "release-manifest.json").write_bytes(
            canonical_json_bytes(manifest.model_dump(mode="json"))
        )
        output_root.mkdir(parents=True, exist_ok=True)
        temporary.replace(bundle_root)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    loaded = load_release_manifest(manifest_path)
    return manifest_path, loaded


def load_release_manifest(
    manifest_path: Path,
    *,
    verify_checksums: bool = True,
) -> ReleaseManifest:
    """Load a release manifest and safely verify every declared artifact."""
    path = manifest_path.resolve()
    if not path.is_file():
        raise ManifestValidationError(
            f"release manifest does not exist: {manifest_path}"
        )
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        manifest = ReleaseManifest.model_validate(loaded)
    except (json.JSONDecodeError, ValueError) as error:
        raise ManifestValidationError(
            f"invalid release manifest {manifest_path}: {error}"
        ) from error

    root = path.parent.resolve()
    for artifact in manifest.artifacts:
        candidate = (root / Path(artifact.path)).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise ManifestValidationError(
                f"artifact path escapes release root: {artifact.path}"
            ) from error
        if not candidate.is_file():
            raise ManifestValidationError(
                f"release artifact is missing: {artifact.path}"
            )
        if verify_checksums:
            actual = sha256_file(candidate)
            if actual != artifact.sha256:
                raise ManifestValidationError(
                    f"checksum mismatch for {artifact.artifact_id!r}: "
                    f"expected {artifact.sha256}, got {actual}"
                )
    return manifest


def artifact_paths(
    manifest_path: Path,
    manifest: ReleaseManifest,
) -> dict[ArtifactType, Path]:
    """Resolve unique dashboard artifact roles after manifest validation."""
    root = manifest_path.resolve().parent
    return {
        artifact_type: (root / manifest.artifact(artifact_type).path).resolve()
        for artifact_type in DASHBOARD_REQUIRED_ARTIFACT_TYPES
    }
