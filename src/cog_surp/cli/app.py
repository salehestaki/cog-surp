"""Top-level Cog-Surp command-line application."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated

import psutil
import typer
import yaml
from rich.console import Console
from rich.table import Table

from cog_surp import __version__
from cog_surp.analysis import (
    AnalysisConfig,
    analyze_model_condition_effects,
    compare_two_models,
    evaluate_held_out_models,
    fit_hierarchical_model,
)
from cog_surp.cli.release_commands import (
    dashboard_app,
    demo_app,
    register_report_commands,
)
from cog_surp.datasets import DERCoAdapter, ERPCoreN400Adapter
from cog_surp.domain.datasets import DatasetConfig
from cog_surp.eeg import (
    ClusterAnalysisConfig,
    DERCoPreprocessingConfig,
    ERPPreprocessingConfig,
    aggregate_erp_core_cohort,
    discover_erp_core_subject_runs,
    extract_derco_subject_article,
    preprocess_erp_core_subject,
    run_sensor_time_cluster_analysis,
)
from cog_surp.features import build_derco_feature_table
from cog_surp.lm import (
    BoundaryAwareStrategy,
    SubtokenSumStrategy,
    TransformersBackend,
    compare_probability_strategies,
)
from cog_surp.lm.pipeline import score_stimulus_artifact
from cog_surp.provenance.checksums import sha256_file
from cog_surp.provenance.manifests import (
    canonical_json_bytes,
    write_dataset_manifest,
)
from cog_surp.provenance.runtime import collect_runtime_provenance
from cog_surp.reporting import build_research_report
from cog_surp.stimuli import (
    ControlledGeneratorConfig,
    generate_controlled_stimuli,
    load_derco_stimuli,
    load_erp_core_stimuli,
    validate_llm_candidates,
)

app = typer.Typer(
    name="cog-surp",
    help="Reproducible surprisal-N400 benchmarking workbench.",
    no_args_is_help=True,
)
console = Console()
datasets_app = typer.Typer(help="Inspect and fetch supported EEG datasets.")
eeg_app = typer.Typer(help="Preprocess EEG and extract prespecified outcomes.")
stimuli_app = typer.Typer(help="Validate and materialize linguistic stimuli.")
lm_app = typer.Typer(help="Score observed text with autoregressive language models.")
features_app = typer.Typer(help="Join EEG, stimulus, and computational features.")
analyze_app = typer.Typer(help="Run hierarchical and held-out analyses.")
report_app = typer.Typer(help="Generate artifact-driven research reports.")
provenance_app = typer.Typer(help="Capture release-grade runtime provenance.")
app.add_typer(datasets_app, name="datasets")
app.add_typer(eeg_app, name="eeg")
app.add_typer(stimuli_app, name="stimuli")
app.add_typer(lm_app, name="lm")
app.add_typer(features_app, name="features")
app.add_typer(analyze_app, name="analyze")
app.add_typer(report_app, name="report")
app.add_typer(dashboard_app, name="app")
app.add_typer(demo_app, name="demo")
app.add_typer(provenance_app, name="provenance")
register_report_commands(report_app)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def root(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Show the installed Cog-Surp version and exit.",
            is_eager=True,
            callback=_version_callback,
        ),
    ] = False,
) -> None:
    """Reproducible surprisal-N400 benchmarking workbench."""


@dataclass(frozen=True, slots=True)
class Check:
    """One machine-readable environment check."""

    name: str
    status: str
    detail: str
    required: bool


def _gpu_check() -> Check:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return Check(
            "gpu",
            "optional-missing",
            "nvidia-smi not found; CPU workflows remain supported",
            False,
        )
    result = subprocess.run(
        [
            executable,
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        return Check("gpu", "warning", result.stderr.strip() or "query failed", False)
    return Check("gpu", "ok", result.stdout.strip(), False)


def collect_doctor_checks(project_root: Path | None = None) -> list[Check]:
    """Collect deterministic capability checks without mutating the host."""
    root = (project_root or Path.cwd()).resolve()
    python_ok = sys.version_info[:2] == (3, 12)
    memory_gib = psutil.virtual_memory().total / 1024**3
    disk_gib = shutil.disk_usage(root).free / 1024**3
    if disk_gib >= 10:
        disk_status = "ok"
        disk_detail = f"{disk_gib:.2f} GiB free at {root}"
    elif disk_gib >= 2:
        disk_status = "warning"
        disk_detail = (
            f"{disk_gib:.2f} GiB free at {root}; enough for installation and "
            "CPU fixtures, but real-data/model runs should reserve at least 10 GiB"
        )
    else:
        disk_status = "fail"
        disk_detail = (
            f"{disk_gib:.2f} GiB free at {root}; at least 2 GiB is required "
            "for installation and CPU fixtures"
        )
    required_paths = ("pyproject.toml", "configs", "artifacts")
    missing = [path for path in required_paths if not (root / path).exists()]

    return [
        Check("cog-surp", "ok", __version__, True),
        Check(
            "python",
            "ok" if python_ok else "fail",
            platform.python_version(),
            True,
        ),
        Check("operating-system", "ok", platform.platform(), True),
        Check(
            "memory",
            "ok" if memory_gib >= 8 else "warning",
            f"{memory_gib:.2f} GiB total",
            False,
        ),
        Check(
            "disk",
            disk_status,
            disk_detail,
            True,
        ),
        Check(
            "project-layout",
            "ok" if not missing else "fail",
            "required paths present"
            if not missing
            else f"missing: {', '.join(missing)}",
            True,
        ),
        _gpu_check(),
    ]


@datasets_app.command("list")
def datasets_list() -> None:
    """List supported real EEG datasets and their access status."""
    table = Table(title="Cog-Surp datasets")
    table.add_column("ID")
    table.add_column("Status")
    table.add_column("License")
    table.add_column("Primary role")
    table.add_row(
        "erp-core-n400",
        "public",
        "CC-BY-SA-4.0",
        "primary controlled N400 vertical slice",
    )
    table.add_row(
        "derco",
        "public; bounded fetch",
        "NOASSERTION",
        "word-aligned H3 candidate; do not redistribute",
    )
    console.print(table)


@datasets_app.command("fetch")
def datasets_fetch(
    dataset_id: Annotated[str, typer.Argument(help="Dataset adapter ID.")],
    subject: Annotated[
        list[str] | None,
        typer.Option("--subject", help="Subject number; repeat for multiple."),
    ] = None,
    run: Annotated[
        list[str] | None,
        typer.Option("--run", help="Dataset run/article ID; repeat for multiple."),
    ] = None,
    metadata_only: Annotated[
        bool,
        typer.Option(help="Skip EEG signal files while validating metadata access."),
    ] = False,
    data_root: Annotated[
        Path,
        typer.Option(help="Root for immutable source data."),
    ] = Path("data/raw"),
    run_id: Annotated[
        str | None,
        typer.Option(help="Reuse an explicit run ID."),
    ] = None,
) -> None:
    """Fetch ERP CORE files, validate checksums, and write a manifest."""
    if dataset_id not in {"erp-core-n400", "derco"}:
        raise typer.BadParameter(
            f"unsupported dataset {dataset_id!r}; run `cog-surp datasets list`"
        )
    normalized_subjects = tuple(
        sorted({value.removeprefix("sub-").zfill(3) for value in subject or []})
    )
    resolved = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "destination": str(data_root.resolve()),
        "subjects": normalized_subjects,
        "runs": tuple(sorted(set(run or []))),
        "metadata_only": metadata_only,
        "implementation_sha256": {
            path.name: sha256_file(path)
            for path in (
                Path(__file__).parents[1]
                / "datasets"
                / f"{'erp_core' if dataset_id == 'erp-core-n400' else 'derco'}.py",
                Path(__file__).parents[1] / "datasets" / "osf.py",
            )
        },
    }
    config_hash = hashlib.sha256(canonical_json_bytes(resolved)).hexdigest()
    effective_run_id = run_id or f"dataset-{config_hash[:12]}"
    run_root = Path("artifacts") / "runs" / effective_run_id
    run_root.mkdir(parents=True, exist_ok=True)
    resolved_path = run_root / "resolved-config.json"
    if resolved_path.exists() and resolved_path.read_bytes() != canonical_json_bytes(
        resolved
    ):
        raise typer.BadParameter(
            f"run ID {effective_run_id!r} already has a different immutable config"
        )
    manifest_path = run_root / "dataset-manifest.json"
    if resolved_path.exists() and _valid_dataset_manifest(
        manifest_path,
        data_root / dataset_id,
    ):
        typer.echo(
            json.dumps(
                {
                    "event": "dataset_fetch_reused",
                    "run_id": effective_run_id,
                    "manifest": str(manifest_path.resolve()),
                },
                sort_keys=True,
            )
        )
        return
    resolved_path.write_bytes(canonical_json_bytes(resolved))
    typer.echo(
        json.dumps(
            {
                "event": "dataset_fetch_started",
                "run_id": effective_run_id,
                "dataset_id": dataset_id,
            },
            sort_keys=True,
        ),
        err=True,
    )
    config = DatasetConfig(
        dataset_id=dataset_id,
        destination=data_root,
        subjects=normalized_subjects,
        runs=tuple(sorted(set(run or []))),
        metadata_only=metadata_only,
    )
    adapter = ERPCoreN400Adapter() if dataset_id == "erp-core-n400" else DERCoAdapter()
    manifest = adapter.fetch(config)
    digest = write_dataset_manifest(manifest, manifest_path)
    typer.echo(
        json.dumps(
            {
                "event": "dataset_fetch_completed",
                "run_id": effective_run_id,
                "artifact_count": len(manifest.artifacts),
                "manifest_sha256": digest,
                "manifest": str(manifest_path.resolve()),
            },
            sort_keys=True,
        )
    )


def _find_dataset_manifest(dataset_id: str, signal_suffix: str) -> Path:
    candidates: list[Path] = []
    for path in Path("artifacts/runs").glob("*/dataset-manifest.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("dataset_id") != dataset_id:
            continue
        artifact_paths = {
            artifact["relative_path"] for artifact in payload.get("artifacts", [])
        }
        if any(value.endswith(signal_suffix) for value in artifact_paths):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            "no checksummed signal manifest found; run "
            f"`cog-surp datasets fetch {dataset_id}` with the requested subject/run"
        )
    return sorted(candidates)[-1]


def _valid_artifact_manifest(run_root: Path, manifest_path: Path) -> bool:
    """Return whether every file in an existing artifact manifest verifies."""
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = manifest.get("artifacts")
        if records is None and isinstance(manifest.get("artifact"), dict):
            records = [manifest["artifact"]]
        if not isinstance(records, list):
            return False
        return bool(records) and all(
            (run_root / record["path"]).is_file()
            and sha256_file(run_root / record["path"]) == record["sha256"]
            for record in records
        )
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return False


def _valid_dataset_manifest(manifest_path: Path, dataset_root: Path) -> bool:
    """Return whether every downloaded file in a dataset manifest verifies."""
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = manifest["artifacts"]
        return bool(records) and all(
            (dataset_root / record["relative_path"]).is_file()
            and (dataset_root / record["relative_path"]).stat().st_size
            == record["size_bytes"]
            and sha256_file(dataset_root / record["relative_path"]) == record["sha256"]
            for record in records
        )
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return False


@eeg_app.command("preprocess")
def eeg_preprocess(
    config_path: Annotated[
        Path,
        typer.Option("--config", help="Versioned EEG YAML configuration."),
    ],
    subject: Annotated[
        str,
        typer.Option(help="Dataset participant ID."),
    ] = "001",
    article: Annotated[
        str,
        typer.Option(help="DERCo article ID when dataset_id is derco."),
    ] = "article_0",
    dataset_root: Annotated[
        Path | None,
        typer.Option(help="Fetched dataset root."),
    ] = None,
    dataset_manifest: Annotated[
        Path | None,
        typer.Option(help="Checksummed upstream dataset manifest."),
    ] = None,
    run_id: Annotated[
        str | None,
        typer.Option(help="Reuse an explicit immutable run ID."),
    ] = None,
) -> None:
    """Preprocess real EEG and write a single-trial N400 Parquet artifact."""
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise typer.BadParameter("EEG config must be a YAML mapping")
    dataset_id = str(loaded.get("dataset_id"))
    config: ERPPreprocessingConfig | DERCoPreprocessingConfig
    if dataset_id == "erp-core-n400":
        normalized = subject.removeprefix("sub-").zfill(3)
        config = ERPPreprocessingConfig.model_validate(loaded)
        resolved_dataset_root = dataset_root or Path("data/raw/erp-core-n400")
        signal_suffix = f"sub-{normalized}_task-N400_eeg.fdt"
    elif dataset_id == "derco":
        normalized = subject
        config = DERCoPreprocessingConfig.model_validate(loaded)
        resolved_dataset_root = dataset_root or Path("data/raw/derco")
        signal_suffix = (
            f"EEG_data/preprocessed/{normalized}/{article}/preprocessed_epoch.fif"
        )
    else:
        raise typer.BadParameter(f"unsupported EEG dataset_id: {dataset_id!r}")
    upstream = dataset_manifest or _find_dataset_manifest(dataset_id, signal_suffix)
    upstream_hash = sha256_file(upstream)
    implementation_name = (
        "preprocessing.py" if dataset_id == "erp-core-n400" else "derco.py"
    )
    implementation_path = Path(__file__).parents[1] / "eeg" / implementation_name
    resolved = {
        "schema_version": 1,
        "command": "eeg preprocess",
        "dataset_id": dataset_id,
        "subject": normalized,
        "article": article if dataset_id == "derco" else None,
        "dataset_root": str(resolved_dataset_root.resolve()),
        "dataset_manifest": str(upstream.resolve()),
        "dataset_manifest_sha256": upstream_hash,
        "configuration": config.model_dump(mode="json"),
        "implementation_sha256": sha256_file(implementation_path),
    }
    config_hash = hashlib.sha256(canonical_json_bytes(resolved)).hexdigest()
    effective_run_id = run_id or f"eeg-{config_hash[:12]}"
    run_root = Path("artifacts") / "runs" / effective_run_id
    run_root.mkdir(parents=True, exist_ok=True)
    resolved_path = run_root / "resolved-config.json"
    payload = canonical_json_bytes(resolved)
    if resolved_path.exists() and resolved_path.read_bytes() != payload:
        raise typer.BadParameter(
            f"run ID {effective_run_id!r} already has a different immutable config"
        )
    artifact_manifest_path = run_root / "artifact-manifest.json"
    if resolved_path.exists() and _valid_artifact_manifest(
        run_root, artifact_manifest_path
    ):
        typer.echo(
            json.dumps(
                {
                    "event": "eeg_preprocess_reused",
                    "run_id": effective_run_id,
                    "artifact_manifest": str(artifact_manifest_path.resolve()),
                },
                sort_keys=True,
            )
        )
        return
    resolved_path.write_bytes(payload)
    typer.echo(
        json.dumps(
            {
                "event": "eeg_preprocess_started",
                "run_id": effective_run_id,
                "subject": normalized,
                "data_status": "real",
            },
            sort_keys=True,
        ),
        err=True,
    )
    output_dir = run_root / "eeg" / normalized
    artifact_paths: tuple[Path, ...]
    if isinstance(config, ERPPreprocessingConfig):
        erp_artifacts = preprocess_erp_core_subject(
            dataset_root=resolved_dataset_root,
            subject=normalized,
            config=config,
            output_dir=output_dir,
            run_id=effective_run_id,
        )
        artifact_paths = (
            erp_artifacts.single_trials,
            erp_artifacts.summary,
            erp_artifacts.figure,
            erp_artifacts.evokeds,
            erp_artifacts.single_trials.parent / "preprocessing-summary.json",
        )
        accepted_trials = erp_artifacts.accepted_trials
        rejected_trials = erp_artifacts.rejected_trials
    else:
        derco_artifacts = extract_derco_subject_article(
            dataset_root=resolved_dataset_root,
            subject=normalized,
            article=article,
            config=config,
            output_dir=output_dir / article,
            run_id=effective_run_id,
        )
        artifact_paths = (
            derco_artifacts.single_trials,
            derco_artifacts.summary,
        )
        accepted_trials = derco_artifacts.accepted_trials
        rejected_trials = derco_artifacts.rejected_trials
    artifact_records = [
        {
            "path": str(path.relative_to(run_root)).replace("\\", "/"),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in artifact_paths
    ]
    artifact_manifest = {
        "schema_version": 1,
        "run_id": effective_run_id,
        "data_status": "real",
        "analysis_status": config.analysis_status,
        "parents": [{"path": str(upstream.resolve()), "sha256": upstream_hash}],
        "artifacts": artifact_records,
    }
    artifact_manifest_path.write_bytes(canonical_json_bytes(artifact_manifest))
    typer.echo(
        json.dumps(
            {
                "event": "eeg_preprocess_completed",
                "run_id": effective_run_id,
                "accepted_trials": accepted_trials,
                "rejected_trials": rejected_trials,
                "artifact_manifest": str(artifact_manifest_path.resolve()),
            },
            sort_keys=True,
        )
    )


@eeg_app.command("summarize-cohort")
def eeg_summarize_cohort(
    config_path: Annotated[
        Path,
        typer.Option("--config", help="Primary ERP CORE preprocessing YAML."),
    ],
    dataset_manifest: Annotated[
        Path,
        typer.Option(help="Checksummed upstream ERP CORE dataset manifest."),
    ],
    artifacts_root: Annotated[
        Path,
        typer.Option(help="Directory containing subject preprocessing runs."),
    ] = Path("artifacts/runs"),
    run_id: Annotated[
        str | None,
        typer.Option(help="Reuse an explicit immutable cohort run ID."),
    ] = None,
) -> None:
    """Aggregate verified ERP CORE subjects and estimate the primary H1 contrast."""
    config = ERPPreprocessingConfig.from_yaml(config_path)
    if config.analysis_status != "primary":
        raise typer.BadParameter("cohort H1 requires a primary preprocessing config")
    subjects = discover_erp_core_subject_runs(
        artifacts_root=artifacts_root,
        config=config,
        dataset_manifest=dataset_manifest,
    )
    expected = {f"sub-{value:03d}" for value in range(1, 41)} - {"sub-027"}
    observed = {subject.participant for subject in subjects}
    if observed != expected:
        raise typer.BadParameter(
            "expected the 39 public ERP CORE participants "
            f"(001-040 excluding 027); missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )
    parent_records = [
        {
            "run_id": subject.run_id,
            "path": str(subject.manifest.resolve()),
            "sha256": sha256_file(subject.manifest),
        }
        for subject in subjects
    ]
    dataset_hash = sha256_file(dataset_manifest)
    implementation_path = Path(__file__).parents[1] / "eeg" / "cohort.py"
    resolved = {
        "schema_version": 1,
        "command": "eeg summarize-cohort",
        "dataset_id": "erp-core-n400",
        "dataset_manifest": str(dataset_manifest.resolve()),
        "dataset_manifest_sha256": dataset_hash,
        "configuration": config.model_dump(mode="json"),
        "implementation_sha256": sha256_file(implementation_path),
        "subject_runs": parent_records,
    }
    payload = canonical_json_bytes(resolved)
    config_hash = hashlib.sha256(payload).hexdigest()
    effective_run_id = run_id or f"eeg-cohort-{config_hash[:12]}"
    run_root = artifacts_root / effective_run_id
    run_root.mkdir(parents=True, exist_ok=True)
    resolved_path = run_root / "resolved-config.json"
    if resolved_path.exists() and resolved_path.read_bytes() != payload:
        raise typer.BadParameter(
            f"run ID {effective_run_id!r} already has a different immutable config"
        )
    manifest_path = run_root / "artifact-manifest.json"
    if resolved_path.exists() and _valid_artifact_manifest(run_root, manifest_path):
        typer.echo(
            json.dumps(
                {
                    "event": "eeg_cohort_reused",
                    "run_id": effective_run_id,
                    "artifact_manifest": str(manifest_path.resolve()),
                },
                sort_keys=True,
            )
        )
        return
    resolved_path.write_bytes(payload)
    cohort_artifacts = aggregate_erp_core_cohort(
        subject_runs=subjects,
        config=config,
        output_dir=run_root / "eeg" / "cohort",
        run_id=effective_run_id,
    )
    artifact_paths = tuple(asdict(cohort_artifacts).values())
    artifact_manifest = {
        "schema_version": 1,
        "run_id": effective_run_id,
        "data_status": "real",
        "analysis_status": "primary",
        "parents": [
            {
                "path": str(dataset_manifest.resolve()),
                "sha256": dataset_hash,
            },
            *parent_records,
        ],
        "artifacts": [
            {
                "path": str(path.relative_to(run_root)).replace("\\", "/"),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in artifact_paths
        ],
    }
    manifest_path.write_bytes(canonical_json_bytes(artifact_manifest))
    result = json.loads(cohort_artifacts.h1_estimate.read_text(encoding="utf-8"))
    typer.echo(
        json.dumps(
            {
                "event": "eeg_cohort_completed",
                "run_id": effective_run_id,
                "participants": result["participant_counts"],
                "primary_effect": result["primary_rule_based_cohort"],
                "artifact_manifest": str(manifest_path.resolve()),
            },
            sort_keys=True,
        )
    )


@eeg_app.command("cluster-exploratory")
def eeg_cluster_exploratory(
    config_path: Annotated[
        Path,
        typer.Option("--config", help="Versioned exploratory cluster YAML."),
    ],
    cohort_run: Annotated[
        Path,
        typer.Option(help="Completed ERP CORE cohort run directory."),
    ],
    run_id: Annotated[
        str | None,
        typer.Option(help="Reuse an explicit immutable run ID."),
    ] = None,
) -> None:
    """Run an exploratory real-data sensor-time cluster permutation test."""
    import pandas as pd

    config = ClusterAnalysisConfig.from_yaml(config_path)
    if config.data_status != "real":
        raise typer.BadParameter("cohort cluster command requires data_status: real")
    qc_path = cohort_run / "eeg" / "cohort" / "participant-qc.parquet"
    cohort_manifest = cohort_run / "artifact-manifest.json"
    if not qc_path.is_file() or not cohort_manifest.is_file():
        raise typer.BadParameter("cohort run lacks QC or artifact manifest")
    qc = pd.read_parquet(qc_path)
    included = qc.loc[qc["participant_included"].astype(bool)]
    evoked_paths = [
        Path("artifacts")
        / "runs"
        / str(row.preprocessing_run_id)
        / "eeg"
        / str(row.participant).removeprefix("sub-")
        / "condition-evokeds-ave.fif"
        for row in included.itertuples(index=False)
    ]
    missing = [path for path in evoked_paths if not path.is_file()]
    if missing:
        raise typer.BadParameter(f"missing participant evokeds: {missing}")
    parent_paths = [cohort_manifest, *evoked_paths]
    parents = [
        {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in parent_paths
    ]
    implementation = Path(__file__).parents[1] / "eeg" / "cluster.py"
    resolved = {
        "schema_version": 1,
        "command": "eeg cluster-exploratory",
        "configuration": config.model_dump(mode="json"),
        "parents": parents,
        "implementation_sha256": sha256_file(implementation),
    }
    payload = canonical_json_bytes(resolved)
    config_hash = hashlib.sha256(payload).hexdigest()
    effective_run_id = run_id or f"cluster-{config_hash[:12]}"
    run_root = Path("artifacts") / "runs" / effective_run_id
    run_root.mkdir(parents=True, exist_ok=True)
    resolved_path = run_root / "resolved-config.json"
    if resolved_path.exists() and resolved_path.read_bytes() != payload:
        raise typer.BadParameter(
            f"run ID {effective_run_id!r} already has a different immutable config"
        )
    manifest_path = run_root / "cluster-manifest.json"
    if resolved_path.exists() and _valid_artifact_manifest(run_root, manifest_path):
        typer.echo(
            json.dumps(
                {
                    "event": "cluster_analysis_reused",
                    "run_id": effective_run_id,
                    "artifact_manifest": str(manifest_path.resolve()),
                },
                sort_keys=True,
            )
        )
        return
    resolved_path.write_bytes(payload)
    artifacts = run_sensor_time_cluster_analysis(
        evoked_paths=evoked_paths,
        config=config,
        output_dir=run_root,
        run_id=effective_run_id,
    )
    artifact_paths = tuple(asdict(artifacts).values())
    manifest = {
        "schema_version": 1,
        "run_id": effective_run_id,
        "analysis_status": "exploratory",
        "data_status": "real",
        "parents": parents,
        "artifacts": [
            {
                "path": path.name,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in artifact_paths
        ],
    }
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    metadata = json.loads(artifacts.metadata.read_text(encoding="utf-8"))
    typer.echo(
        json.dumps(
            {
                "event": "cluster_analysis_completed",
                "run_id": effective_run_id,
                "participants": metadata["participants"],
                "clusters": metadata["clusters"],
                "clusters_passing_alpha": metadata["clusters_passing_alpha"],
                "artifact_manifest": str(manifest_path.resolve()),
            },
            sort_keys=True,
        )
    )


@stimuli_app.command("validate")
def stimuli_validate(
    config_path: Annotated[
        Path,
        typer.Option("--config", help="Versioned dataset YAML configuration."),
    ],
    dataset_root: Annotated[
        Path | None,
        typer.Option(help="Fetched dataset root."),
    ] = None,
    run_id: Annotated[
        str | None,
        typer.Option(help="Reuse an explicit immutable run ID."),
    ] = None,
) -> None:
    """Validate publisher stimuli and write a tidy Parquet artifact."""
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise typer.BadParameter("dataset config must be a YAML mapping")
    dataset_id = str(loaded.get("dataset_id"))
    if dataset_id == "erp-core-n400":
        resolved_dataset_root = dataset_root or Path("data/raw/erp-core-n400")
        source_paths = sorted(
            (resolved_dataset_root / "stimuli").glob("N400_stimuli_list*_English.txt")
        )
        frame = load_erp_core_stimuli(resolved_dataset_root)
        validation_status = "publisher-validated"
        limitations = [
            "The public BIDS events identify condition codes but not presented words.",
            "Stimulus rows cannot be joined to individual ERP CORE trials or items.",
        ]
    elif dataset_id == "derco":
        resolved_dataset_root = dataset_root or Path("data/raw/derco")
        source_paths = sorted(
            (resolved_dataset_root / "prediction").glob(
                "human_prediction_article_*.csv"
            )
        )
        frame = load_derco_stimuli(resolved_dataset_root)
        validation_status = "publisher-native-human-prediction"
        limitations = [
            "The public OSF dataset does not declare a dataset license.",
            (
                "Raw exact-match cloze is retained for audit; the publisher-corrected "
                "p_cloze attached to EEG epochs is canonical for analysis."
            ),
        ]
    else:
        raise typer.BadParameter(f"unsupported stimulus dataset_id: {dataset_id!r}")
    source_hashes = {
        str(path.relative_to(resolved_dataset_root)).replace("\\", "/"): sha256_file(
            path
        )
        for path in source_paths
    }
    resolved = {
        "schema_version": 1,
        "command": "stimuli validate",
        "dataset_id": dataset_id,
        "dataset_root": str(resolved_dataset_root.resolve()),
        "source_sha256": source_hashes,
        "implementation_sha256": sha256_file(
            Path(__file__).parents[1]
            / "stimuli"
            / f"{'erp_core' if dataset_id == 'erp-core-n400' else 'derco'}.py"
        ),
    }
    config_hash = hashlib.sha256(canonical_json_bytes(resolved)).hexdigest()
    effective_run_id = run_id or f"stimuli-{config_hash[:12]}"
    run_root = Path("artifacts") / "runs" / effective_run_id
    run_root.mkdir(parents=True, exist_ok=True)
    resolved_path = run_root / "resolved-config.json"
    payload = canonical_json_bytes(resolved)
    if resolved_path.exists() and resolved_path.read_bytes() != payload:
        raise typer.BadParameter(
            f"run ID {effective_run_id!r} already has a different immutable config"
        )
    manifest_path = run_root / "stimulus-manifest.json"
    if resolved_path.exists() and _valid_artifact_manifest(run_root, manifest_path):
        typer.echo(
            json.dumps(
                {
                    "event": "stimuli_validation_reused",
                    "run_id": effective_run_id,
                    "artifact_manifest": str(manifest_path.resolve()),
                },
                sort_keys=True,
            )
        )
        return
    resolved_path.write_bytes(payload)
    artifact = run_root / "stimuli.parquet"
    frame.to_parquet(artifact, index=False)
    manifest = {
        "schema_version": 1,
        "run_id": effective_run_id,
        "dataset_id": dataset_id,
        "validation_status": validation_status,
        "record_count": len(frame),
        "unique_items": int(frame["item"].nunique()),
        "unique_targets": int(frame["target_word"].nunique()),
        "artifact": {
            "path": artifact.name,
            "size_bytes": artifact.stat().st_size,
            "sha256": sha256_file(artifact),
        },
        "limitations": limitations,
    }
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    typer.echo(
        json.dumps(
            {
                "event": "stimuli_validation_completed",
                "run_id": effective_run_id,
                "records": len(frame),
                "unique_targets": int(frame["target_word"].nunique()),
                "artifact": str(artifact.resolve()),
            },
            sort_keys=True,
        )
    )


@stimuli_app.command("generate-controlled")
def stimuli_generate_controlled(
    config_path: Annotated[
        Path,
        typer.Option("--config", help="Versioned controlled-generator YAML."),
    ],
    run_id: Annotated[
        str | None,
        typer.Option(help="Reuse an explicit immutable run ID."),
    ] = None,
) -> None:
    """Generate deterministic paired model-side stress-test stimuli."""
    config = ControlledGeneratorConfig.from_yaml(config_path)
    implementation = Path(__file__).parents[1] / "stimuli" / "controlled.py"
    resolved = {
        "schema_version": 1,
        "command": "stimuli generate-controlled",
        "configuration": config.model_dump(mode="json"),
        "implementation_sha256": sha256_file(implementation),
    }
    payload = canonical_json_bytes(resolved)
    config_hash = hashlib.sha256(payload).hexdigest()
    effective_run_id = run_id or f"controlled-{config_hash[:12]}"
    run_root = Path("artifacts") / "runs" / effective_run_id
    run_root.mkdir(parents=True, exist_ok=True)
    resolved_path = run_root / "resolved-config.json"
    if resolved_path.exists() and resolved_path.read_bytes() != payload:
        raise typer.BadParameter(
            f"run ID {effective_run_id!r} already has a different immutable config"
        )
    manifest_path = run_root / "stimulus-manifest.json"
    if resolved_path.exists() and _valid_artifact_manifest(run_root, manifest_path):
        typer.echo(
            json.dumps(
                {
                    "event": "controlled_stimuli_reused",
                    "run_id": effective_run_id,
                    "artifact_manifest": str(manifest_path.resolve()),
                },
                sort_keys=True,
            )
        )
        return
    resolved_path.write_bytes(payload)
    frame = generate_controlled_stimuli(config)
    artifact = run_root / "stimuli.parquet"
    frame.to_parquet(artifact, index=False)
    manifest = {
        "schema_version": 1,
        "run_id": effective_run_id,
        "data_status": "synthetic-stimulus",
        "scientific_use": "model-side-stress-test-only",
        "artifact": {
            "path": artifact.name,
            "sha256": sha256_file(artifact),
            "size_bytes": artifact.stat().st_size,
            "records": len(frame),
        },
    }
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    typer.echo(
        json.dumps(
            {
                "event": "controlled_stimuli_completed",
                "run_id": effective_run_id,
                "records": len(frame),
                "artifact": str(artifact.resolve()),
            },
            sort_keys=True,
        )
    )


@stimuli_app.command("validate-candidates")
def stimuli_validate_candidates(
    input_path: Annotated[
        Path,
        typer.Option("--input", help="Auditable LLM-assisted candidate JSONL."),
    ],
    run_id: Annotated[
        str | None,
        typer.Option(help="Reuse an explicit immutable run ID."),
    ] = None,
) -> None:
    """Validate LLM-assisted candidates as non-validated model stress tests."""
    implementation = Path(__file__).parents[1] / "stimuli" / "controlled.py"
    resolved = {
        "schema_version": 1,
        "command": "stimuli validate-candidates",
        "input_path": str(input_path.resolve()),
        "input_sha256": sha256_file(input_path),
        "implementation_sha256": sha256_file(implementation),
    }
    payload = canonical_json_bytes(resolved)
    config_hash = hashlib.sha256(payload).hexdigest()
    effective_run_id = run_id or f"candidates-{config_hash[:12]}"
    run_root = Path("artifacts") / "runs" / effective_run_id
    run_root.mkdir(parents=True, exist_ok=True)
    resolved_path = run_root / "resolved-config.json"
    if resolved_path.exists() and resolved_path.read_bytes() != payload:
        raise typer.BadParameter(
            f"run ID {effective_run_id!r} already has a different immutable config"
        )
    manifest_path = run_root / "stimulus-manifest.json"
    if resolved_path.exists() and _valid_artifact_manifest(run_root, manifest_path):
        typer.echo(
            json.dumps(
                {
                    "event": "candidate_validation_reused",
                    "run_id": effective_run_id,
                    "artifact_manifest": str(manifest_path.resolve()),
                },
                sort_keys=True,
            )
        )
        return
    resolved_path.write_bytes(payload)
    frame = validate_llm_candidates(input_path)
    artifact = run_root / "stimuli.parquet"
    frame.to_parquet(artifact, index=False)
    manifest = {
        "schema_version": 1,
        "run_id": effective_run_id,
        "data_status": "synthetic-stimulus",
        "scientific_use": "model-side-stress-test-only",
        "parent": {
            "path": str(input_path.resolve()),
            "sha256": sha256_file(input_path),
        },
        "artifact": {
            "path": artifact.name,
            "sha256": sha256_file(artifact),
            "size_bytes": artifact.stat().st_size,
            "records": len(frame),
        },
    }
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    typer.echo(
        json.dumps(
            {
                "event": "candidate_validation_completed",
                "run_id": effective_run_id,
                "records": len(frame),
                "artifact": str(artifact.resolve()),
            },
            sort_keys=True,
        )
    )


def _latest_stimulus_artifact() -> Path:
    candidates = sorted(Path("artifacts/runs").glob("stimuli-*/stimuli.parquet"))
    if not candidates:
        raise FileNotFoundError(
            "no validated stimuli found; run `cog-surp stimuli validate "
            "--config configs/datasets/erp_core.yaml`"
        )
    return candidates[-1]


def _primary_derco_eeg_artifacts() -> list[Path]:
    paths: list[Path] = []
    for config_path in Path("artifacts/runs").glob("eeg-*/resolved-config.json"):
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
        configuration = loaded.get("configuration", {})
        if (
            loaded.get("dataset_id") == "derco"
            and configuration.get("analysis_status") == "primary"
        ):
            candidate = (
                config_path.parent
                / "eeg"
                / str(loaded["subject"])
                / str(loaded["article"])
                / "single-trial-n400.parquet"
            )
            if candidate.exists():
                paths.append(candidate)
    if not paths:
        raise FileNotFoundError("no completed primary DERCo EEG artifacts found")
    return sorted(paths)


@features_app.command("build")
def features_build(
    stimuli_path: Annotated[
        Path,
        typer.Option(help="Validated stimulus Parquet artifact."),
    ],
    surprisal_path: Annotated[
        Path,
        typer.Option(help="Auditable model-surprisal Parquet artifact."),
    ],
    run_id: Annotated[
        str | None,
        typer.Option(help="Reuse an explicit immutable run ID."),
    ] = None,
) -> None:
    """Join primary DERCo EEG to word-aligned stimulus and model features."""
    eeg_paths = _primary_derco_eeg_artifacts()
    inputs = [stimuli_path, surprisal_path, *eeg_paths]
    input_records = [
        {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in inputs
    ]
    resolved = {
        "schema_version": 1,
        "command": "features build",
        "dataset_id": "derco",
        "input_artifacts": input_records,
        "alignment_key": "publisher WordID",
        "implementation_sha256": sha256_file(
            Path(__file__).parents[1] / "features" / "alignment.py"
        ),
    }
    config_hash = hashlib.sha256(canonical_json_bytes(resolved)).hexdigest()
    effective_run_id = run_id or f"features-{config_hash[:12]}"
    run_root = Path("artifacts") / "runs" / effective_run_id
    run_root.mkdir(parents=True, exist_ok=True)
    resolved_path = run_root / "resolved-config.json"
    payload = canonical_json_bytes(resolved)
    if resolved_path.exists() and resolved_path.read_bytes() != payload:
        raise typer.BadParameter(
            f"run ID {effective_run_id!r} already has a different immutable config"
        )
    manifest_path = run_root / "feature-manifest.json"
    if resolved_path.exists() and _valid_artifact_manifest(run_root, manifest_path):
        typer.echo(
            json.dumps(
                {
                    "event": "features_build_reused",
                    "run_id": effective_run_id,
                    "artifact_manifest": str(manifest_path.resolve()),
                },
                sort_keys=True,
            )
        )
        return
    resolved_path.write_bytes(payload)
    frame = build_derco_feature_table(
        eeg_paths=eeg_paths,
        stimuli_path=stimuli_path,
        surprisal_path=surprisal_path,
    )
    artifact = run_root / "features.parquet"
    frame.to_parquet(artifact, index=False)
    manifest = {
        "schema_version": 1,
        "run_id": effective_run_id,
        "dataset_id": "derco",
        "data_status": "real",
        "alignment_status": "authoritative-publisher-word-id",
        "participant_count": int(frame["participant"].nunique()),
        "item_count": int(frame["item"].nunique()),
        "record_count": len(frame),
        "parents": input_records,
        "artifact": {
            "path": artifact.name,
            "sha256": sha256_file(artifact),
            "size_bytes": artifact.stat().st_size,
        },
    }
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    typer.echo(
        json.dumps(
            {
                "event": "features_build_completed",
                "run_id": effective_run_id,
                "records": len(frame),
                "participants": int(frame["participant"].nunique()),
                "items": int(frame["item"].nunique()),
                "artifact": str(artifact.resolve()),
            },
            sort_keys=True,
        )
    )


@analyze_app.command("model-effect")
def analyze_model_effect(
    surprisal_path: Annotated[
        list[Path],
        typer.Option(
            "--surprisal-path",
            help="ERP CORE exact-surprisal artifact; repeat for each model.",
        ),
    ],
    run_id: Annotated[
        str | None,
        typer.Option(help="Reuse an explicit immutable run ID."),
    ] = None,
) -> None:
    """Estimate H2 matched-target condition effects for multiple models."""
    if len(surprisal_path) < 2:
        raise typer.BadParameter("provide at least two --surprisal-path values")
    parents = [
        {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in surprisal_path
    ]
    implementation = Path(__file__).parents[1] / "analysis" / "model_effect.py"
    resolved = {
        "schema_version": 1,
        "command": "analyze model-effect",
        "hypothesis": "H2",
        "estimand": "matched-target unrelated minus related surprisal",
        "parents": parents,
        "implementation_sha256": sha256_file(implementation),
    }
    payload = canonical_json_bytes(resolved)
    config_hash = hashlib.sha256(payload).hexdigest()
    effective_run_id = run_id or f"model-effect-{config_hash[:12]}"
    run_root = Path("artifacts") / "runs" / effective_run_id
    run_root.mkdir(parents=True, exist_ok=True)
    resolved_path = run_root / "resolved-config.json"
    if resolved_path.exists() and resolved_path.read_bytes() != payload:
        raise typer.BadParameter(
            f"run ID {effective_run_id!r} already has a different immutable config"
        )
    manifest_path = run_root / "analysis-manifest.json"
    if resolved_path.exists() and _valid_artifact_manifest(run_root, manifest_path):
        typer.echo(
            json.dumps(
                {
                    "event": "model_effect_reused",
                    "run_id": effective_run_id,
                    "artifact_manifest": str(manifest_path.resolve()),
                },
                sort_keys=True,
            )
        )
        return
    resolved_path.write_bytes(payload)
    artifacts = analyze_model_condition_effects(
        surprisal_paths=surprisal_path,
        output_dir=run_root,
        run_id=effective_run_id,
    )
    artifact_paths = tuple(asdict(artifacts).values())
    manifest = {
        "schema_version": 1,
        "run_id": effective_run_id,
        "hypothesis": "H2",
        "data_status": "real-stimulus-metadata",
        "parents": parents,
        "artifacts": [
            {
                "path": path.name,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in artifact_paths
        ],
    }
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    summary = json.loads(artifacts.summary_json.read_text(encoding="utf-8"))
    typer.echo(
        json.dumps(
            {
                "event": "model_effect_completed",
                "run_id": effective_run_id,
                "models": summary["models"],
                "artifact_manifest": str(manifest_path.resolve()),
            },
            sort_keys=True,
        )
    )


@analyze_app.command("causal-condition")
def analyze_causal_condition(
    h1_trials_path: Annotated[
        Path,
        typer.Option(help="ERP CORE accepted cohort single-trial artifact."),
    ],
    h2_surprisal_path: Annotated[
        list[Path],
        typer.Option(
            "--h2-surprisal-path",
            help="ERP CORE exact LM score artifact; repeat for each model.",
        ),
    ],
    run_id: Annotated[
        str | None,
        typer.Option(help="Reuse an explicit immutable run ID."),
    ] = None,
) -> None:
    """Audit identified real condition effects and execute DoWhy refuters."""
    from cog_surp.causal import analyze_real_condition_effects

    if len(h2_surprisal_path) < 2:
        raise typer.BadParameter("provide at least two --h2-surprisal-path values")
    input_paths = [h1_trials_path, *h2_surprisal_path]
    parents = [
        {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in input_paths
    ]
    resolved = {
        "schema_version": 1,
        "command": "analyze causal-condition",
        "estimands": ["condition -> human N400", "condition -> model surprisal"],
        "parents": parents,
        "implementation_sha256": {
            path.name: sha256_file(path)
            for path in (
                Path(__file__).parents[1] / "causal" / "dowhy_analysis.py",
                Path(__file__).parents[1] / "causal" / "graph.py",
                Path(__file__).parents[1] / "causal" / "real_data.py",
            )
        },
    }
    payload = canonical_json_bytes(resolved)
    config_hash = hashlib.sha256(payload).hexdigest()
    effective_run_id = run_id or f"causal-{config_hash[:12]}"
    run_root = Path("artifacts") / "runs" / effective_run_id
    run_root.mkdir(parents=True, exist_ok=True)
    resolved_path = run_root / "resolved-config.json"
    if resolved_path.exists() and resolved_path.read_bytes() != payload:
        raise typer.BadParameter(
            f"run ID {effective_run_id!r} already has a different immutable config"
        )
    manifest_path = run_root / "causal-manifest.json"
    if resolved_path.exists() and _valid_artifact_manifest(run_root, manifest_path):
        typer.echo(
            json.dumps(
                {
                    "event": "causal_condition_reused",
                    "run_id": effective_run_id,
                    "artifact_manifest": str(manifest_path.resolve()),
                },
                sort_keys=True,
            )
        )
        return
    resolved_path.write_bytes(payload)
    artifacts = analyze_real_condition_effects(
        h1_trials_path=h1_trials_path,
        h2_surprisal_paths=h2_surprisal_path,
        output_dir=run_root,
        run_id=effective_run_id,
    )
    artifact_paths = tuple(asdict(artifacts).values())
    manifest = {
        "schema_version": 1,
        "run_id": effective_run_id,
        "data_status": "real",
        "claim_boundary": "separate A->Y and A->S effects; no S->Y edge",
        "parents": parents,
        "artifacts": [
            {
                "path": path.name,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in artifact_paths
        ],
    }
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    typer.echo(
        json.dumps(
            {
                "event": "causal_condition_completed",
                "run_id": effective_run_id,
                "report": str(artifacts.report.resolve()),
                "artifact_manifest": str(manifest_path.resolve()),
            },
            sort_keys=True,
        )
    )


@analyze_app.command("predictive")
def analyze_predictive(
    features_path: Annotated[
        Path,
        typer.Option(help="Joined real-data feature Parquet artifact."),
    ],
    folds: Annotated[
        int,
        typer.Option(min=2, help="Grouped cross-validation folds."),
    ] = 5,
    run_id: Annotated[
        str | None,
        typer.Option(help="Reuse an explicit immutable run ID."),
    ] = None,
) -> None:
    """Compare controls and computational predictors on grouped holdouts."""
    resolved = {
        "schema_version": 1,
        "command": "analyze predictive",
        "features_path": str(features_path.resolve()),
        "features_sha256": sha256_file(features_path),
        "folds": folds,
        "splits": ["leave-items-out", "leave-participants-out"],
        "estimand": "held-out predictive alignment",
        "implementation_sha256": sha256_file(
            Path(__file__).parents[1] / "analysis" / "predictive.py"
        ),
    }
    config_hash = hashlib.sha256(canonical_json_bytes(resolved)).hexdigest()
    effective_run_id = run_id or f"predictive-{config_hash[:12]}"
    run_root = Path("artifacts") / "runs" / effective_run_id
    run_root.mkdir(parents=True, exist_ok=True)
    resolved_path = run_root / "resolved-config.json"
    payload = canonical_json_bytes(resolved)
    if resolved_path.exists() and resolved_path.read_bytes() != payload:
        raise typer.BadParameter(
            f"run ID {effective_run_id!r} already has a different immutable config"
        )
    manifest_path = run_root / "analysis-manifest.json"
    if resolved_path.exists() and _valid_artifact_manifest(run_root, manifest_path):
        typer.echo(
            json.dumps(
                {
                    "event": "predictive_analysis_reused",
                    "run_id": effective_run_id,
                    "artifact_manifest": str(manifest_path.resolve()),
                },
                sort_keys=True,
            )
        )
        return
    resolved_path.write_bytes(payload)
    import pandas as pd

    features = pd.read_parquet(features_path)
    results = evaluate_held_out_models(features, folds=folds)
    artifact = run_root / "held-out-metrics.parquet"
    results.to_parquet(artifact, index=False)
    summary = (
        results.groupby(["split", "model"], as_index=False)
        .agg(
            mean_rmse_uv=("rmse_uv", "mean"),
            sd_rmse_uv=("rmse_uv", "std"),
            mean_r2=("r2", "mean"),
            sd_r2=("r2", "std"),
            folds=("fold", "count"),
        )
        .sort_values(["split", "mean_rmse_uv"])
    )
    summary_artifact = run_root / "held-out-summary.parquet"
    summary.to_parquet(summary_artifact, index=False)
    manifest = {
        "schema_version": 1,
        "run_id": effective_run_id,
        "data_status": "real",
        "estimand": "held-out predictive alignment",
        "claim_boundary": "Performance differences are predictive, not causal.",
        "parent": {
            "path": str(features_path.resolve()),
            "sha256": sha256_file(features_path),
        },
        "artifacts": [
            {
                "path": path.name,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in (artifact, summary_artifact)
        ],
    }
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    typer.echo(
        json.dumps(
            {
                "event": "predictive_analysis_completed",
                "run_id": effective_run_id,
                "artifact": str(artifact.resolve()),
                "summary": str(summary_artifact.resolve()),
            },
            sort_keys=True,
        )
    )


@analyze_app.command("compare-models")
def analyze_compare_models(
    reference_scores: Annotated[Path, typer.Option(help="Reference score artifact.")],
    comparison_scores: Annotated[Path, typer.Option(help="Comparison score artifact.")],
    reference_posterior: Annotated[
        Path, typer.Option(help="Reference posterior summary.")
    ],
    comparison_posterior: Annotated[
        Path, typer.Option(help="Comparison posterior summary.")
    ],
    reference_predictive: Annotated[
        Path, typer.Option(help="Reference held-out summary.")
    ],
    comparison_predictive: Annotated[
        Path, typer.Option(help="Comparison held-out summary.")
    ],
    run_id: Annotated[
        str | None,
        typer.Option(help="Reuse an explicit immutable run ID."),
    ] = None,
) -> None:
    """Materialize a two-family model robustness comparison."""
    inputs = [
        reference_scores,
        comparison_scores,
        reference_posterior,
        comparison_posterior,
        reference_predictive,
        comparison_predictive,
    ]
    parents = [
        {"path": str(path.resolve()), "sha256": sha256_file(path)} for path in inputs
    ]
    resolved = {
        "schema_version": 1,
        "command": "analyze compare-models",
        "parents": parents,
        "implementation_sha256": sha256_file(
            Path(__file__).parents[1] / "analysis" / "robustness.py"
        ),
    }
    config_hash = hashlib.sha256(canonical_json_bytes(resolved)).hexdigest()
    effective_run_id = run_id or f"robustness-{config_hash[:12]}"
    run_root = Path("artifacts") / "runs" / effective_run_id
    run_root.mkdir(parents=True, exist_ok=True)
    resolved_path = run_root / "resolved-config.json"
    payload = canonical_json_bytes(resolved)
    if resolved_path.exists() and resolved_path.read_bytes() != payload:
        raise typer.BadParameter(
            f"run ID {effective_run_id!r} already has a different immutable config"
        )
    manifest_path = run_root / "robustness-manifest.json"
    if resolved_path.exists() and _valid_artifact_manifest(run_root, manifest_path):
        typer.echo(
            json.dumps(
                {
                    "event": "model_robustness_reused",
                    "run_id": effective_run_id,
                    "artifact_manifest": str(manifest_path.resolve()),
                },
                sort_keys=True,
            )
        )
        return
    resolved_path.write_bytes(payload)
    item_frame, summary_frame = compare_two_models(
        reference_scores=reference_scores,
        comparison_scores=comparison_scores,
        reference_posterior=reference_posterior,
        comparison_posterior=comparison_posterior,
        reference_predictive=reference_predictive,
        comparison_predictive=comparison_predictive,
    )
    items = run_root / "item-comparison.parquet"
    summary = run_root / "model-comparison.parquet"
    item_frame.to_parquet(items, index=False)
    summary_frame.to_parquet(summary, index=False)
    manifest = {
        "schema_version": 1,
        "run_id": effective_run_id,
        "data_status": "real",
        "claim_boundary": "cross-model predictive robustness, not homology",
        "parents": parents,
        "artifacts": [
            {
                "path": path.name,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in (items, summary)
        ],
    }
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    typer.echo(
        json.dumps(
            {
                "event": "model_robustness_completed",
                "run_id": effective_run_id,
                "shared_items": len(item_frame),
                "pearson": float(summary_frame["cross_model_pearson"].iloc[0]),
                "summary": str(summary.resolve()),
            },
            sort_keys=True,
        )
    )


@analyze_app.command("fit")
def analyze_fit(
    config_path: Annotated[
        Path,
        typer.Option("--config", help="Versioned Bayesian analysis YAML."),
    ],
    features_path: Annotated[
        Path,
        typer.Option(help="Joined real-data feature Parquet artifact."),
    ],
    run_id: Annotated[
        str | None,
        typer.Option(help="Reuse an explicit immutable run ID."),
    ] = None,
) -> None:
    """Fit the crossed participant/item Bayesian N400 model."""
    config = AnalysisConfig.from_yaml(config_path)
    resolved = {
        "schema_version": 1,
        "command": "analyze fit",
        "features_path": str(features_path.resolve()),
        "features_sha256": sha256_file(features_path),
        "configuration": config.model_dump(mode="json"),
        "implementation_sha256": sha256_file(
            Path(__file__).parents[1] / "analysis" / "bayesian.py"
        ),
    }
    config_hash = hashlib.sha256(canonical_json_bytes(resolved)).hexdigest()
    effective_run_id = run_id or f"analysis-{config_hash[:12]}"
    run_root = Path("artifacts") / "runs" / effective_run_id
    run_root.mkdir(parents=True, exist_ok=True)
    resolved_path = run_root / "resolved-config.json"
    payload = canonical_json_bytes(resolved)
    if resolved_path.exists() and resolved_path.read_bytes() != payload:
        raise typer.BadParameter(
            f"run ID {effective_run_id!r} already has a different immutable config"
        )
    manifest_path = run_root / "analysis-manifest.json"
    if resolved_path.exists() and _valid_artifact_manifest(run_root, manifest_path):
        typer.echo(
            json.dumps(
                {
                    "event": "bayesian_analysis_reused",
                    "run_id": effective_run_id,
                    "artifact_manifest": str(manifest_path.resolve()),
                },
                sort_keys=True,
            )
        )
        return
    resolved_path.write_bytes(payload)
    typer.echo(
        json.dumps(
            {
                "event": "bayesian_analysis_started",
                "run_id": effective_run_id,
                "analysis_status": config.analysis_status,
            },
            sort_keys=True,
        ),
        err=True,
    )
    artifacts = fit_hierarchical_model(
        features_path=features_path,
        config=config,
        output_dir=run_root,
        run_id=effective_run_id,
    )
    outputs = (artifacts.posterior, artifacts.summary, artifacts.diagnostics)
    manifest = {
        "schema_version": 1,
        "run_id": effective_run_id,
        "data_status": "real",
        "analysis_status": config.analysis_status,
        "parent": {
            "path": str(features_path.resolve()),
            "sha256": sha256_file(features_path),
        },
        "artifacts": [
            {
                "path": path.name,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in outputs
        ],
    }
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    typer.echo(
        json.dumps(
            {
                "event": "bayesian_analysis_completed",
                "run_id": effective_run_id,
                "posterior": str(artifacts.posterior.resolve()),
                "diagnostics": str(artifacts.diagnostics.resolve()),
            },
            sort_keys=True,
        )
    )


@report_app.command("build")
def report_build(
    features_path: Annotated[Path, typer.Option(help="Feature artifact.")],
    predictive_summary_path: Annotated[
        Path,
        typer.Option(help="Held-out summary artifact."),
    ],
    posterior_summary_path: Annotated[
        Path,
        typer.Option(help="Posterior summary artifact."),
    ],
    diagnostics_path: Annotated[
        Path,
        typer.Option(help="Bayesian diagnostics JSON."),
    ],
    robustness_path: Annotated[
        Path | None,
        typer.Option(help="Optional cross-model robustness summary."),
    ] = None,
    h1_path: Annotated[
        Path | None,
        typer.Option(help="Optional controlled ERP CORE H1 estimate."),
    ] = None,
    h2_path: Annotated[
        Path | None,
        typer.Option(help="Optional controlled ERP CORE H2 estimate."),
    ] = None,
    causal_path: Annotated[
        Path | None,
        typer.Option(help="Optional real-data DoWhy causal audit."),
    ] = None,
    cluster_path: Annotated[
        Path | None,
        typer.Option(help="Optional exploratory sensor-time metadata."),
    ] = None,
    run_id: Annotated[
        str | None,
        typer.Option(help="Reuse an explicit immutable run ID."),
    ] = None,
) -> None:
    """Build a bounded, traceable Markdown research report."""
    inputs = [
        features_path,
        predictive_summary_path,
        posterior_summary_path,
        diagnostics_path,
    ]
    if robustness_path is not None:
        inputs.append(robustness_path)
    if h1_path is not None:
        inputs.append(h1_path)
    if h2_path is not None:
        inputs.append(h2_path)
    if causal_path is not None:
        inputs.append(causal_path)
    if cluster_path is not None:
        inputs.append(cluster_path)
    parents = [
        {"path": str(path.resolve()), "sha256": sha256_file(path)} for path in inputs
    ]
    resolved = {
        "schema_version": 1,
        "command": "report build",
        "parents": parents,
        "data_status": "real",
        "implementation_sha256": sha256_file(
            Path(__file__).parents[1] / "reporting" / "research_report.py"
        ),
    }
    config_hash = hashlib.sha256(canonical_json_bytes(resolved)).hexdigest()
    effective_run_id = run_id or f"report-{config_hash[:12]}"
    run_root = Path("artifacts") / "runs" / effective_run_id
    run_root.mkdir(parents=True, exist_ok=True)
    resolved_path = run_root / "resolved-config.json"
    payload = canonical_json_bytes(resolved)
    if resolved_path.exists() and resolved_path.read_bytes() != payload:
        raise typer.BadParameter(
            f"run ID {effective_run_id!r} already has a different immutable config"
        )
    manifest_path = run_root / "report-manifest.json"
    if resolved_path.exists() and _valid_artifact_manifest(run_root, manifest_path):
        typer.echo(
            json.dumps(
                {
                    "event": "report_build_reused",
                    "run_id": effective_run_id,
                    "artifact_manifest": str(manifest_path.resolve()),
                },
                sort_keys=True,
            )
        )
        return
    resolved_path.write_bytes(payload)
    artifact = run_root / "research-report.md"
    build_research_report(
        features_path=features_path,
        predictive_summary_path=predictive_summary_path,
        posterior_summary_path=posterior_summary_path,
        diagnostics_path=diagnostics_path,
        output_path=artifact,
        robustness_path=robustness_path,
        h1_path=h1_path,
        h2_path=h2_path,
        causal_path=causal_path,
        cluster_path=cluster_path,
    )
    manifest = {
        "schema_version": 1,
        "run_id": effective_run_id,
        "data_status": "real",
        "parents": parents,
        "artifact": {
            "path": artifact.name,
            "sha256": sha256_file(artifact),
            "size_bytes": artifact.stat().st_size,
        },
    }
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    typer.echo(
        json.dumps(
            {
                "event": "report_build_completed",
                "run_id": effective_run_id,
                "artifact": str(artifact.resolve()),
            },
            sort_keys=True,
        )
    )


@provenance_app.command("snapshot")
def provenance_snapshot(
    parent: Annotated[
        list[Path] | None,
        typer.Option("--parent", help="Parent artifact; repeat as needed."),
    ] = None,
    run_id: Annotated[
        str | None,
        typer.Option(help="Reuse an explicit immutable run ID."),
    ] = None,
) -> None:
    """Hash code, environment, dependencies, hardware, and parent artifacts."""
    parents = sorted(parent or [], key=lambda path: str(path.resolve()))
    parent_records = [
        {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in parents
    ]
    environment = collect_runtime_provenance(Path.cwd())
    package_hash = hashlib.sha256(
        canonical_json_bytes(environment["packages"])
    ).hexdigest()
    resolved = {
        "schema_version": 1,
        "command": "provenance snapshot",
        "parents": parent_records,
        "code": environment["code"],
        "runtime": {
            "python": environment["runtime"]["python"],
            "platform": environment["runtime"]["platform"],
        },
        "packages_sha256": package_hash,
    }
    config_hash = hashlib.sha256(canonical_json_bytes(resolved)).hexdigest()
    effective_run_id = run_id or f"provenance-{config_hash[:12]}"
    run_root = Path("artifacts") / "runs" / effective_run_id
    run_root.mkdir(parents=True, exist_ok=True)
    resolved_path = run_root / "resolved-config.json"
    payload = canonical_json_bytes(resolved)
    if resolved_path.exists() and resolved_path.read_bytes() != payload:
        raise typer.BadParameter(
            f"run ID {effective_run_id!r} already has a different immutable config"
        )
    resolved_path.write_bytes(payload)
    artifact = run_root / "environment.json"
    manifest_path = run_root / "provenance-manifest.json"
    if artifact.exists() and manifest_path.exists():
        typer.echo(
            json.dumps(
                {
                    "event": "provenance_snapshot_reused",
                    "run_id": effective_run_id,
                    "artifact": str(artifact.resolve()),
                },
                sort_keys=True,
            )
        )
        return
    artifact.write_bytes(canonical_json_bytes(environment))
    manifest = {
        "schema_version": 1,
        "run_id": effective_run_id,
        "parents": parent_records,
        "artifact": {
            "path": artifact.name,
            "sha256": sha256_file(artifact),
            "size_bytes": artifact.stat().st_size,
        },
    }
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    typer.echo(
        json.dumps(
            {
                "event": "provenance_snapshot_completed",
                "run_id": effective_run_id,
                "artifact": str(artifact.resolve()),
                "code_tree_sha256": environment["code"]["code_tree_sha256"],
                "uv_lock_sha256": environment["code"]["uv_lock_sha256"],
            },
            sort_keys=True,
        )
    )


@lm_app.command("compare-strategies")
def lm_compare_strategies(
    reference_path: Annotated[
        Path,
        typer.Option(help="Reference LM score artifact."),
    ],
    comparison_path: Annotated[
        Path,
        typer.Option(help="Alternative-strategy score artifact."),
    ],
    run_id: Annotated[
        str | None,
        typer.Option(help="Reuse an explicit immutable run ID."),
    ] = None,
) -> None:
    """Materialize a strategy-only target-probability comparison."""
    parents = [
        {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in (reference_path, comparison_path)
    ]
    resolved = {
        "schema_version": 1,
        "command": "lm compare-strategies",
        "parents": parents,
        "implementation_sha256": sha256_file(
            Path(__file__).parents[1] / "lm" / "comparison.py"
        ),
    }
    config_hash = hashlib.sha256(canonical_json_bytes(resolved)).hexdigest()
    effective_run_id = run_id or f"strategy-{config_hash[:12]}"
    run_root = Path("artifacts") / "runs" / effective_run_id
    run_root.mkdir(parents=True, exist_ok=True)
    resolved_path = run_root / "resolved-config.json"
    payload = canonical_json_bytes(resolved)
    if resolved_path.exists() and resolved_path.read_bytes() != payload:
        raise typer.BadParameter(
            f"run ID {effective_run_id!r} already has a different immutable config"
        )
    manifest_path = run_root / "strategy-manifest.json"
    if resolved_path.exists() and _valid_artifact_manifest(run_root, manifest_path):
        typer.echo(
            json.dumps(
                {
                    "event": "strategy_comparison_reused",
                    "run_id": effective_run_id,
                    "artifact_manifest": str(manifest_path.resolve()),
                },
                sort_keys=True,
            )
        )
        return
    resolved_path.write_bytes(payload)
    frame, summary = compare_probability_strategies(
        reference_path,
        comparison_path,
    )
    artifact = run_root / "strategy-comparison.parquet"
    summary_path = run_root / "strategy-summary.json"
    frame.to_parquet(artifact, index=False)
    summary_path.write_bytes(canonical_json_bytes(summary))
    manifest = {
        "schema_version": 1,
        "run_id": effective_run_id,
        "parents": parents,
        "artifacts": [
            {
                "path": path.name,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in (artifact, summary_path)
        ],
    }
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    typer.echo(
        json.dumps(
            {
                "event": "strategy_comparison_completed",
                "run_id": effective_run_id,
                "shared_items": summary["shared_items"],
                "max_absolute_difference_nats": summary["max_absolute_difference_nats"],
                "summary": str(summary_path.resolve()),
            },
            sort_keys=True,
        )
    )


@lm_app.command("score")
def lm_score(
    config_path: Annotated[
        Path,
        typer.Option("--config", help="Versioned model-matrix YAML."),
    ],
    stimuli_path: Annotated[
        Path | None,
        typer.Option(help="Validated stimulus Parquet artifact."),
    ] = None,
    model_role: Annotated[
        str,
        typer.Option(help="Role from the configured model matrix."),
    ] = "cpu-smoke",
    strategy_name: Annotated[
        str,
        typer.Option(
            "--strategy",
            help="Token-to-region strategy: boundary-aware or subtoken-sum.",
        ),
    ] = "boundary-aware",
    batch_size: Annotated[
        int,
        typer.Option(min=1, help="Teacher-forced inference batch size."),
    ] = 16,
    run_id: Annotated[
        str | None,
        typer.Option(help="Reuse an explicit immutable run ID."),
    ] = None,
) -> None:
    """Score validated targets using exact teacher-forced causal inference."""
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or not isinstance(loaded.get("models"), list):
        raise typer.BadParameter("model matrix must contain a models list")
    matches = [model for model in loaded["models"] if model.get("role") == model_role]
    if len(matches) != 1:
        raise typer.BadParameter(
            f"expected one model with role {model_role!r}, found {len(matches)}"
        )
    model = matches[0]
    required_model_keys = {
        "repository_id",
        "revision",
        "license",
        "dtype",
        "quantization",
    }
    missing = required_model_keys - set(model)
    if missing:
        raise typer.BadParameter(f"model config missing keys: {sorted(missing)}")
    if model["quantization"] != "none":
        raise typer.BadParameter("principal scorer does not accept quantized models")
    valid_strategies = {"boundary-aware", "subtoken-sum"}
    if strategy_name not in valid_strategies:
        raise typer.BadParameter(
            f"strategy must be one of {sorted(valid_strategies)}, got {strategy_name!r}"
        )
    strategy = (
        BoundaryAwareStrategy()
        if strategy_name == "boundary-aware"
        else SubtokenSumStrategy()
    )
    source = stimuli_path or _latest_stimulus_artifact()
    resolved = {
        "schema_version": 1,
        "command": "lm score",
        "model": model,
        "probability_strategy": strategy_name,
        "batch_size": batch_size,
        "stimuli_path": str(source.resolve()),
        "stimuli_sha256": sha256_file(source),
        "implementation_sha256": {
            path.name: sha256_file(path)
            for path in (
                Path(__file__).parents[1] / "lm" / "backends.py",
                Path(__file__).parents[1] / "lm" / "pipeline.py",
                Path(__file__).parents[1] / "lm" / "regions.py",
            )
        },
    }
    config_hash = hashlib.sha256(canonical_json_bytes(resolved)).hexdigest()
    effective_run_id = run_id or f"lm-{model_role}-{config_hash[:12]}"
    run_root = Path("artifacts") / "runs" / effective_run_id
    run_root.mkdir(parents=True, exist_ok=True)
    resolved_path = run_root / "resolved-config.json"
    payload = canonical_json_bytes(resolved)
    if resolved_path.exists() and resolved_path.read_bytes() != payload:
        raise typer.BadParameter(
            f"run ID {effective_run_id!r} already has a different immutable config"
        )
    manifest_path = run_root / "lm-manifest.json"
    if resolved_path.exists() and _valid_artifact_manifest(run_root, manifest_path):
        typer.echo(
            json.dumps(
                {
                    "event": "lm_score_reused",
                    "run_id": effective_run_id,
                    "artifact_manifest": str(manifest_path.resolve()),
                },
                sort_keys=True,
            )
        )
        return
    resolved_path.write_bytes(payload)
    typer.echo(
        json.dumps(
            {
                "event": "lm_score_started",
                "run_id": effective_run_id,
                "model_id": model["repository_id"],
                "model_revision": model["revision"],
                "probability_strategy": strategy_name,
            },
            sort_keys=True,
        ),
        err=True,
    )
    backend = TransformersBackend(
        model_id=str(model["repository_id"]),
        revision=str(model["revision"]),
        strategy=strategy,
        dtype=str(model["dtype"]),
        device="cpu",
    )
    artifact = run_root / "surprisal.parquet"
    record_count = score_stimulus_artifact(
        stimuli_path=source,
        backend=backend,
        output_path=artifact,
        batch_size=batch_size,
    )
    manifest = {
        "schema_version": 1,
        "run_id": effective_run_id,
        "model": model,
        "probability_strategy": strategy_name,
        "input": {"path": str(source.resolve()), "sha256": sha256_file(source)},
        "artifact": {
            "path": artifact.name,
            "sha256": sha256_file(artifact),
            "size_bytes": artifact.stat().st_size,
            "records": record_count,
        },
    }
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    typer.echo(
        json.dumps(
            {
                "event": "lm_score_completed",
                "run_id": effective_run_id,
                "records": record_count,
                "artifact": str(artifact.resolve()),
                "artifact_sha256": manifest["artifact"]["sha256"],
            },
            sort_keys=True,
        )
    )


@app.command()
def doctor(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON."),
    ] = False,
) -> None:
    """Inspect runtime capabilities and fail only on required checks."""
    checks = collect_doctor_checks()
    failed = any(check.required and check.status == "fail" for check in checks)
    if json_output:
        typer.echo(
            json.dumps(
                {
                    "schema_version": 1,
                    "ok": not failed,
                    "checks": [asdict(check) for check in checks],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        table = Table(title="Cog-Surp environment")
        table.add_column("Check")
        table.add_column("Status")
        table.add_column("Required")
        table.add_column("Detail")
        for check in checks:
            table.add_row(
                check.name,
                check.status,
                "yes" if check.required else "no",
                check.detail,
            )
        console.print(table)
    if failed:
        raise typer.Exit(code=1)


def main() -> None:
    """Run the command-line application."""
    app()


if __name__ == "__main__":
    main()
