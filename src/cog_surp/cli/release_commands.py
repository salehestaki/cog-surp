"""Release-manifest, demo, and dashboard CLI commands."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

from cog_surp.release import (
    ManifestValidationError,
    ReleaseBuildConfig,
    build_release_bundle,
    build_synthetic_demo,
    load_release_manifest,
)

dashboard_app = typer.Typer(help="Inspect completed artifacts in Streamlit.")
demo_app = typer.Typer(help="Build the deterministic public synthetic demo.")


def register_report_commands(report_app: typer.Typer) -> None:
    """Attach manifest commands to the existing report command group."""
    report_app.command("manifest")(report_manifest)
    report_app.command("validate-manifest")(report_validate_manifest)


def report_manifest(
    config_path: Annotated[
        Path,
        typer.Option("--config", help="Versioned release-bundle YAML."),
    ],
    output_root: Annotated[
        Path,
        typer.Option(help="Immutable release-bundle directory."),
    ] = Path("artifacts/releases"),
) -> None:
    """Build and validate one coherent report/dashboard release manifest."""
    try:
        config = ReleaseBuildConfig.from_yaml(config_path)
        manifest_path, manifest = build_release_bundle(
            config=config,
            output_root=output_root,
            project_root=Path.cwd(),
        )
    except (ManifestValidationError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(
        json.dumps(
            {
                "event": "release_manifest_completed",
                "release_id": manifest.release_id,
                "data_status": manifest.data_status.value,
                "artifacts": len(manifest.artifacts),
                "datasets": [value.dataset_id for value in manifest.datasets],
                "models": [value.model_id for value in manifest.models],
                "manifest": str(manifest_path.resolve()),
            },
            sort_keys=True,
        )
    )


def report_validate_manifest(
    manifest_path: Annotated[
        Path,
        typer.Option("--manifest", help="Unified release manifest."),
    ],
) -> None:
    """Verify release schema, lineage, safe paths, and every artifact checksum."""
    try:
        manifest = load_release_manifest(manifest_path)
    except ManifestValidationError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(
        json.dumps(
            {
                "event": "release_manifest_valid",
                "release_id": manifest.release_id,
                "data_status": manifest.data_status.value,
                "artifacts": len(manifest.artifacts),
            },
            sort_keys=True,
        )
    )


@dashboard_app.command("run")
def dashboard_run(
    manifest_path: Annotated[
        Path,
        typer.Option(
            "--manifest",
            help="Validated unified release manifest.",
        ),
    ] = Path("demo/bundle/release-manifest.json"),
) -> None:
    """Launch the manifest-only Streamlit dashboard."""
    try:
        load_release_manifest(manifest_path)
    except ManifestValidationError as error:
        raise typer.BadParameter(str(error)) from error
    source = Path(__file__).parents[1] / "dashboard" / "app.py"
    environment = os.environ.copy()
    environment["COG_SURP_MANIFEST"] = str(manifest_path.resolve())
    result = subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(source)],
        check=False,
        env=environment,
    )
    if result.returncode:
        raise typer.Exit(result.returncode)


@demo_app.command("build")
def demo_build(
    output_dir: Annotated[
        Path,
        typer.Option("--output", help="Synthetic demo bundle directory."),
    ] = Path("demo/bundle"),
) -> None:
    """Build or validate the deterministic CPU-only synthetic demo."""
    try:
        manifest_path, manifest = build_synthetic_demo(
            output_dir=output_dir,
            project_root=Path.cwd(),
        )
    except (ManifestValidationError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(
        json.dumps(
            {
                "event": "synthetic_demo_ready",
                "release_id": manifest.release_id,
                "data_status": manifest.data_status.value,
                "artifacts": len(manifest.artifacts),
                "manifest": str(manifest_path.resolve()),
            },
            sort_keys=True,
        )
    )
