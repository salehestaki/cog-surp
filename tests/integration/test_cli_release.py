from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from cog_surp import __version__
from cog_surp.cli.app import app
from cog_surp.release import build_synthetic_demo

runner = CliRunner()


def test_root_help_and_version() -> None:
    help_result = runner.invoke(app, ["--help"])
    version_result = runner.invoke(app, ["--version"])

    assert help_result.exit_code == 0
    assert "Reproducible surprisal-N400" in help_result.stdout
    assert version_result.exit_code == 0
    assert version_result.stdout.strip() == __version__


def test_doctor_and_dataset_listing() -> None:
    doctor = runner.invoke(app, ["doctor", "--json"])
    datasets = runner.invoke(app, ["datasets", "list"])

    assert doctor.exit_code == 0
    assert '"ok": true' in doctor.stdout
    assert datasets.exit_code == 0
    assert "erp-core-n400" in datasets.stdout
    assert "derco" in datasets.stdout


def test_manifest_validation_and_dashboard_help(tmp_path: Path) -> None:
    manifest_path, manifest = build_synthetic_demo(
        output_dir=tmp_path / "bundle",
        project_root=Path.cwd(),
    )

    validation = runner.invoke(
        app,
        ["report", "validate-manifest", "--manifest", str(manifest_path)],
    )
    dashboard_help = runner.invoke(app, ["app", "run", "--help"])

    assert validation.exit_code == 0
    assert manifest.release_id in validation.stdout
    assert dashboard_help.exit_code == 0
    assert "--manifest" in dashboard_help.stdout


def test_invalid_release_configuration_is_actionable(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("schema_version: 1\n", encoding="utf-8")

    result = runner.invoke(
        app,
        ["report", "manifest", "--config", str(invalid)],
    )

    assert result.exit_code == 2
    assert "Invalid value" in result.output


def test_representative_subcommand_help() -> None:
    for command in (
        ["eeg", "preprocess", "--help"],
        ["lm", "score", "--help"],
        ["analyze", "fit", "--help"],
        ["demo", "build", "--help"],
    ):
        result = runner.invoke(app, command)
        assert result.exit_code == 0, (command, result.stdout)
