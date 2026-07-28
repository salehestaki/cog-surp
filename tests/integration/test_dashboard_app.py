from __future__ import annotations

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from cog_surp.release import ArtifactType, build_synthetic_demo

APP_PATH = Path("src/cog_surp/dashboard/app.py").resolve()


@pytest.fixture
def demo_manifest(tmp_path: Path) -> Path:
    manifest_path, _ = build_synthetic_demo(
        output_dir=tmp_path / "bundle",
        project_root=Path.cwd(),
    )
    return manifest_path


def _run_dashboard(monkeypatch: pytest.MonkeyPatch, manifest_path: Path) -> AppTest:
    monkeypatch.setenv("COG_SURP_MANIFEST", str(manifest_path))
    return AppTest.from_file(str(APP_PATH), default_timeout=30).run()


def _manifest_json(path: Path) -> dict[str, object]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _write_manifest(path: Path, value: dict[str, object]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_dashboard_starts_and_renders_all_sections(
    monkeypatch: pytest.MonkeyPatch,
    demo_manifest: Path,
) -> None:
    app = _run_dashboard(monkeypatch, demo_manifest)

    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "Overview",
        "Stimuli & model",
        "Human EEG",
        "Alignment",
        "Causal assumptions",
        "Provenance",
    ]
    assert any("NOT HUMAN EVIDENCE" in error.value for error in app.error)
    assert any("Active release:" in caption.value for caption in app.caption)


def test_provenance_displays_release_and_git_revision(
    monkeypatch: pytest.MonkeyPatch,
    demo_manifest: Path,
) -> None:
    manifest = _manifest_json(demo_manifest)
    app = _run_dashboard(monkeypatch, demo_manifest)
    rendered_json = "\n".join(str(element.value) for element in app.json)

    assert str(manifest["release_id"]) in rendered_json
    assert str(manifest["git_commit"]) in rendered_json


def test_missing_artifact_is_actionable_in_dashboard(
    monkeypatch: pytest.MonkeyPatch,
    demo_manifest: Path,
) -> None:
    manifest = _manifest_json(demo_manifest)
    artifact = manifest["artifacts"][0]  # type: ignore[index]
    (demo_manifest.parent / artifact["path"]).unlink()  # type: ignore[index]

    app = _run_dashboard(monkeypatch, demo_manifest)

    assert not app.exception
    assert any("artifact is missing" in error.value for error in app.error)


def test_checksum_corruption_is_detected_in_dashboard(
    monkeypatch: pytest.MonkeyPatch,
    demo_manifest: Path,
) -> None:
    manifest = _manifest_json(demo_manifest)
    artifact = manifest["artifacts"][0]  # type: ignore[index]
    (demo_manifest.parent / artifact["path"]).write_text(  # type: ignore[index]
        "corrupt",
        encoding="utf-8",
    )

    app = _run_dashboard(monkeypatch, demo_manifest)

    assert not app.exception
    assert any("checksum mismatch" in error.value for error in app.error)


def test_incompatible_lineage_is_rejected_in_dashboard(
    monkeypatch: pytest.MonkeyPatch,
    demo_manifest: Path,
) -> None:
    manifest = _manifest_json(demo_manifest)
    artifacts = manifest["artifacts"]
    posterior = next(  # type: ignore[arg-type]
        artifact
        for artifact in artifacts
        if artifact["artifact_type"] == ArtifactType.POSTERIOR_SUMMARY.value
    )
    posterior["parent_run_ids"] = ["different-feature-run"]
    _write_manifest(demo_manifest, manifest)

    app = _run_dashboard(monkeypatch, demo_manifest)

    assert not app.exception
    assert any("does not descend from feature run" in item.value for item in app.error)


def test_unsafe_manifest_path_is_rejected_in_dashboard(
    monkeypatch: pytest.MonkeyPatch,
    demo_manifest: Path,
) -> None:
    manifest = _manifest_json(demo_manifest)
    manifest["artifacts"][0]["path"] = "../private-data.parquet"  # type: ignore[index]
    _write_manifest(demo_manifest, manifest)

    app = _run_dashboard(monkeypatch, demo_manifest)

    assert not app.exception
    assert any("cannot contain" in item.value for item in app.error)


def test_status_contracts_with_valid_manifest_fixtures(
    monkeypatch: pytest.MonkeyPatch,
    demo_manifest: Path,
) -> None:
    manifest = _manifest_json(demo_manifest)
    manifest["label"] = "Real-status UI contract fixture; no findings rendered"
    manifest["data_status"] = "real"
    for dataset in manifest["datasets"]:  # type: ignore[union-attr]
        dataset["data_status"] = "real"
    for artifact in manifest["artifacts"]:  # type: ignore[union-attr]
        artifact["data_status"] = "real"
    _write_manifest(demo_manifest, manifest)
    monkeypatch.setenv("STATUS_MANIFEST", str(demo_manifest))
    status_app = AppTest.from_string(
        """
import os
from pathlib import Path
import streamlit as st
from cog_surp.dashboard.bundle import global_status_message
from cog_surp.release import load_release_manifest

manifest = load_release_manifest(Path(os.environ["STATUS_MANIFEST"]))
st.success(global_status_message(manifest.data_status))
"""
    ).run()

    assert not status_app.exception
    assert [item.value for item in status_app.success] == ["REAL HUMAN EEG"]

    manifest["data_status"] = "mixed"
    manifest["artifacts"][0]["data_status"] = "synthetic"  # type: ignore[index]
    _write_manifest(demo_manifest, manifest)
    mixed_app = AppTest.from_string(
        """
import os
from pathlib import Path
import streamlit as st
from cog_surp.dashboard.bundle import global_status_message, panel_status_message
from cog_surp.release import ArtifactType, load_release_manifest

manifest = load_release_manifest(Path(os.environ["STATUS_MANIFEST"]))
st.warning(global_status_message(manifest.data_status))
st.caption(panel_status_message(manifest, {ArtifactType.FEATURES}))
"""
    ).run()

    assert not mixed_app.exception
    assert not mixed_app.success
    assert any("MIXED DATA" in item.value for item in mixed_app.warning)
    assert any("Panel data:" in item.value for item in mixed_app.caption)


def test_dashboard_source_never_discovers_latest_artifacts() -> None:
    source = APP_PATH.read_text(encoding="utf-8")
    bundle_source = Path("src/cog_surp/dashboard/bundle.py").read_text(encoding="utf-8")

    for forbidden in ("_latest", ".glob(", ".rglob(", "getmtime", "st_mtime"):
        assert forbidden not in source
        assert forbidden not in bundle_source
