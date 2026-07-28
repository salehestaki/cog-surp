from __future__ import annotations

from pathlib import Path

from cog_surp.release import ArtifactType, DataStatus, build_synthetic_demo


def test_synthetic_demo_is_deterministic_and_self_validating(
    tmp_path: Path,
) -> None:
    output = tmp_path / "bundle"

    first_path, first = build_synthetic_demo(
        output_dir=output,
        project_root=Path.cwd(),
    )
    second_path, second = build_synthetic_demo(
        output_dir=output,
        project_root=Path.cwd(),
    )

    assert first_path == second_path
    assert first.release_id == second.release_id
    assert first.data_status is DataStatus.SYNTHETIC
    assert {artifact.data_status for artifact in first.artifacts} == {
        DataStatus.SYNTHETIC
    }
    assert len(first.artifacts) == 18
    report = first_path.parent / first.artifact(ArtifactType.REPORT).path
    report_text = report.read_text(encoding="utf-8")
    assert first.release_id in report_text
    assert "NOT HUMAN EVIDENCE" in report_text
    for artifact in first.artifacts:
        artifact_path = first_path.parent / artifact.path
        if artifact_path.suffix in {".json", ".md", ".svg"}:
            assert b"\r\n" not in artifact_path.read_bytes()


def test_demo_refuses_to_overwrite_partial_output(tmp_path: Path) -> None:
    output = tmp_path / "bundle"
    output.mkdir()
    (output / "unexpected.txt").write_text("preserve me", encoding="utf-8")

    try:
        build_synthetic_demo(output_dir=output, project_root=Path.cwd())
    except ValueError as error:
        assert "refusing overwrite" in str(error)
    else:
        raise AssertionError("partial demo output was silently overwritten")
