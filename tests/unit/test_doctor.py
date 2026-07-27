from __future__ import annotations

from pathlib import Path

from cog_surp.cli.app import collect_doctor_checks


def test_doctor_reports_missing_layout(tmp_path: Path) -> None:
    checks = {check.name: check for check in collect_doctor_checks(tmp_path)}

    assert checks["project-layout"].status == "fail"
    assert checks["project-layout"].required
    assert "pyproject.toml" in checks["project-layout"].detail


def test_doctor_accepts_minimum_layout(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").touch()
    (tmp_path / "configs").mkdir()
    (tmp_path / "artifacts").mkdir()

    checks = {check.name: check for check in collect_doctor_checks(tmp_path)}

    assert checks["project-layout"].status == "ok"
