from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from cog_surp.cli import app as cli_app
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


@pytest.mark.parametrize(
    ("free_gib", "expected"),
    [(12.0, "ok"), (8.22, "warning"), (1.5, "fail")],
)
def test_doctor_distinguishes_fixture_and_real_data_disk_needs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    free_gib: float,
    expected: str,
) -> None:
    monkeypatch.setattr(
        cli_app.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=int(free_gib * 1024**3)),
    )

    checks = {check.name: check for check in collect_doctor_checks(tmp_path)}

    assert checks["disk"].status == expected
