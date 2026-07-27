from __future__ import annotations

from pathlib import Path

from cog_surp.provenance import collect_runtime_provenance


def test_runtime_provenance_hashes_code_and_lock(tmp_path: Path) -> None:
    (tmp_path / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (tmp_path / "module.py").write_text("VALUE = 1\n", encoding="utf-8")

    result = collect_runtime_provenance(tmp_path)

    assert result["code"]["code_tree_files"] == 2
    assert len(result["code"]["code_tree_sha256"]) == 64
    assert len(result["code"]["uv_lock_sha256"]) == 64
    assert result["runtime"]["python"]
    assert result["packages"]
