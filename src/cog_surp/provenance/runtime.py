"""Runtime, code-tree, and dependency provenance capture."""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

from cog_surp.provenance.checksums import sha256_file


def _git(
    root: Path,
    arguments: list[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        capture_output=True,
        check=check,
        text=True,
        encoding="utf-8",
    )


def _code_tree(root: Path) -> tuple[str, int]:
    result = _git(
        root,
        ["ls-files", "--cached", "--others", "--exclude-standard"],
        check=False,
    )
    if result.returncode:
        paths = sorted(
            path.relative_to(root)
            for path in root.rglob("*")
            if path.is_file() and ".git" not in path.parts
        )
    else:
        paths = [Path(value) for value in result.stdout.splitlines() if value]
    digest = hashlib.sha256()
    included = 0
    for relative in sorted(paths, key=lambda path: path.as_posix()):
        source = root / relative
        if not source.is_file():
            continue
        digest.update(relative.as_posix().encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(source)))
        included += 1
    return digest.hexdigest(), included


def collect_runtime_provenance(project_root: Path) -> dict[str, Any]:
    """Collect a deterministic code identity plus time-varying runtime metadata."""
    root = project_root.resolve()
    status = _git(root, ["status", "--porcelain=v1"], check=False)
    revision = _git(root, ["rev-parse", "HEAD"], check=False)
    tree_hash, tree_files = _code_tree(root)
    packages = {
        distribution.metadata["Name"]: distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata["Name"]
    }
    lock = root / "uv.lock"
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "project_root": str(root),
        "code": {
            "git_revision": (
                revision.stdout.strip() if revision.returncode == 0 else None
            ),
            "git_dirty": bool(status.stdout.strip()) or revision.returncode != 0,
            "git_status_sha256": hashlib.sha256(status.stdout.encode()).hexdigest(),
            "code_tree_sha256": tree_hash,
            "code_tree_files": tree_files,
            "uv_lock_sha256": sha256_file(lock) if lock.exists() else None,
        },
        "runtime": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "memory_bytes": psutil.virtual_memory().total,
        },
        "packages": dict(sorted(packages.items(), key=lambda item: item[0].casefold())),
    }
