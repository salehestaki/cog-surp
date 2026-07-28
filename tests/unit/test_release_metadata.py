from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import yaml

from cog_surp import __version__


def test_citation_cites_cog_surp_as_software() -> None:
    citation: dict[str, Any] = yaml.safe_load(
        Path("CITATION.cff").read_text(encoding="utf-8")
    )
    preferred = citation["preferred-citation"]

    assert citation["type"] == "software"
    assert "Cog-Surp" in citation["title"]
    assert citation["version"] == __version__
    assert preferred["type"] == "software"
    assert "Cog-Surp" in preferred["title"]
    assert "ERP CORE" not in preferred["title"]
    assert citation["authors"][0]["orcid"] == ("https://orcid.org/0009-0002-0642-4384")


def test_package_metadata_is_consistent() -> None:
    metadata = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]

    assert metadata["version"] == __version__
    assert metadata["license"] == "Apache-2.0"
    assert metadata["authors"][0]["name"] == "Saleh Estaki Organi"
    assert metadata["urls"]["Repository"] == ("https://github.com/salehestaki/cog-surp")


def test_license_contains_complete_apache_2_terms() -> None:
    license_text = Path("LICENSE").read_text(encoding="utf-8")

    for section in (
        "1. Definitions.",
        "2. Grant of Copyright License.",
        "3. Grant of Patent License.",
        "4. Redistribution.",
        "5. Submission of Contributions.",
        "6. Trademarks.",
        "7. Disclaimer of Warranty.",
        "8. Limitation of Liability.",
        "9. Accepting Warranty or Additional Liability.",
        "END OF TERMS AND CONDITIONS",
    ):
        assert section in license_text
