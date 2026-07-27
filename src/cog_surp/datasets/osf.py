"""Small, testable OSF v2 client used for immutable public datasets."""

from __future__ import annotations

import json
import os
import shutil
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast


@dataclass(frozen=True, slots=True)
class OSFFile:
    """File metadata returned by OSF."""

    file_id: str
    name: str
    materialized_path: str
    size_bytes: int
    download_url: str
    sha256: str | None


Transport = Callable[[str], bytes]
Downloader = Callable[[str, Path], None]


def _default_transport(url: str) -> bytes:
    with _open_url(url) as response:
        return cast(bytes, response.read())


def _open_url(url: str) -> Any:
    proxies = {
        scheme: value.strip()
        for scheme, name in (("http", "HTTP_PROXY"), ("https", "HTTPS_PROXY"))
        if (value := os.environ.get(name))
    }
    opener = urllib.request.build_opener(urllib.request.ProxyHandler(proxies))
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.api+json",
            "User-Agent": "cog-surp/0.1 dataset-adapter",
        },
    )
    return opener.open(request, timeout=60)


def _default_downloader(url: str, destination: Path) -> None:
    with _open_url(url) as response, destination.open("wb") as stream:
        shutil.copyfileobj(response, stream, length=1024 * 1024)


class OSFClient:
    """Read-only paginated traversal of one public OSF storage tree."""

    def __init__(
        self,
        transport: Transport = _default_transport,
        downloader: Downloader = _default_downloader,
    ) -> None:
        self._transport = transport
        self._downloader = downloader

    def entries(self, url: str) -> Iterator[dict[str, Any]]:
        """Yield all entries across OSF pagination."""
        next_url: str | None = url
        while next_url:
            payload = cast(dict[str, Any], json.loads(self._transport(next_url)))
            data = cast(list[dict[str, Any]], payload["data"])
            yield from data
            links = cast(dict[str, Any], payload.get("links", {}))
            next_value = links.get("next")
            next_url = str(next_value) if next_value else None

    def walk_files(
        self,
        url: str,
        descend: Callable[[str], bool] | None = None,
    ) -> Iterator[OSFFile]:
        """Recursively yield files below an OSF folder API URL."""
        for entry in self.entries(url):
            attributes = cast(dict[str, Any], entry["attributes"])
            if attributes["kind"] == "folder":
                folder_path = str(attributes["materialized_path"])
                if descend is not None and not descend(folder_path):
                    continue
                relationships = cast(dict[str, Any], entry["relationships"])
                files = cast(dict[str, Any], relationships["files"])
                links = cast(dict[str, Any], files["links"])
                related = cast(dict[str, Any], links["related"])
                yield from self.walk_files(str(related["href"]), descend)
                continue
            links = cast(dict[str, Any], entry["links"])
            extra = cast(dict[str, Any], attributes.get("extra", {}))
            hashes = cast(dict[str, Any], extra.get("hashes", {}))
            yield OSFFile(
                file_id=str(entry["id"]),
                name=str(attributes["name"]),
                materialized_path=str(
                    PurePosixPath(str(attributes["materialized_path"]))
                ),
                size_bytes=int(attributes["size"]),
                download_url=str(links["download"]),
                sha256=str(hashes["sha256"]) if hashes.get("sha256") else None,
            )

    def find_folder(self, url: str, name: str) -> str:
        """Return the related-files URL for a direct child folder."""
        for entry in self.entries(url):
            attributes = cast(dict[str, Any], entry["attributes"])
            if attributes["kind"] == "folder" and attributes["name"] == name:
                relationships = cast(dict[str, Any], entry["relationships"])
                files = cast(dict[str, Any], relationships["files"])
                links = cast(dict[str, Any], files["links"])
                related = cast(dict[str, Any], links["related"])
                return str(related["href"])
        raise LookupError(f"OSF folder not found: {name}")

    def download(self, source_url: str, destination: Path) -> None:
        """Download one file through the injected transport."""
        self._downloader(source_url, destination)
