"""Cog-Surp scientific benchmarking workbench."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cog-surp")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = ["__version__"]
