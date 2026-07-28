"""Artifact-only Streamlit dashboard package."""

from cog_surp.dashboard.bundle import (
    DashboardBundle,
    global_status_message,
    load_dashboard_bundle,
    panel_status_message,
)

__all__ = [
    "DashboardBundle",
    "global_status_message",
    "load_dashboard_bundle",
    "panel_status_message",
]
