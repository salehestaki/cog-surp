"""Real EEG dataset adapters."""

from cog_surp.datasets.derco import DERCoAdapter
from cog_surp.datasets.erp_core import ERPCoreN400Adapter

__all__ = ["DERCoAdapter", "ERPCoreN400Adapter"]
