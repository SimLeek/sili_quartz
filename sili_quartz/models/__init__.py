"""One directory, every family, side by side. sili versions will sit here as *_sili.py."""

from sili_quartz.models.combined import Combined, CombinedState
from sili_quartz.models.common import recall_loss
from sili_quartz.models.memory import MemoryState, TitansMemory
from sili_quartz.models.rmt import RMT, RMTState
from sili_quartz.models.titans import TitansMAC
from sili_quartz.models.transformer import Transformer

FAMILIES = {"transformer": Transformer, "rmt": RMT, "titans": TitansMAC, "combined": Combined}

__all__ = [
    "FAMILIES",
    "RMT",
    "Combined",
    "CombinedState",
    "MemoryState",
    "RMTState",
    "TitansMAC",
    "TitansMemory",
    "Transformer",
    "recall_loss",
]
