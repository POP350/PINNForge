"""Stage C: retrieve prior and posterior knowledge for a PDE context."""

from .knowledge_base import KnowledgeBaseManager, PosteriorUpdater
from .pinnacle_registry import (
    get_pinnacle_profile,
    list_pinnacle_problems,
    pinnacle_registry_summary,
)

__all__ = [
    "KnowledgeBaseManager",
    "PosteriorUpdater",
    "get_pinnacle_profile",
    "list_pinnacle_problems",
    "pinnacle_registry_summary",
]

