"""AI Search service package - LangChain 1.0 Clean Implementation.

This package provides intelligent archive search using LangChain 1.0 agents
with structured output only (no verbose text responses).

Usage:
    from app.services.ai_search import get_archive_search_agent
    
    agent = get_archive_search_agent()
    result = agent.search("batik")
    
    print(result["archives"])  # List of matching archives
    print(result["total"])     # Count
    print(result["query"])     # Query echo
"""

from app.services.ai_search.agent_v2 import (
    ArchiveSearchAgentV2,
    get_archive_search_agent
)
from app.services.ai_search.tools import search_archives_db

__all__ = [
    "ArchiveSearchAgentV2",
    "get_archive_search_agent",
    "search_archives_db"
]

__all__ = [
    "ArchiveSearchAgent",
    "get_archive_search_agent",
    "search_archives_db"
]
