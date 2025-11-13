"""AI Search service package.

This package provides intelligent archive search using LangChain 1.0 agents.

Usage:
    from app.services.ai_search import get_archive_search_agent
    
    agent = get_archive_search_agent()
    result = agent.search("I want batik")
    
    print(result["message"])  # Agent's response
    print(result["archives"])  # List of matching archives
"""

from app.services.ai_search.agent import (
    ArchiveSearchAgent,
    get_archive_search_agent
)
from app.services.ai_search.tools import search_archives_db

__all__ = [
    "ArchiveSearchAgent",
    "get_archive_search_agent",
    "search_archives_db"
]
