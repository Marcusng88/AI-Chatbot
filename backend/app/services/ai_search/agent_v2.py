"""
Clean AI Search Agent - LangChain 1.0 Best Practices.

This agent returns ONLY structured archive data with NO text responses.
Following chat UX best practices: immediate feedback, structured output only.
"""

import logging
from typing import List, Dict, Any, AsyncIterator, Optional
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage

from app.core.config import settings
from app.services.ai_search.tools import search_archives_db

logger = logging.getLogger(__name__)


# Simplified system prompt - focus on query generation only
SEARCH_AGENT_PROMPT = """You are a heritage archive search assistant.

Your ONLY job is to generate diverse search queries and use the search_archives_db tool.

When a user asks for something:
1. Generate 3-5 diverse query variations covering different aspects
2. Call search_archives_db with ALL queries at once
3. The tool returns structured archive data

DO NOT:
- Write explanatory text or responses
- Describe what you're doing
- Explain the results

ONLY:
- Generate diverse queries
- Call the search tool
- Return the structured results

Examples:
User: "I want batik"
You call: search_archives_db(queries=["batik", "batik textile", "traditional Malaysian batik", "batik fabric heritage", "hand-dyed batik patterns"])

User: "traditional crafts"
You call: search_archives_db(queries=["traditional crafts", "heritage handicrafts", "Malaysian artisan work", "cultural craftsmanship", "historical craft techniques"])
"""


class ArchiveSearchAgentV2:
    """
    Simplified archive search agent focusing on structured output only.
    
    Key improvements:
    - No verbose responses - only structured data
    - Simplified state management
    - Direct query generation via agent
    - Clean streaming support
    """
    
    def __init__(self):
        logger.info("Initializing ArchiveSearchAgentV2 (LangChain 1.0)")
        
        # Initialize Gemini model
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=settings.GOOGLE_GENAI_API_KEY,
            temperature=0.2,  # Lower for focused query generation
        )
        
        # Single tool: search_archives_db
        self.tools = [search_archives_db]
        logger.info(f"Configured with {len(self.tools)} tool(s): {[tool.name for tool in self.tools]}")
        
        # Memory for conversation persistence
        self.memory = MemorySaver()
        
        # Create agent - NO middleware, just pure agent logic
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=SEARCH_AGENT_PROMPT,
            checkpointer=self.memory,
        )
        logger.info("ArchiveSearchAgentV2 initialized")
    
    def search(
        self, 
        user_query: str, 
        thread_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Synchronous search returning ONLY structured archive data.
        
        Args:
            user_query: User's search query
            thread_id: Optional conversation thread ID
            
        Returns:
            {
                "archives": [...],  # Structured archive list
                "total": int,        # Total count
                "query": str         # Echo of user query
            }
        """
        thread_id = thread_id or "default"
        logger.info(f"Search: '{user_query}' (thread={thread_id})")
        
        try:
            config = {"configurable": {"thread_id": thread_id}}
            
            # Invoke agent
            result = self.agent.invoke(
                {"messages": [{"role": "user", "content": user_query}]},
                config=config
            )
            
            # Extract archives from tool artifacts
            archives = self._extract_archives(result)
            
            logger.info(f"Found {len(archives)} archives")
            
            return {
                "archives": archives,
                "total": len(archives),
                "query": user_query
            }
            
        except Exception as e:
            logger.error(f"Search error: {e}", exc_info=True)
            raise
    
    async def search_stream(
        self, 
        user_query: str,
        thread_id: Optional[str] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Streaming search with progressive updates.
        
        Yields:
            - {"type": "searching", "query": str}  # Agent is generating queries
            - {"type": "results", "archives": [...], "total": int}  # Results found
            - {"type": "done"}  # Completion signal
        """
        thread_id = thread_id or "default"
        logger.info(f"Stream search: '{user_query}' (thread={thread_id})")
        
        try:
            config = {"configurable": {"thread_id": thread_id}}
            
            # Immediately acknowledge search started
            yield {
                "type": "searching",
                "query": user_query
            }
            
            all_archives: List[Dict[str, Any]] = []
            
            # Stream agent execution
            async for event in self.agent.astream(
                {"messages": [{"role": "user", "content": user_query}]},
                config=config,
                stream_mode="values"
            ):
                # Extract archives from any tool messages
                archives = self._extract_archives(event)
                
                if archives and len(archives) > len(all_archives):
                    all_archives = archives
                    # Send incremental results
                    yield {
                        "type": "results",
                        "archives": archives,
                        "total": len(archives)
                    }
            
            # Final results
            yield {
                "type": "done",
                "archives": all_archives,
                "total": len(all_archives)
            }
            
            logger.info(f"Stream complete: {len(all_archives)} archives")
            
        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            yield {
                "type": "error",
                "message": str(e)
            }
    
    def _extract_archives(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract archive data from agent result."""
        archives: Dict[str, Dict[str, Any]] = {}
        
        messages = result.get("messages", [])
        
        for msg in messages:
            # Check for tool message with artifact
            if hasattr(msg, "artifact") and msg.artifact:
                if isinstance(msg.artifact, list):
                    for archive in msg.artifact:
                        if isinstance(archive, dict) and "id" in archive:
                            archives[archive["id"]] = archive
        
        return list(archives.values())


# Singleton instance
_agent_instance = None


def get_archive_search_agent() -> ArchiveSearchAgentV2:
    """Get or create the agent singleton."""
    global _agent_instance
    if _agent_instance is None:
        logger.info("Creating new ArchiveSearchAgentV2 singleton")
        _agent_instance = ArchiveSearchAgentV2()
    return _agent_instance
