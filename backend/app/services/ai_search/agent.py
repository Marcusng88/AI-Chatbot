"""AI Search Agent using LangChain 1.0."""

import logging
from typing import List, Dict, Any, AsyncIterator, Optional
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from app.core.config import settings
from app.services.ai_search.tools import search_archives_db
from app.services.ai_search.prompt import ARCHIVE_SEARCH_SYSTEM_PROMPT
from app.services.ai_search.middleware import (
    ArchiveSearchState,
    ErrorHandlingMiddleware,
    DynamicPromptMiddleware,
    StateTrackingMiddleware
)

# Configure logger
logger = logging.getLogger(__name__)


class ArchiveSearchAgent:
    """
    AI Agent for searching archived materials using LangChain 1.0.
    
    This agent uses Google's Gemini model with intelligent query generation to:
    1. Analyze user requests and generate multiple diverse search queries
    2. Execute vector searches across all query variations
    3. Deduplicate and combine results for comprehensive coverage
    4. Validate results against user intent
    5. Refine searches if results don't match expectations
    6. Track conversation state and search history
    7. Persist conversation across multiple interactions
    
    Architecture (LangChain 1.0 Best Practices):
    - Custom state schema (ArchiveSearchState) for tracking searches
    - Agent-driven multi-query generation (not hardcoded middleware)
    - Error handling middleware with retry logic
    - Dynamic prompt middleware for context injection
    - State tracking middleware for result validation
    - Memory checkpointing for conversation persistence
    
    Key Improvement: The agent now intelligently generates query variations
    based on user intent, rather than using hardcoded query expansion rules.
    """
    
    def __init__(self):
        """Initialize the archive search agent with lightweight middleware stack."""
        logger.info("Initializing ArchiveSearchAgent with LangChain 1.0 architecture")
        
        # Initialize the LLM (Google Gemini)
        logger.debug("Creating ChatGoogleGenerativeAI model (gemini-2.5-flash-lite)")
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=settings.GOOGLE_GENAI_API_KEY,
            temperature=0.3,  # Lower temperature for more focused responses
        )
        
        # Define the tools available to the agent
        self.tools = [search_archives_db]
        logger.info(f"Agent configured with {len(self.tools)} tool(s): {[tool.name for tool in self.tools]}")
        
        # Initialize lightweight middleware stack (removed MultiQueryMiddleware)
        # The agent now handles query generation intelligently via prompting
        self.middleware = [
            ErrorHandlingMiddleware(max_retries=3, backoff_factor=2.0),  # Handle errors with retries
            DynamicPromptMiddleware(max_tokens=4000),  # Manage context and trim messages
            StateTrackingMiddleware(min_similarity_threshold=0.3),  # Track search state
        ]
        logger.info(f"Initialized lightweight middleware stack with {len(self.middleware)} components (no hardcoded multi-query)")
        
        # Initialize memory for conversation persistence
        self.memory = MemorySaver()
        logger.debug("Initialized MemorySaver for conversation checkpointing")
        
        # Create the agent using LangChain 1.0's create_agent
        logger.debug("Creating agent with LangChain 1.0 create_agent")
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            state_schema=ArchiveSearchState,  # Custom state schema
            system_prompt=ARCHIVE_SEARCH_SYSTEM_PROMPT,  # Enhanced prompt for multi-query generation
            middleware=self.middleware,  # Lightweight middleware stack
            checkpointer=self.memory,  # Enable conversation persistence
        )
        logger.info("ArchiveSearchAgent initialization complete - agent will generate queries intelligently")
    
    def search(
        self, 
        user_query: str, 
        thread_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Synchronous search for archives based on user query.
        
        The agent will:
        1. Analyze the user's request
        2. Generate 3-5 diverse query variations  
        3. Call search_archives_db with the list of queries
        4. Receive deduplicated results
        5. Validate results against user intent
        6. Refine and search again if needed
        
        Args:
            user_query: The user's search request (e.g., "I want batik")
            thread_id: Optional conversation thread ID for persistence (default: "default")
            
        Returns:
            Dictionary containing:
            - message: The agent's response message
            - archives: List of matching archive records with full details
            - metadata: Search metadata (queries made, total archives, etc.)
        """
        if thread_id is None:
            thread_id = "default"
            
        logger.info(f"Starting synchronous search for query: '{user_query}' (thread_id={thread_id})")
        
        try:
            # Configure the agent with thread ID for persistence
            config = {"configurable": {"thread_id": thread_id}}
            
            # Invoke the agent with the user's query
            logger.debug("Invoking agent - it will generate diverse queries and validate results")
            result = self.agent.invoke(
                {
                    "messages": [{"role": "user", "content": user_query}],
                    "search_queries_made": [],  # Initialize state
                    "archives_found": {},
                    "conversation_turn": 0,
                    "tool_call_count": 0,
                },
                config=config
            )
            
            # Extract the agent's final response
            messages = result.get("messages", [])
            agent_message = messages[-1] if messages else None
            agent_response = agent_message.content if hasattr(agent_message, 'content') else str(agent_message)
            logger.debug(f"Agent response: {agent_response[:100]}...")
            
            # Extract archives from state (deduplicated by middleware)
            archives_found = result.get("archives_found", {})
            archives_list = list(archives_found.values())
            
            # Get search metadata
            search_queries = result.get("search_queries_made", [])
            tool_calls = result.get("tool_call_count", 0)
            
            logger.info(f"Search completed: {len(archives_list)} archives, {len(search_queries)} queries generated by agent, {tool_calls} tool calls")
            
            return {
                "message": agent_response,
                "archives": archives_list,
                "metadata": {
                    "queries_made": search_queries,
                    "total_archives": len(archives_list),
                    "tool_calls": tool_calls,
                    "conversation_turn": result.get("conversation_turn", 0)
                }
            }
        except Exception as e:
            logger.error(f"Error during synchronous search: {str(e)}", exc_info=True)
            raise
    
    async def search_stream(
        self, 
        user_query: str,
        thread_id: Optional[str] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Asynchronous streaming search for archives.
        
        This method streams the agent's progress as it:
        1. Generates diverse query variations
        2. Searches the database
        3. Validates and refines results
        
        This allows for real-time updates to the frontend showing the agent's reasoning.
        
        Args:
            user_query: The user's search request
            thread_id: Optional conversation thread ID for persistence (default: "default")
            
        Yields:
            Dictionaries containing updates:
            - type: "message" | "archives" | "tool_call" | "final"
            - content: The content of the update
            - metadata: Additional context (optional)
        """
        if thread_id is None:
            thread_id = "default"
            
        logger.info(f"Starting streaming search for query: '{user_query}' (thread_id={thread_id})")
        
        try:
            # Configure the agent with thread ID for persistence
            config = {"configurable": {"thread_id": thread_id}}
            
            message_count = 0
            final_state = None
            
            async for event in self.agent.astream(
                {
                    "messages": [{"role": "user", "content": user_query}],
                    "search_queries_made": [],
                    "archives_found": {},
                    "conversation_turn": 0,
                    "tool_call_count": 0,
                },
                config=config,
                stream_mode="values"
            ):
                message_count += 1
                final_state = event  # Keep track of latest state
                
                messages = event.get("messages", [])
                if messages:
                    last_message = messages[-1]
                    logger.debug(f"Stream event {message_count}: Received message")
                    
                    # Yield AI message content
                    if hasattr(last_message, 'content') and last_message.content:
                        logger.debug(f"Yielding message content: {last_message.content[:50]}...")
                        yield {
                            "type": "message",
                            "content": last_message.content
                        }
                    
                    # Yield tool call notifications (shows query generation)
                    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                        for tool_call in last_message.tool_calls:
                            logger.debug(f"Yielding tool call: {tool_call.get('name')} with args: {tool_call.get('args', {}).get('queries', 'N/A')}")
                            yield {
                                "type": "tool_call",
                                "content": {
                                    "tool": tool_call.get("name"),
                                    "args": tool_call.get("args")
                                }
                            }
                    
                    # Yield archives from tool messages
                    if hasattr(last_message, 'artifact') and last_message.artifact:
                        logger.debug(f"Yielding archives artifact with {len(last_message.artifact)} items")
                        yield {
                            "type": "archives",
                            "content": last_message.artifact
                        }
            
            # Extract final state and send comprehensive summary
            if final_state:
                archives_found = final_state.get("archives_found", {})
                archives_list = list(archives_found.values())
                search_queries = final_state.get("search_queries_made", [])
                tool_calls = final_state.get("tool_call_count", 0)
                
                logger.info(f"Streaming completed: {message_count} events, {len(archives_list)} archives, agent-generated queries")
                
                yield {
                    "type": "final",
                    "archives": archives_list,
                    "metadata": {
                        "queries_made": search_queries,
                        "total_archives": len(archives_list),
                        "tool_calls": tool_calls,
                        "conversation_turn": final_state.get("conversation_turn", 0)
                    }
                }
        except Exception as e:
            logger.error(f"Error during streaming search: {str(e)}", exc_info=True)
            raise


# Create a singleton instance
_agent_instance = None


def get_archive_search_agent() -> ArchiveSearchAgent:
    """Get or create the archive search agent singleton."""
    global _agent_instance
    if _agent_instance is None:
        logger.info("Creating new ArchiveSearchAgent singleton instance")
        _agent_instance = ArchiveSearchAgent()
    else:
        logger.debug("Returning existing ArchiveSearchAgent singleton instance")
    return _agent_instance
