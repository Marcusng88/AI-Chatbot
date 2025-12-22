"""LEGACY: Middleware for AI Search Agent - NOT CURRENTLY USED.

This file contains middleware implementations that were used with the old agent.py.
The new agent_v2.py uses a simpler approach without middleware.

Kept for reference only. Can be deleted if not needed.
"""

import logging
import time
from typing import Any, Dict, List, Callable, Optional
from typing_extensions import NotRequired
from langchain.agents.middleware import AgentState, AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse, ToolCallRequest, ToolMessage
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, trim_messages
from app.core.config import settings

logger = logging.getLogger(__name__)


# ============================================================================
# CUSTOM STATE SCHEMA
# ============================================================================

class ArchiveSearchState(AgentState):
    """
    Custom state for archive search agent.
    Extends base AgentState with search-specific tracking.
    """
    # Track all search queries made in this conversation
    search_queries_made: NotRequired[List[str]]
    
    # Track all unique archives found (by ID)
    archives_found: NotRequired[Dict[str, Dict[str, Any]]]
    
    # User context and preferences
    user_context: NotRequired[Dict[str, Any]]
    
    # Conversation turn counter
    conversation_turn: NotRequired[int]
    
    # Total tool calls made
    tool_call_count: NotRequired[int]


# ============================================================================
# ERROR HANDLING MIDDLEWARE
# ============================================================================

class ErrorHandlingMiddleware(AgentMiddleware[ArchiveSearchState]):
    """
    Handles tool call errors with retry logic and fallbacks.
    """
    
    state_schema = ArchiveSearchState
    
    def __init__(self, max_retries: int = 3, backoff_factor: float = 2.0):
        """
        Initialize error handling middleware.
        
        Args:
            max_retries: Maximum number of retry attempts
            backoff_factor: Exponential backoff multiplier
        """
        super().__init__()
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        logger.info(f"Initialized ErrorHandlingMiddleware (retries={max_retries}, backoff={backoff_factor})")
    
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage],
    ) -> ToolMessage:
        """
        Execute tool with retry logic and error handling.
        """
        tool_name = request.tool_call.get("name", "unknown")
        logger.debug(f"ErrorHandlingMiddleware wrapping: {tool_name}")
        
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                if attempt > 0:
                    # Exponential backoff
                    sleep_time = self.backoff_factor ** attempt
                    logger.info(f"Retry attempt {attempt + 1}/{self.max_retries} for {tool_name} after {sleep_time}s")
                    time.sleep(sleep_time)
                
                # Execute the tool
                result = handler(request)
                
                # Validate result
                if self._is_valid_result(result):
                    if attempt > 0:
                        logger.info(f"Tool {tool_name} succeeded on attempt {attempt + 1}")
                    return result
                else:
                    logger.warning(f"Tool {tool_name} returned invalid result on attempt {attempt + 1}")
                    last_error = "Invalid result returned"
                    
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Tool {tool_name} failed on attempt {attempt + 1}: {last_error}")
                
                # Don't retry on certain errors
                if "authentication" in last_error.lower() or "permission" in last_error.lower():
                    logger.error(f"Non-retryable error for {tool_name}: {last_error}")
                    break
        
        # All retries failed - return graceful error message
        logger.error(f"Tool {tool_name} failed after {self.max_retries} attempts. Last error: {last_error}")
        
        return ToolMessage(
            content=f"Search temporarily unavailable. Please try again. (Error: {last_error})",
            artifact=[],
            tool_call_id=request.tool_call.get("id", "")
        )
    
    def _is_valid_result(self, result: ToolMessage) -> bool:
        """Check if tool result is valid."""
        if not result:
            return False
        
        # Check for error messages in content
        if hasattr(result, 'content'):
            content = result.content.lower()
            if "error" in content and "found 0" not in content:
                return False
        
        return True


# ============================================================================
# DYNAMIC PROMPT MIDDLEWARE
# ============================================================================

class DynamicPromptMiddleware(AgentMiddleware[ArchiveSearchState]):
    """
    Injects dynamic context and trims messages before each LLM call.
    """
    
    state_schema = ArchiveSearchState
    
    def __init__(self, max_tokens: int = 4000):
        """
        Initialize dynamic prompt middleware.
        
        Args:
            max_tokens: Maximum tokens to keep in conversation history
        """
        super().__init__()
        self.max_tokens = max_tokens
        logger.info(f"Initialized DynamicPromptMiddleware (max_tokens={max_tokens})")
    
    def before_model(
        self,
        state: ArchiveSearchState,
        runtime: Any
    ) -> Optional[Dict[str, Any]]:
        """
        Process state before calling the model.
        """
        logger.debug("DynamicPromptMiddleware: Processing state before model call")
        
        messages = state.get("messages", [])
        
        # Trim messages to prevent context overflow
        trimmed_messages = trim_messages(
            messages,
            max_tokens=self.max_tokens,
            strategy="last",
            token_counter=len,  # Simple token counter
        )
        
        if len(trimmed_messages) < len(messages):
            logger.info(f"Trimmed messages from {len(messages)} to {len(trimmed_messages)}")
        
        # Inject context about previous searches
        search_queries = state.get("search_queries_made", [])
        archives_found = state.get("archives_found", {})
        
        if search_queries:
            context_message = SystemMessage(
                content=f"Context: You have already searched {len(search_queries)} times. "
                        f"Found {len(archives_found)} unique archives so far. "
                        f"Previous queries: {', '.join(search_queries[-3:])}"
            )
            trimmed_messages.insert(-1, context_message)  # Insert before last user message
            logger.debug(f"Injected context about {len(search_queries)} previous searches")
        
        return {"messages": trimmed_messages}


# ============================================================================
# STATE TRACKING MIDDLEWARE
# ============================================================================

class StateTrackingMiddleware(AgentMiddleware[ArchiveSearchState]):
    """
    Tracks search state and validates response quality.
    """
    
    state_schema = ArchiveSearchState
    
    def __init__(self, min_similarity_threshold: float = 0.3):
        """
        Initialize state tracking middleware.
        
        Args:
            min_similarity_threshold: Minimum similarity score to consider valid
        """
        super().__init__()
        self.min_similarity_threshold = min_similarity_threshold
        logger.info(f"Initialized StateTrackingMiddleware (min_similarity={min_similarity_threshold})")
    
    def after_model(
        self,
        state: ArchiveSearchState,
        runtime: Any
    ) -> Optional[Dict[str, Any]]:
        """
        Update state after model generates response.
        """
        logger.debug("StateTrackingMiddleware: Processing state after model call")
        
        messages = state.get("messages", [])
        if not messages:
            return None
        
        # Increment conversation turn
        current_turn = state.get("conversation_turn", 0)
        
        # Extract archives from tool messages
        archives_found = state.get("archives_found", {})
        
        for message in messages:
            if hasattr(message, 'artifact') and message.artifact:
                for archive in message.artifact:
                    archive_id = archive.get('id')
                    similarity = archive.get('similarity', 0)
                    
                    # Only track high-quality results
                    if archive_id and similarity >= self.min_similarity_threshold:
                        if archive_id not in archives_found:
                            archives_found[archive_id] = archive
                            logger.debug(f"Tracked new archive: {archive.get('title')} (similarity: {similarity:.2f})")
        
        # Increment tool call counter
        tool_call_count = state.get("tool_call_count", 0)
        last_message = messages[-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            tool_call_count += len(last_message.tool_calls)
        
        logger.info(f"State update: Turn {current_turn + 1}, {len(archives_found)} archives tracked, {tool_call_count} tool calls")
        
        return {
            "conversation_turn": current_turn + 1,
            "archives_found": archives_found,
            "tool_call_count": tool_call_count
        }

class ErrorHandlingMiddleware(AgentMiddleware[ArchiveSearchState]):
    """
    Handles tool call errors with retry logic and fallbacks.
    """
    
    state_schema = ArchiveSearchState
    
    def __init__(self, max_retries: int = 3, backoff_factor: float = 2.0):
        """
        Initialize error handling middleware.
        
        Args:
            max_retries: Maximum number of retry attempts
            backoff_factor: Exponential backoff multiplier
        """
        super().__init__()
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        logger.info(f"Initialized ErrorHandlingMiddleware (retries={max_retries}, backoff={backoff_factor})")
    
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage],
    ) -> ToolMessage:
        """
        Execute tool with retry logic and error handling.
        """
        tool_name = request.tool_call.get("name", "unknown")
        logger.debug(f"ErrorHandlingMiddleware wrapping: {tool_name}")
        
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                if attempt > 0:
                    # Exponential backoff
                    sleep_time = self.backoff_factor ** attempt
                    logger.info(f"Retry attempt {attempt + 1}/{self.max_retries} for {tool_name} after {sleep_time}s")
                    time.sleep(sleep_time)
                
                # Execute the tool
                result = handler(request)
                
                # Validate result
                if self._is_valid_result(result):
                    if attempt > 0:
                        logger.info(f"Tool {tool_name} succeeded on attempt {attempt + 1}")
                    return result
                else:
                    logger.warning(f"Tool {tool_name} returned invalid result on attempt {attempt + 1}")
                    last_error = "Invalid result returned"
                    
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Tool {tool_name} failed on attempt {attempt + 1}: {last_error}")
                
                # Don't retry on certain errors
                if "authentication" in last_error.lower() or "permission" in last_error.lower():
                    logger.error(f"Non-retryable error for {tool_name}: {last_error}")
                    break
        
        # All retries failed - return graceful error message
        logger.error(f"Tool {tool_name} failed after {self.max_retries} attempts. Last error: {last_error}")
        
        return ToolMessage(
            content=f"Search temporarily unavailable. Please try again. (Error: {last_error})",
            artifact=[],
            tool_call_id=request.tool_call.get("id", "")
        )
    
    def _is_valid_result(self, result: ToolMessage) -> bool:
        """Check if tool result is valid."""
        if not result:
            return False
        
        # Check for error messages in content
        if hasattr(result, 'content'):
            content = result.content.lower()
            if "error" in content and "found 0" not in content:
                return False
        
        return True


# ============================================================================
# DYNAMIC PROMPT MIDDLEWARE
# ============================================================================

class DynamicPromptMiddleware(AgentMiddleware[ArchiveSearchState]):
    """
    Injects dynamic context and trims messages before each LLM call.
    """
    
    state_schema = ArchiveSearchState
    
    def __init__(self, max_tokens: int = 4000):
        """
        Initialize dynamic prompt middleware.
        
        Args:
            max_tokens: Maximum tokens to keep in conversation history
        """
        super().__init__()
        self.max_tokens = max_tokens
        logger.info(f"Initialized DynamicPromptMiddleware (max_tokens={max_tokens})")
    
    def before_model(
        self,
        state: ArchiveSearchState,
        runtime: Any
    ) -> Optional[Dict[str, Any]]:
        """
        Process state before calling the model.
        """
        logger.debug("DynamicPromptMiddleware: Processing state before model call")
        
        messages = state.get("messages", [])
        
        # Trim messages to prevent context overflow
        trimmed_messages = trim_messages(
            messages,
            max_tokens=self.max_tokens,
            strategy="last",
            token_counter=len,  # Simple token counter
        )
        
        if len(trimmed_messages) < len(messages):
            logger.info(f"Trimmed messages from {len(messages)} to {len(trimmed_messages)}")
        
        # Inject context about previous searches
        search_queries = state.get("search_queries_made", [])
        archives_found = state.get("archives_found", {})
        
        if search_queries:
            context_message = SystemMessage(
                content=f"Context: You have already searched {len(search_queries)} times. "
                        f"Found {len(archives_found)} unique archives so far. "
                        f"Previous queries: {', '.join(search_queries[-3:])}"
            )
            trimmed_messages.insert(-1, context_message)  # Insert before last user message
            logger.debug(f"Injected context about {len(search_queries)} previous searches")
        
        return {"messages": trimmed_messages}


# ============================================================================
# STATE TRACKING MIDDLEWARE
# ============================================================================

class StateTrackingMiddleware(AgentMiddleware[ArchiveSearchState]):
    """
    Tracks search state and validates response quality.
    """
    
    state_schema = ArchiveSearchState
    
    def __init__(self, min_similarity_threshold: float = 0.3):
        """
        Initialize state tracking middleware.
        
        Args:
            min_similarity_threshold: Minimum similarity score to consider valid
        """
        super().__init__()
        self.min_similarity_threshold = min_similarity_threshold
        logger.info(f"Initialized StateTrackingMiddleware (min_similarity={min_similarity_threshold})")
    
    def after_model(
        self,
        state: ArchiveSearchState,
        runtime: Any
    ) -> Optional[Dict[str, Any]]:
        """
        Update state after model generates response.
        """
        logger.debug("StateTrackingMiddleware: Processing state after model call")
        
        messages = state.get("messages", [])
        if not messages:
            return None
        
        # Increment conversation turn
        current_turn = state.get("conversation_turn", 0)
        
        # Extract archives from tool messages
        archives_found = state.get("archives_found", {})
        
        for message in messages:
            if hasattr(message, 'artifact') and message.artifact:
                for archive in message.artifact:
                    archive_id = archive.get('id')
                    similarity = archive.get('similarity', 0)
                    
                    # Only track high-quality results
                    if archive_id and similarity >= self.min_similarity_threshold:
                        if archive_id not in archives_found:
                            archives_found[archive_id] = archive
                            logger.debug(f"Tracked new archive: {archive.get('title')} (similarity: {similarity:.2f})")
        
        # Increment tool call counter
        tool_call_count = state.get("tool_call_count", 0)
        last_message = messages[-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            tool_call_count += len(last_message.tool_calls)
        
        logger.info(f"State update: Turn {current_turn + 1}, {len(archives_found)} archives tracked, {tool_call_count} tool calls")
        
        return {
            "conversation_turn": current_turn + 1,
            "archives_found": archives_found,
            "tool_call_count": tool_call_count
        }
