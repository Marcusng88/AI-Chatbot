"""
Heritage Archive Search Agent - LangChain 1.0 with Chain-of-Thought.

This agent implements intent classification, query generation, and chain-of-thought
reasoning to automatically try alternative search strategies when needed.
"""

import logging
from typing import List, Dict, Any, AsyncIterator, Optional
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import AIMessage

from app.core.config import settings
from app.services.ai_search.tools import search_archives_db, read_archives_data

logger = logging.getLogger(__name__)



# Redesigned system prompt for enhanced intelligence and autonomous multi-tool usage
SEARCH_AGENT_PROMPT = """
<role>
You are an intelligent Malaysian heritage archive search assistant. Your job is to help users find cultural heritage materials by autonomously generating comprehensive queries and intelligently combining search strategies.
Current date: {today}
</role>

<database_schema>
You have access to a heritage database with these fields:
- **title** (TEXT): Archive title
- **description** (TEXT): Detailed description
- **summary** (TEXT): Brief summary
- **tags[]** (TEXT ARRAY): Keywords like "batik", "kelantan", "traditional", "craft", "temple"
- **media_types[]** (TEXT ARRAY): "image", "video", "audio", "document"
- **dates[]** (TIMESTAMPTZ ARRAY): Historical content dates
- **created_at** (TIMESTAMPTZ): Upload date
- **storage_paths[]** (TEXT ARRAY): File paths
</database_schema>

<intent_classification>
STEP 1: Classify user intent FIRST

| Intent | Examples | Action |
|--------|----------|--------|
| HERITAGE_SEARCH | "batik", "Penang temples", "traditional crafts", "show videos" | Use tools intelligently |
| VAGUE_REQUEST | "show me something", "I want stuff", "anything interesting" | Ask for specifics: type? region? media? |
| GREETING | "hello", "hi", "thanks", "how are you" | Brief friendly response, NO tools |
| UNRELATED | "weather", "news", "math problem" | Politely decline, NO tools |

</intent_classification>

<response_guidelines>
**VAGUE_REQUEST** → Ask clarifying questions:
"What type of heritage materials are you interested in? For example:
- Type: batik, crafts, architecture, ceremonies
- Region: Penang, Kelantan, Sabah, Melaka
- Media: images, videos, documents"

**GREETING** → "Hello! I can help you find Malaysian heritage materials. What would you like to search for?"

**UNRELATED** → "I can only search heritage archives. What cultural materials interest you?"

**HERITAGE_SEARCH** → Proceed to intelligent search strategy below.
</response_guidelines>

<intelligent_search_strategy>
For HERITAGE_SEARCH, follow this autonomous multi-step approach:

**STEP A: ANALYZE & GENERATE COMPREHENSIVE QUERY**
- Extract user intent: What specifically are they looking for?
- Identify: topic, region, type, media format
- Generate ONE comprehensive single-sentence query that captures full intent
  
Examples:
- User: "batik" → Query: "traditional Malaysian batik textiles wax-resist dyed fabric patterns sarong"
- User: "Penang temples" → Query: "Penang heritage temples religious architecture historical buildings worship sites"
- User: "old photos Melaka" → Query: "historical photographs vintage images Melaka heritage documentation colonial era"

**STEP B: SEMANTIC SEARCH (PRIMARY)**
1. Use `search_archives_db` with comprehensive query
   - Default threshold: 0.7 (high precision)
   - Default limit: 10
2. If results found → Return structured data immediately
3. If ZERO results → Proceed to Step C

**STEP C: AUTONOMOUS MULTI-STRATEGY FALLBACK**
When semantic search returns nothing, intelligently try multiple approaches:

1. **Metadata Tag Filtering** (if user mentioned specific terms)
   - Extract key terms: regions (kelantan, sabah), types (batik, craft), categories
   - Try: `read_archives_data(filter_by="tag", filter_value=<term>)`
   
2. **Media Type Filtering** (if user wants specific format)
   - If query mentions: "videos" → filter_by="media_type", filter_value="video"
   - If query mentions: "photos/images" → filter_by="media_type", filter_value="image"
   - If query mentions: "documents" → filter_by="media_type", filter_value="document"

3. **Title Search** (broader text matching)
   - Try: `read_archives_data(filter_by="title", filter_value=<main_term>)`

4. **Combined Filters** (smart combinations)
   - Example: "Kelantan videos" → media_type="video" + then filter results by "kelantan" tag
   - Use tools MULTIPLE TIMES to refine results

5. **Relaxed Semantic Search** (last resort)
   - Try `search_archives_db` again with threshold=0.5 or 0.4
   - Broader query variations

**STEP D: RELEVANCE VALIDATION (CRITICAL)**
Before returning ANY results:
- Review: Does title/description/tags actually match user request?
- Filter out: Irrelevant matches (e.g., user asked "Sabah" but result is "Johor")
- Only return: Genuinely relevant materials

**STEP E: RETURN RESULTS**
- If found → Return structured data ONLY (no verbose explanations)
- If nothing → Simple message: "No archives found matching your query."
</intelligent_search_strategy>

<tool_usage_rules>
**search_archives_db** - Semantic AI search
- Use for: Descriptive queries, concept matching
- Input: Single comprehensive query sentence
- Can be used MULTIPLE TIMES with different queries/thresholds
- Default: threshold=0.7, match_count=10

**read_archives_data** - Metadata filtering
- Use for: Specific filtering (tags, media_type, title, dates)
- Can be used MULTIPLE TIMES with different filters
- Filters: filter_by="tag"|"media_type"|"title", filter_value=<value>
- Date filtering: date_after, date_before (for upload dates)
- Smart combinations: Try multiple filter strategies autonomously

**Autonomous Multi-Tool Usage:**
You can and SHOULD use tools multiple times in creative combinations:
1. Try semantic search → if fails
2. Try tag filter → if partial results
3. Try title filter with broader term → if fails
4. Try relaxed semantic search → return best results

Be intelligent and persistent in finding relevant archives.
</tool_usage_rules>

<examples>
**Example 1: Specific Query**
USER: "batik from Kelantan"
→ Generate query: "traditional Kelantan batik textiles wax-resist fabric patterns heritage"
→ search_archives_db(query=..., threshold=0.7)
→ Return results

**Example 2: Zero Results Fallback**
USER: "Sabah traditional culture"
→ search_archives_db(query="Sabah traditional cultural heritage indigenous customs") → 0 results
→ read_archives_data(filter_by="tag", filter_value="sabah") → Check relevance
→ If still nothing: read_archives_data(filter_by="title", filter_value="sabah")
→ Return relevant findings or "No archives found"

**Example 3: Media-Specific**
USER: "show me all videos"
→ read_archives_data(filter_by="media_type", filter_value="video", limit=20)
→ Return results

**Example 4: Vague Query**
USER: "show me something interesting"
→ VAGUE_REQUEST
→ "What type of heritage materials interest you? (e.g., batik, temples, traditional crafts, specific regions)"

**Example 5: Combined Strategy**
USER: "Penang heritage videos"
→ search_archives_db(query="Penang heritage historical cultural videos documentation") → 0 results
→ read_archives_data(filter_by="media_type", filter_value="video") → Get videos
→ Filter for "Penang" in tags/title → Return relevant videos
</examples>

<critical_rules>
✓ ALWAYS classify intent first
✓ Generate comprehensive single-sentence queries for semantic search
✓ Use tools MULTIPLE TIMES if needed (autonomous multi-strategy)
✓ Validate relevance before returning results
✓ Return structured data only (no verbose explanations when results found)
✓ Be persistent: try multiple approaches before giving up
✓ Understand database schema and use intelligent filter combinations

✗ NEVER call tools for greetings/unrelated queries
✗ NEVER return irrelevant results
✗ NEVER show verbose explanations when returning archives
✗ NEVER give up after one tool call - be autonomous and intelligent
</critical_rules>
"""



class ArchiveSearchAgentV2:
    """
    Heritage archive search agent with intent classification and chain-of-thought reasoning.
    
    Features:
    - Intent classification (HERITAGE_SEARCH, UNCLEAR, UNRELATED, GREETING)
    - Single focused query generation
    - Chain-of-thought reasoning for automatic fallback strategies
    - Multi-tool orchestration (semantic search + metadata browsing)
    - Structured output for search results
    - Text responses for non-search intents
    
    Available Tools:
    1. search_archives_db: Semantic vector search for finding similar archives
    2. read_archives_data: Read-only metadata filtering and browsing (no write operations)
    """
    
    def __init__(self):
        logger.info("Initializing ArchiveSearchAgentV2 with chain-of-thought reasoning (LangChain 1.0)")
        
        # Initialize Gemini model
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=settings.GOOGLE_GENAI_API_KEY,
            temperature=0.2,  # Lower for focused query generation
        )
        
        # Tools: search_archives_db (vector search) + read_archives_data (metadata filtering)
        self.tools = [search_archives_db, read_archives_data]
        logger.info(f"Configured with {len(self.tools)} tool(s): {[tool.name for tool in self.tools]}")
        
        # Memory for conversation persistence
        self.memory = InMemorySaver()
        
        # Get current date/time for the system prompt
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create agent with chain-of-thought reasoning
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=SEARCH_AGENT_PROMPT.format(today=today),
            # checkpointer=self.memory,
        )
        logger.info("ArchiveSearchAgentV2 initialized with chain-of-thought multi-tool reasoning")
    
    def search(
        self, 
        user_query: str, 
        thread_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Synchronous search returning structured archive data or text message.
        
        Args:
            user_query: User's search query
            thread_id: Optional conversation thread ID
            
        Returns:
            For HERITAGE_SEARCH intent:
            {
                "archives": [...],  # Structured archive list
                "total": int,        # Total count
                "query": str         # Echo of user query
            }
            
            For non-search intents (UNCLEAR, UNRELATED, GREETING):
            {
                "message": str,      # Text response
                "archives": [],      # Empty
                "total": 0,          # Zero
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
            
            # Check if agent returned text message (non-search intent)
            text_message = self._extract_text_message(result)
            if text_message:
                logger.info(f"Non-search intent detected: {text_message[:50]}...")
                return {
                    "message": text_message,
                    "archives": [],
                    "total": 0,
                    "query": user_query
                }
            
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
        Streaming search with progressive updates and intent detection.
        
        Yields:
            - {"type": "searching", "query": str}  # Agent is processing
            - {"type": "results", "archives": [...], "total": int}  # Results found
            - {"type": "message", "message": str}  # Text response (non-search)
            - {"type": "done", "archives": [...], "total": int}  # Completion signal
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
            text_message: Optional[str] = None
            
            # Stream agent execution
            async for event in self.agent.astream(
                {"messages": [{"role": "user", "content": user_query}]},
                config=config,
                stream_mode="values"
            ):
                # Check for text message (non-search intent)
                if not text_message:
                    msg = self._extract_text_message(event)
                    if msg:
                        text_message = msg
                        yield {
                            "type": "message",
                            "message": text_message
                        }
                        continue
                
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
            
            logger.info(f"Stream complete: {len(all_archives)} archives, text_message={bool(text_message)}")
            
        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            yield {
                "type": "error",
                "message": str(e)
            }
    
    def _extract_text_message(self, result: Dict[str, Any]) -> Optional[str]:
        """
        Extract text message from agent response (for non-search intents).
        
        Returns text message if agent responded without calling search tool,
        None otherwise (indicating HERITAGE_SEARCH intent).
        """
        messages = result.get("messages", [])
        
        # Check if last message is from AI and contains no tool calls
        if messages:
            last_msg = messages[-1]
            if isinstance(last_msg, AIMessage):
                # If AI message has no tool calls and no tool artifacts in history,
                # it's a text response (non-search intent)
                has_tool_calls = hasattr(last_msg, "tool_calls") and last_msg.tool_calls
                has_tool_artifacts = any(
                    hasattr(msg, "artifact") and msg.artifact
                    for msg in messages
                )
                
                if not has_tool_calls and not has_tool_artifacts:
                    content = last_msg.content
                    
                    # Handle multimodal content format from Gemini
                    # Content can be a list of dicts like [{'type': 'text', 'text': '...'}]
                    if isinstance(content, list):
                        text_parts = []
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                text_parts.append(part.get("text", ""))
                            elif isinstance(part, str):
                                text_parts.append(part)
                        content = " ".join(text_parts) if text_parts else ""
                    
                    # Filter out tool code that was incorrectly returned as text
                    # This happens when the model outputs code instead of calling tools
                    tool_code_patterns = [
                        "tool_code",
                        "default_api.",
                        "search_archives_db(",
                        "read_archives_data(",
                        "print(default_api",
                    ]
                    
                    # If content contains tool code patterns, don't return it as a message
                    if any(pattern in content for pattern in tool_code_patterns):
                        logger.warning(f"Tool code detected in content, filtering out: {content}")
                        return None
                    
                    # Pure text response
                    return content
        
        return None
    
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
