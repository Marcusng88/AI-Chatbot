"""
Clean AI Search Endpoint - Structured Responses Only.

Returns ONLY structured archive data, NO text responses.
Perfect for modern chat UX with immediate feedback.
"""

from typing import Dict, Any, List
from datetime import datetime
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import json

from app.services.ai_search.agent_v2 import get_archive_search_agent


router = APIRouter()


class SearchRequest(BaseModel):
    """Request model for AI search."""
    query: str = Field(..., min_length=1, description="Search query")
    thread_id: str | None = Field(None, description="Optional conversation thread ID")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "batik",
                "thread_id": "user-123"
            }
        }


class ArchiveResult(BaseModel):
    """Structured archive result."""
    id: str
    title: str
    description: str | None = None
    media_types: List[str]
    dates: List[str] | None = None
    tags: List[str] | None = None
    file_uris: List[str] | None = None
    storage_paths: List[str] | None = None
    genai_file_ids: List[str] | None = None
    created_at: str
    updated_at: str | None = None
    similarity: float | None = None


class SearchResponse(BaseModel):
    """Clean search response - only structured data."""
    archives: List[ArchiveResult]
    total: int
    query: str
    message: str | None = None  # Only for "no results" case
    
    class Config:
        json_schema_extra = {
            "example": {
                "archives": [
                    {
                        "id": "uuid-here",
                        "title": "Traditional Batik Patterns",
                        "description": "Collection of batik patterns",
                        "media_types": ["image"],
                        "dates": ["2024-01-15T00:00:00Z"],
                        "tags": ["batik", "textile"],
                        "file_uris": ["https://storage.url/file.jpg"],
                        "created_at": "2024-01-15T10:00:00Z",
                        "similarity": 0.85
                    }
                ],
                "total": 1,
                "query": "batik"
            }
        }


@router.post("/ai-search", response_model=SearchResponse)
async def ai_search(request: SearchRequest):
    """
    AI-powered archive search returning ONLY structured results.
    
    NO text responses - only structured archive data.
    If no results found, returns empty array with message.
    
    **Clean Response Format:**
    - archives: Array of matching archives
    - total: Count of results
    - query: Echo of user query (for confirmation)
    - message: Only present if no results ("No matching archives found")
    
    **Features:**
    - Automatic multi-query generation (3-5 diverse variations)
    - Vector similarity search across all queries
    - Deduplication by archive ID
    - Thread-based conversation persistence
    
    **Example:**
    ```
    POST /ai-search
    {"query": "batik", "thread_id": "user-123"}
    
    Response:
    {
      "archives": [...],
      "total": 5,
      "query": "batik"
    }
    ```
    """
    try:
        # Get agent
        agent = get_archive_search_agent()
        
        # Perform search
        result = agent.search(
            user_query=request.query,
            thread_id=request.thread_id
        )
        
        # Build response
        archives = result.get("archives", [])
        total = result.get("total", 0)
        
        response_data = {
            "archives": archives,
            "total": total,
            "query": request.query
        }
        
        # Only add message if no results
        if total == 0:
            response_data["message"] = "No matching archives found"
        
        return SearchResponse(**response_data)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


@router.post("/ai-search/stream")
async def ai_search_stream(request: SearchRequest):
    """
    Streaming AI search with progressive updates.
    
    Returns Server-Sent Events (SSE) in clean format:
    
    **Event Types:**
    - `query_received`: Immediate acknowledgment
      ```json
      {"type": "query_received", "query": "batik", "timestamp": "..."}
      ```
    
    - `searching`: Agent is generating queries
      ```json
      {"type": "searching", "query": "batik"}
      ```
    
    - `results`: Progressive results (sent as archives are found)
      ```json
      {"type": "results", "archives": [...], "total": 5}
      ```
    
    - `done`: Final completion with all results
      ```json
      {"type": "done", "archives": [...], "total": 5, "message": "..."}
      ```
    
    - `error`: Error occurred
      ```json
      {"type": "error", "message": "..."}
      ```
    
    **Frontend should:**
    1. Listen for `query_received` → clear input, show user message
    2. Listen for `searching` → show loading indicator
    3. Listen for `results` → progressively display archives
    4. Listen for `done` → finalize UI, hide loading
    """
    try:
        # Get agent
        agent = get_archive_search_agent()
        
        async def event_generator():
            """Generate SSE events."""
            try:
                # IMMEDIATE: Acknowledge query received
                yield f"data: {json.dumps({
                    'type': 'query_received',
                    'query': request.query,
                    'thread_id': request.thread_id,
                    'timestamp': datetime.now().isoformat()
                })}\\n\\n"
                
                # Stream agent results
                all_archives: List[Dict[str, Any]] = []
                
                async for update in agent.search_stream(
                    user_query=request.query,
                    thread_id=request.thread_id
                ):
                    # Forward all events
                    yield f"data: {json.dumps(update)}\\n\\n"
                    
                    # Track archives for final message
                    if update.get("type") == "done":
                        all_archives = update.get("archives", [])
                
                # Send completion with message
                final_event = {
                    "type": "complete",
                    "total": len(all_archives),
                    "query": request.query
                }
                
                if len(all_archives) == 0:
                    final_event["message"] = "No matching archives found"
                
                yield f"data: {json.dumps(final_event)}\\n\\n"
                
            except Exception as e:
                # Error event
                yield f"data: {json.dumps({
                    'type': 'error',
                    'message': str(e)
                })}\\n\\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
                "Access-Control-Allow-Origin": "*"
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Streaming search failed: {str(e)}"
        )
