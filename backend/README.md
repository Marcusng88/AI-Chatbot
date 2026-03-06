# BWM Heritage Archive — Backend

FastAPI backend for the Badan Warisan Malaysia AI chatbot. Handles archive CRUD, AI-powered metadata generation, and natural-language semantic search over the heritage database.

## Project Structure

```
backend/
├── main.py                        # Entry point — starts uvicorn
├── Dockerfile                     # Docker image for Render deployment
├── render.yaml                    # Render cloud deployment config
├── requirements.txt
└── app/
    ├── main.py                    # FastAPI app factory (CORS, router mounting)
    ├── api/v1/
    │   ├── api.py                 # Aggregates all routers
    │   └── endpoints/
    │       ├── archives.py        # Archive CRUD + file upload
    │       ├── ai_search_v2.py    # AI search — sync & SSE streaming
    │       └── items.py / users.py
    ├── core/
    │   ├── config.py              # Pydantic settings (reads .env)
    │   ├── security.py            # JWT helpers
    │   └── supabase.py            # Supabase client singleton
    ├── schemas/
    │   └── archive.py             # ArchiveCreate / ArchiveResponse / ArchiveUpdate
    ├── services/
    │   ├── archive_service.py     # Upload to Supabase Storage, Gemini metadata, embeddings
    │   └── ai_search/
    │       ├── agent_v2.py        # LangGraph ReAct agent (intent classify + search)
    │       ├── tools.py           # search_archives_db & read_archives_data tools
    │       ├── middleware.py      # Search refinement / retry middleware
    │       └── prompt.py          # System prompt constants
    └── utils/helpers.py
```

---

## Setup

### 1. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create `backend/.env`:

```env
GOOGLE_GENAI_API_KEY=your_google_ai_api_key

SUPABASE_URL=https://<project-ref>.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
```

### 4. Run the server

```bash
python main.py
# or
uvicorn app.main:app --reload
```

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Health check: http://localhost:8000/health

---

## API Endpoints

### Archives — `/api/v1/archives`

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/archives` | List all archives (newest first) |
| `POST` | `/archives` | Upload files + create archive (`multipart/form-data`) |
| `GET` | `/archives/{id}` | Get a single archive |
| `PUT` | `/archives/{id}` | Update title, tags, description, dates |
| `DELETE` | `/archives/{id}` | Delete archive record + Supabase Storage files |
| `POST` | `/archives/suggest-metadata` | AI-generated title, tags & description from uploaded files |

#### Upload fields (multipart/form-data)

| Field | Type | Required |
|-------|------|----------|
| `files` | `File[]` | Yes |
| `title` | `string` | Yes |
| `media_types` | `string[]` | Yes (`image`\|`video`\|`audio`\|`document`) |
| `tags` | `string[]` | No |
| `description` | `string` | No |
| `dates` | `string[]` | No (ISO 8601) |

### AI Search — `/api/v1`

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/ai-search` | Synchronous search, returns `SearchResponse` |
| `POST` | `/ai-search/stream` | SSE streaming search |

#### POST `/ai-search`

```json
// Request
{ "query": "traditional batik from Kelantan", "thread_id": "user-123" }

// Response — heritage results
{
  "response_type": "results",
  "archives": [{ "id": "...", "title": "...", "similarity": 0.87 }],
  "total": 5,
  "query": "traditional batik from Kelantan",
  "message": null
}

// Response — greeting / unrelated / vague
{
  "response_type": "message",
  "archives": [],
  "total": 0,
  "query": "hello",
  "message": "Hello! I can help you find Malaysian heritage materials."
}
```

---

## AI Search Agent

The agent (`services/ai_search/agent_v2.py`) is a **LangGraph ReAct agent** using `gemini-2.5-flash-lite`.

**Tools:**
- `search_archives_db(query, match_threshold=0.7, match_count=10)` — pgvector cosine similarity search via Supabase RPC `match_archives`
- `read_archives_data(filter_by, filter_value)` — metadata filter fallback (tag / media_type / title)

**Intent categories:** `HERITAGE_SEARCH` | `VAGUE_REQUEST` | `GREETING` | `UNRELATED`

**Search flow:**
1. Classify intent — non-heritage queries return a text message (no DB query)
2. Generate a comprehensive semantic query from user input
3. Run `search_archives_db` (threshold 0.7)
4. If zero results → autonomous fallback: tag filter → media-type filter → title search

Conversation memory is maintained per `thread_id` using LangGraph `InMemorySaver`.

---

## Key Dependencies

| Package | Purpose |
|---|---|
| `fastapi` | Web framework |
| `langchain` / `langgraph` | LLM agent orchestration |
| `langchain-google-genai` | Gemini LLM + `text-embedding-004` |
| `google-genai` | Gemini file upload for metadata generation |
| `supabase` | PostgreSQL + Storage client |
| `pydantic-settings` | `.env` config loading |
| `uvicorn` | ASGI server |

---

## Testing

```bash
pytest
```

---

## Deployment (Render)

Deploys as a Docker container (Singapore region) via `render.yaml`.  
Set `GOOGLE_GENAI_API_KEY`, `SUPABASE_URL`, and `SUPABASE_SERVICE_ROLE_KEY` in the Render service environment variables.

