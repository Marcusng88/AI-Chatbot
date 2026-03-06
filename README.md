# Badan Warisan Malaysia — AI Heritage Archive Chatbot

An AI-powered digital archive management system built for **Badan Warisan Negara (BWM)**, Malaysia's national heritage organization. The system enables curators to upload, manage, and intelligently search thousands of cultural heritage materials (photos, videos, audio, documents) through a natural-language chatbot interface.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Environment Variables](#environment-variables)
- [API Reference](#api-reference)
- [Database Schema](#database-schema)
- [AI Search Agent](#ai-search-agent)
- [Deployment](#deployment)

---

## Overview

**Problem**: With thousands of heritage items in the database, browsing is impractical for curators.

**Solution**: A search-first interface where an AI agent interprets natural-language queries, classifies intent, generates semantic embeddings, and returns the most relevant heritage archive items — with intelligent fallback strategies when semantic search yields no results.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend (Vercel)                      │
│          React + TypeScript + Vite + Tailwind CSS           │
│                                                             │
│  Sidebar → ChatPanelV2 ──────► SearchResultsPanel          │
│            CuratorDashboard                                 │
│            SettingsPanel                                    │
│            AddArchiveModal (file upload)                    │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP / SSE
┌──────────────────────▼──────────────────────────────────────┐
│                     Backend (Render)                        │
│                FastAPI + LangGraph + Python                 │
│                                                             │
│  /api/v1/archives      → Archive CRUD                       │
│  /api/v1/ai-search     → AI Search (sync + SSE stream)      │
│  /api/v1/metadata      → AI Metadata Generation             │
└──────────────────────┬──────────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          │                         │
┌─────────▼──────────┐   ┌─────────▼──────────┐
│   Supabase (DB)    │   │  Google GenAI       │
│  PostgreSQL +      │   │  gemini-2.5-flash   │
│  pgvector          │   │  text-embedding-004 │
│  Storage Bucket    │   │                     │
└────────────────────┘   └────────────────────┘
```

---

## Features

### AI Heritage Search
- Natural-language queries interpreted by a LangGraph agent
- Intent classification: `HERITAGE_SEARCH` | `VAGUE_REQUEST` | `GREETING` | `UNRELATED`
- Semantic vector search using Google `text-embedding-004` embeddings
- Automatic multi-strategy fallback: semantic → tag filter → media-type filter → title search
- Real-time streaming results via Server-Sent Events (SSE)
- Conversation memory per `thread_id` (in-memory checkpointer)

### Archive Management (Curator Tools)
- Upload files (images, videos, audio, documents) directly to Supabase Storage
- AI-generated metadata: title, description, tags, summary using Gemini
- Automatic embedding generation stored in pgvector for semantic search
- Full CRUD: create, read, update, delete archives
- Metadata suggestion endpoint before committing an upload

### Curator Dashboard
- Statistics overview (total items, recent uploads)
- Tag management — most-used tags across the collection
- Activity log tracking curator actions
- Bulk operation, export, and analytics tools

### Settings Panel
- Language & display preferences
- AI model configuration (confidence threshold, auto-tagging)
- Notification controls
- Storage / backup management
- User / curator access control

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18, TypeScript, Vite 7 |
| Styling | Tailwind CSS v4, ShadCN UI (Radix UI), Lucide Icons |
| Animations | Motion (Framer Motion) |
| Backend | FastAPI 0.121, Python 3.11+ |
| AI / LLM | Google Gemini (`gemini-2.5-flash-lite`), LangChain, LangGraph |
| Embeddings | Google `text-embedding-004` (768-dim vectors) |
| Database | Supabase PostgreSQL + pgvector extension |
| File Storage | Supabase Storage (`archive-materials` bucket) |
| Backend Deployment | Render (Docker, Singapore region) |
| Frontend Deployment | Vercel |

---

## Project Structure

```
AI-Chatbot/
├── README.md                  ← You are here
├── database.sql               ← PostgreSQL schema with pgvector setup
│
├── backend/                   ← FastAPI application
│   ├── Dockerfile
│   ├── main.py                ← Entry point
│   ├── requirements.txt
│   ├── render.yaml            ← Render deployment config
│   └── app/
│       ├── main.py            ← FastAPI app factory (CORS, routers)
│       ├── api/v1/
│       │   ├── api.py         ← Router aggregation
│       │   └── endpoints/
│       │       ├── archives.py        ← Archive CRUD + file upload
│       │       ├── ai_search_v2.py    ← AI search (sync + SSE)
│       │       └── ai_search.py       ← Legacy search endpoint
│       ├── core/
│       │   ├── config.py      ← Pydantic settings (reads .env)
│       │   ├── security.py    ← JWT utilities
│       │   └── supabase.py    ← Supabase client factory
│       ├── schemas/
│       │   └── archive.py     ← Pydantic request/response models
│       ├── services/
│       │   ├── archive_service.py     ← Upload, embed, CRUD logic
│       │   └── ai_search/
│       │       ├── agent_v2.py        ← LangGraph search agent
│       │       ├── tools.py           ← search_archives_db tool
│       │       ├── middleware.py      ← Search refinement middleware
│       │       └── prompt.py          ← System prompt constants
│       └── utils/
│           └── helpers.py
│
└── frontend/                  ← React application
    ├── package.json
    ├── vite.config.ts
    ├── vercel.json            ← Vercel deployment config
    └── src/
        ├── App.tsx            ← Root component + routing
        ├── services/
        │   └── api.ts         ← All backend API calls
        └── components/
            ├── Sidebar.tsx
            ├── TopBar.tsx
            ├── ChatPanelV2.tsx        ← AI search chat interface
            ├── SearchResultsPanel.tsx ← Live search results
            ├── CuratorDashboard.tsx   ← Stats, tags, activity log
            ├── SettingsPanel.tsx
            ├── AddArchiveModal.tsx    ← File upload form
            ├── ArchiveCard.tsx
            ├── ArchiveDetailModal.tsx
            ├── ChatMessage.tsx
            ├── ChatFilters.tsx
            ├── QuickSearchButtons.tsx
            └── ui/                   ← ShadCN UI components
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- A Supabase project with the `pgvector` extension enabled
- A Google AI API key (Gemini + Embeddings)

### 1. Clone the Repository

```bash
git clone <repo-url>
cd AI-Chatbot
```

### 2. Backend Setup

```bash
cd backend

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your keys (see Environment Variables below)

# Run the server
python main.py
# → API available at http://localhost:8000
# → Swagger docs at http://localhost:8000/docs
```

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Configure environment
echo "VITE_API_BASE_URL=http://localhost:8000/api/v1" > .env

# Run the development server
npm run dev
# → App available at http://localhost:5173
```

### 4. Database Setup

Run the SQL in `database.sql` in your Supabase SQL editor. This creates:
- The `archives` table with vector embedding column
- GIN indexes on `tags`, `media_types`, and `dates` arrays
- HNSW index on `embedding` for fast cosine-similarity search
- The `match_archives` RPC function used by the AI search tool

---

## Environment Variables

### Backend (`backend/.env`)

```env
# Google GenAI
GOOGLE_GENAI_API_KEY=your_google_ai_api_key

# Supabase
SUPABASE_URL=https://<project-ref>.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
```

### Frontend (`frontend/.env`)

```env
VITE_API_BASE_URL=http://localhost:8000/api/v1
```

---

## API Reference

Base URL: `http://localhost:8000/api/v1`

### Archives

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/archives` | List all archives |
| `POST` | `/archives` | Upload files + create archive (multipart/form-data) |
| `GET` | `/archives/{id}` | Get single archive |
| `PUT` | `/archives/{id}` | Update archive metadata |
| `DELETE` | `/archives/{id}` | Delete archive + files |
| `POST` | `/archives/suggest-metadata` | Generate AI metadata from uploaded files |

### AI Search

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/ai-search` | Synchronous AI search |
| `POST` | `/ai-search/stream` | Streaming AI search (SSE) |

#### POST `/ai-search` — Request

```json
{
  "query": "traditional batik from Kelantan",
  "thread_id": "user-123"
}
```

#### Response — Results

```json
{
  "response_type": "results",
  "archives": [
    {
      "id": "uuid",
      "title": "Traditional Batik Patterns",
      "description": "...",
      "media_types": ["image"],
      "tags": ["batik", "kelantan"],
      "file_uris": ["https://..."],
      "similarity": 0.85,
      "created_at": "2024-01-15T10:00:00Z"
    }
  ],
  "total": 1,
  "query": "traditional batik from Kelantan",
  "message": null
}
```

#### Response — Non-search Intent (greeting, unrelated, vague)

```json
{
  "response_type": "message",
  "archives": [],
  "total": 0,
  "query": "hello",
  "message": "Hello! I can help you find Malaysian heritage materials. What would you like to search for?"
}
```

---

## Database Schema

```sql
CREATE TABLE archives (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title        TEXT NOT NULL,
    description  TEXT,
    summary      TEXT NOT NULL,          -- Internal, not exposed in API
    embedding    vector NOT NULL,        -- 768-dim, text-embedding-004
    media_types  TEXT[] NOT NULL,        -- ['image', 'video', 'audio', 'document']
    tags         TEXT[],
    dates        TIMESTAMPTZ[],
    storage_paths TEXT[] NOT NULL,       -- Supabase Storage paths
    created_at   TIMESTAMPTZ DEFAULT now(),
    updated_at   TIMESTAMPTZ DEFAULT now()
);
```

Indexes:
- `GIN` on `tags`, `media_types`, `dates` (fast array filtering)
- `HNSW` on `embedding` using `vector_cosine_ops` (fast similarity search)

---

## AI Search Agent

The agent (`app/services/ai_search/agent_v2.py`) is a **LangGraph ReAct agent** powered by `gemini-2.5-flash-lite` with two tools:

| Tool | Description |
|------|-------------|
| `search_archives_db` | Semantic vector search via `match_archives` RPC (threshold: 0.7, limit: 10) |
| `read_archives_data` | Filter by tag, media type, or title for fallback searches |

### Search Flow

```
User Query
    │
    ▼
Intent Classification
    ├── GREETING / UNRELATED → Text message response (no DB query)
    ├── VAGUE_REQUEST        → Ask clarifying question
    └── HERITAGE_SEARCH
            │
            ▼
    Step A: Generate comprehensive semantic query
            │
            ▼
    Step B: search_archives_db (semantic, threshold 0.7)
            │
            ├── Results found → Return structured archives
            │
            └── No results
                    │
                    ▼
            Step C: Autonomous fallback
            ├── Tag filter (read_archives_data filter_by="tag")
            ├── Media type filter (filter_by="media_type")
            └── Title search (filter_by="title")
```

Conversation context is maintained per `thread_id` using LangGraph's `InMemorySaver` checkpointer.

---

## Deployment

### Backend → Render

Deployed as a Docker container via `render.yaml`:
- Region: Singapore (`sgp`)
- Environment: Docker
- Port: `8000`

```bash
# Build and test Docker image locally
cd backend
docker build -t bwm-backend .
docker run -p 8000:8000 --env-file .env bwm-backend
```

### Frontend → Vercel

Deployed automatically from the `frontend/` directory via `vercel.json`:
- Build: `npm run build`
- Output: `build/`

Set `VITE_API_BASE_URL` to your Render backend URL in Vercel's environment variable settings.

---

## License

Copyright © 2025 Badan Warisan Negara. All rights reserved.
