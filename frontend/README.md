# BWM Heritage Archive — Frontend

React + TypeScript frontend for the Badan Warisan Malaysia AI chatbot. Provides museum curators with a natural-language search interface and archive management tools for Malaysia's national cultural heritage collection.

## 🎯 Purpose

This application helps museum curators and managers at Badan Warisan Negara efficiently manage and retrieve digital heritage archives (photos, videos, documents, oral histories) through an intelligent chatbot interface.

## ✨ Key Features

### 1. **AI Heritage Search** 💬
- Natural language search through thousands of heritage items
- File upload support for AI-assisted tagging
- Smart filters (date range, media type, keywords)
- Real-time search results panel
- Chat history with conversation context

### 2. **Curator Dashboard** 📊
- **Statistics Overview**: Total items, monthly uploads, pending reviews, storage usage
- **Recent Uploads**: Quick access to recently added items
- **Tag Management**: View and manage the most used tags across the collection
- **Activity Log**: Track all curator actions (uploads, edits, deletions)
- **Curator Tools**: Bulk operations, data export, analytics, quality checks

### 3. **Settings Panel** ⚙️
- **General**: Language, timezone, display preferences
- **AI Configuration**: Model selection, search confidence, auto-tagging
- **Notifications**: Email alerts, activity notifications
- **Database**: Backup settings, storage management
- **User Management**: Curator access control

## 🏗️ Architecture

### Why This Design?

**Problem**: With thousands of items in the database, browsing is impractical.

**Solution**: 
- **Search-First**: AI chatbot is the primary way to find items
- **Dashboard**: Curators manage metadata, tags, and recent uploads
- **No Browse View**: Users search, not scroll through thousands of items

### Components Structure

```
/App.tsx                          # Main application
/components/
  ├── Sidebar.tsx                 # Navigation
  ├── TopBar.tsx                  # Header with user info
  ├── ChatPanel.tsx               # AI search interface
  ├── SearchResultsPanel.tsx      # Real-time search results
  ├── CuratorDashboard.tsx        # Management tools and analytics
  ├── SettingsPanel.tsx           # Configuration
  ├── AddArchiveModal.tsx         # Upload new items
  ├── ArchiveCard.tsx             # Item display component
  ├── ChatMessage.tsx             # Chat bubble component
  └── ChatFilters.tsx             # Search filters
```

## 🎨 Design Features

- **Heritage-inspired**: Earth tones (amber/stone palette), subtle textures
- **Professional**: Clean, minimalistic interface suitable for government institutions
- **Responsive**: Works on desktop and mobile devices
- **Animations**: Smooth transitions using Motion (Framer Motion)
- **Accessibility**: Built with ShadCN UI components

## 🚀 Getting Started

### Prerequisites
- Node.js 16+
- npm or yarn

### Installation

```bash
# Install dependencies
npm install

# Run development server
npm run dev
```

### Environment Setup

Create a `frontend/.env` file:

```env
# Backend API URL
VITE_API_BASE_URL=http://localhost:8000/api/v1
```

For production, set `VITE_API_BASE_URL` to your Render backend URL in Vercel's environment variable settings.

## 🔌 Backend Integration

All API calls are in `src/services/api.ts`. The backend runs on FastAPI at `VITE_API_BASE_URL`.

### Key Endpoints:

| Endpoint | Description |
|---|---|
| `POST /ai-search` | Synchronous AI search — returns `AISearchResponse` |
| `POST /ai-search/stream` | Streaming AI search via SSE — emits `AISearchStreamUpdate` events |
| `GET /archives` | List all archives |
| `POST /archives` | Upload files + create archive (`multipart/form-data`) |
| `PUT /archives/{id}` | Update archive metadata |
| `DELETE /archives/{id}` | Delete archive + storage files |
| `POST /archives/suggest-metadata` | AI-generated metadata suggestions from uploaded files |

### Response Types:

```typescript
// Search result response
{ response_type: 'results', archives: ArchiveResponse[], total: number, query: string, message: null }

// Non-heritage / greeting / vague query
{ response_type: 'message', archives: [], total: 0, query: string, message: string }
```

## 📱 Usage

### For Curators:

1. **Search Archives**: Use the AI chatbot to find items naturally
   - Example: "Find photos of batik from Kelantan"
   - Attach files for AI analysis and tagging

2. **Manage Collection**: Go to Dashboard
   - View recent uploads
   - Manage tags and metadata
   - Track activity logs
   - Use bulk operations tools

3. **Configure System**: Use Settings
   - Set language and preferences
   - Configure AI behavior
   - Manage notifications
   - Control user access

## 🛠️ Tech Stack

| Technology | Version | Purpose |
|---|---|---|
| React | 18.3 | UI framework |
| TypeScript | 5.9 | Type safety |
| Vite | 7.x | Build tool & dev server |
| Tailwind CSS | v4.0 | Utility-first styling |
| ShadCN UI (Radix UI) | latest | Accessible UI components |
| Lucide React | 0.487 | Icons |
| Motion (Framer Motion) | 12.x | Animations |
| Sonner | 2.x | Toast notifications |
| react-markdown | 10.x | Render markdown AI responses |
| Recharts | 2.x | Dashboard charts |

## 📊 Data & State

All archive data and AI search results are fetched live from the backend via `src/services/api.ts`. The dashboard statistics and activity log are currently placeholder data — connect them to backend analytics endpoints when available.

## 🔒 Security Notes

- This application is designed for internal curator use
- Not intended for collecting PII or storing sensitive data
- Implement proper authentication in production
- Use secure file upload validation
- Apply role-based access control

## 🌐 Supported Languages

- English (Malaysia)
- Bahasa Melayu
- 中文 (Chinese)
- தமிழ் (Tamil)

## 📝 License

Copyright © 2025 Badan Warisan Negara. All rights reserved.

## 🤝 Contributing

This is a government heritage project. For contributions, contact the Badan Warisan Negara IT department.

---

Built with ❤️ for Malaysia's Heritage Preservation
