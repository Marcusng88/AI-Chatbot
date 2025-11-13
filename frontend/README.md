# Badan Warisan Digital Archive - Chatbot Interface

A professional web application for managing and searching Malaysia's National Heritage collection using AI-powered search.

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

Create a `.env` file:

```env
VITE_API_BASE_URL=http://localhost:3000/api
VITE_UPLOAD_MAX_SIZE=10485760
VITE_AI_MODEL=gpt-4
```

## 🔌 Backend Integration

See `API_INTEGRATION.md` for detailed backend integration guide.

### Key Integration Points:

1. **Chat API**: POST `/api/chat` - AI-powered search
2. **Upload API**: POST `/api/upload` - File uploads with progress
3. **Archives API**: GET/POST/DELETE `/api/archives` - CRUD operations
4. **Search API**: POST `/api/search` - Advanced filtering

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

- **Frontend**: React + TypeScript
- **Styling**: Tailwind CSS v4.0
- **UI Components**: ShadCN UI
- **Icons**: Lucide React
- **Animations**: Motion (Framer Motion)
- **Notifications**: Sonner

## 📊 Mock Data

The application includes mock data for demonstration:
- 5 sample archive items
- Simulated AI responses
- Fake activity logs
- Statistics placeholders

Replace with real API calls in production.

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
