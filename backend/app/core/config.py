from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    PROJECT_NAME: str = "AI-Chatbot API"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "FastAPI application for AI Chatbot"
    API_V1_STR: str = "/api/v1"
    
    # Database settings (configure as needed)
    DATABASE_URL: str = ""
    
    # Security settings
    SECRET_KEY: str = ""
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS settings
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173","https://ai-chatbot-zeta-pearl-32.vercel.app"]
    
    # Google GenAI settings
    GOOGLE_GENAI_API_KEY: str = ""
    
    # Supabase settings
    SUPABASE_URL: str = ""
    SUPABASE_SERVICE_ROLE_KEY: str = ""
    SUPABASE_STORAGE_BUCKET: str = "archive-materials"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

