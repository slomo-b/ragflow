"""
Working Configuration for RagFlow Backend
Fixes all Pydantic validation issues
"""

from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict
from typing import List, Optional
import os
from pathlib import Path


class Settings(BaseSettings):
    """Application settings with proper Pydantic configuration"""
    
    # Pydantic configuration - ALLOW extra fields from .env
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"  # This fixes the "extra inputs not permitted" error
    )
    
    # === Core Configuration ===
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "RagFlow"
    VERSION: str = "2.0.0"
    
    # === Environment ===
    ENVIRONMENT: str = Field(default="development")
    DEBUG: bool = Field(default=True)
    
    # === Security ===
    SECRET_KEY: str = Field(default="your_super_secret_key_here_change_this_in_production")
    ALGORITHM: str = Field(default="HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30)
    
    # === Google AI ===
    GOOGLE_API_KEY: str = Field(default="", description="Google AI API Key")
    GEMINI_MODEL: str = Field(default="gemini-1.5-flash")
    EMBEDDING_MODEL: str = Field(default="models/embedding-001")
    TEMPERATURE: float = Field(default=0.7, ge=0.0, le=2.0)
    MAX_TOKENS: int = Field(default=1024, ge=1, le=4096)
    
    # === Database ===
    DATABASE_URL: str = Field(default="sqlite:///./data/rag_database.db")
    DATABASE_ECHO: bool = Field(default=False)
    
    # === File Upload ===
    MAX_FILE_SIZE: int = Field(default=524288000)  # 10MB
    UPLOAD_DIRECTORY: str = Field(default="./uploads")
    
    # === RAG Configuration ===
    CHUNK_SIZE: int = Field(default=1000, ge=100, le=2000)
    CHUNK_OVERLAP: int = Field(default=200, ge=0, le=500)
    SIMILARITY_THRESHOLD: float = Field(default=0.7, ge=0.0, le=1.0)
    MAX_SEARCH_RESULTS: int = Field(default=10, ge=1, le=50)
    
    # === Logging ===
    LOG_LEVEL: str = Field(default="INFO")
    LOG_FORMAT: str = Field(default="text")
    
    # === Performance ===
    WORKERS: int = Field(default=1, ge=1)
    MAX_REQUESTS: int = Field(default=1000, ge=100)
    TIMEOUT_KEEP_ALIVE: int = Field(default=5, ge=1)
    
    # === Rate Limiting ===
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = Field(default=60)
    CHAT_RATE_LIMIT: int = Field(default=20)
    UPLOAD_RATE_LIMIT: int = Field(default=50)
    
    # === Fixed Lists (as properties to avoid JSON parsing issues) ===
    @property
    def ALLOWED_EXTENSIONS(self) -> List[str]:
        """Get allowed file extensions"""
        return [".pdf", ".docx", ".txt", ".md"]
    
    @property
    def ALLOWED_ORIGINS(self) -> List[str]:
        """Get allowed CORS origins"""
        return [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:8000",
            "http://127.0.0.1:8000"
        ]
    
    @property 
    def ALLOWED_METHODS(self) -> List[str]:
        """Get allowed HTTP methods"""
        return ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    
    @property
    def ALLOWED_HEADERS(self) -> List[str]:
        """Get allowed headers"""
        return ["*"]
    
    def create_directories(self):
        """Create necessary directories"""
        Path(self.UPLOAD_DIRECTORY).mkdir(parents=True, exist_ok=True)
        Path("./data").mkdir(parents=True, exist_ok=True)
        Path("./logs").mkdir(parents=True, exist_ok=True)


# Create settings instance
settings = Settings()

# Create directories
settings.create_directories()

# Validate critical settings and show helpful messages
print("🔧 RagFlow Configuration Loaded:")
print(f"   Environment: {settings.ENVIRONMENT}")
print(f"   Database: {settings.DATABASE_URL}")
print(f"   Upload Directory: {settings.UPLOAD_DIRECTORY}")

if not settings.GOOGLE_API_KEY or settings.GOOGLE_API_KEY == "your_google_ai_api_key_here":
    print("⚠️  WARNING: GOOGLE_API_KEY not set!")
    print("   🔑 Get your API key at: https://ai.google.dev")
    print("   📝 Add it to your .env file: GOOGLE_API_KEY=your_actual_key")
else:
    print(f"✅ Google API Key configured (length: {len(settings.GOOGLE_API_KEY)})")

print(f"💾 Allowed file types: {', '.join(settings.ALLOWED_EXTENSIONS)}")
print(f"📁 Max file size: {settings.MAX_FILE_SIZE / (1024*1024):.1f} MB")