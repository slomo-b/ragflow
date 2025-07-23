# backend/app/config.py - Pydantic v2 kompatibel mit flexibler Validierung
import os
from pathlib import Path
from typing import Optional, List

# Pydantic v2 Import - BaseSettings ist jetzt in pydantic-settings
try:
    from pydantic_settings import BaseSettings
except ImportError:
    # Fallback für ältere Pydantic Versionen
    from pydantic import BaseSettings

from pydantic import Field, ConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Pydantic v2 model configuration
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignoriere unbekannte Felder statt Fehler
        env_prefix="RAGFLOW_"
    )
    
    # === APP CONFIG ===
    app_name: str = Field(default="RagFlow Backend", description="Application name")
    app_version: str = Field(default="3.0.0", description="Application version")
    environment: str = Field(default="development", description="Environment (development/production)")
    debug: bool = Field(default=True, description="Debug mode")
    
    # === API KEYS ===
    google_api_key: Optional[str] = Field(default=None, description="Google AI API Key", alias="GOOGLE_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API Key (optional)")
    
    # === DIRECTORIES ===
    data_dir: Path = Field(default=Path("./data"), description="Data directory")
    upload_dir: Path = Field(default=Path("./uploads"), description="Upload directory")
    chromadb_dir: Path = Field(default=Path("./data/chromadb"), description="ChromaDB directory")
    
    # === FILE PROCESSING ===
    max_file_size: int = Field(default=50 * 1024 * 1024, description="Max file size (50MB)")
    chunk_size: int = Field(default=1000, description="Text chunk size for RAG")
    chunk_overlap: int = Field(default=100, description="Text chunk overlap")
    
    # === AI MODELS ===
    gemini_model: str = Field(default="gemini-1.5-flash", description="Gemini model name")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", description="Sentence transformer model")
    
    # === RAG CONFIG ===
    top_k: int = Field(default=5, description="Top K results for RAG search")
    temperature: float = Field(default=0.7, description="AI temperature")
    max_tokens: int = Field(default=1000, description="Max tokens for AI response")
    
    # === CHROMADB CONFIG ===
    chromadb_anonymized_telemetry: bool = Field(default=False, description="ChromaDB telemetry")
    chromadb_allow_reset: bool = Field(default=True, description="Allow ChromaDB reset (dev only)")
    
    # === LEGACY SUPPORT - Für Kompatibilität mit alter .env ===
    # Diese Felder werden gelesen aber ignoriert - verhindert Validierungsfehler
    rag_chunk_size: Optional[int] = Field(default=None, description="Legacy field")
    rag_chunk_overlap: Optional[int] = Field(default=None, description="Legacy field") 
    rag_top_k: Optional[int] = Field(default=None, description="Legacy field")
    rag_min_similarity: Optional[float] = Field(default=None, description="Legacy field")
    tfidf_max_features: Optional[int] = Field(default=None, description="Legacy field")
    semantic_min_similarity: Optional[float] = Field(default=None, description="Legacy field")
    chat_max_context_length: Optional[int] = Field(default=None, description="Legacy field")
    chat_temperature: Optional[float] = Field(default=None, description="Legacy field")
    log_level: Optional[str] = Field(default=None, description="Legacy field")
    log_file: Optional[str] = Field(default=None, description="Legacy field")
    cors_origins: Optional[str] = Field(default=None, description="Legacy field")
    reload: Optional[bool] = Field(default=None, description="Legacy field")
    
    def model_post_init(self, __context) -> None:
        """Post initialization - create directories and handle legacy values"""
        # Create directories if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.chromadb_dir.mkdir(parents=True, exist_ok=True)
        
        # Handle legacy field mappings
        if self.rag_chunk_size and not hasattr(self, '_chunk_size_set'):
            self.chunk_size = self.rag_chunk_size
        if self.rag_chunk_overlap and not hasattr(self, '_chunk_overlap_set'):
            self.chunk_overlap = self.rag_chunk_overlap
        if self.rag_top_k and not hasattr(self, '_top_k_set'):
            self.top_k = self.rag_top_k
        if self.chat_temperature and not hasattr(self, '_temperature_set'):
            self.temperature = self.chat_temperature
        
        # Validate API keys in production
        if self.environment == "production":
            if not self.google_api_key:
                raise ValueError("GOOGLE_API_KEY is required in production")
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.environment.lower() in ["development", "dev", "local"]
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.environment.lower() in ["production", "prod"]
    
    def get_database_url(self) -> str:
        """Get ChromaDB database URL/path"""
        return str(self.chromadb_dir.absolute())
    
    def summary(self) -> dict:
        """Get configuration summary (without sensitive data)"""
        return {
            "app_name": self.app_name,
            "app_version": self.app_version,
            "environment": self.environment,
            "debug": self.debug,
            "google_api_configured": bool(self.google_api_key),
            "openai_api_configured": bool(self.openai_api_key),
            "data_dir": str(self.data_dir),
            "upload_dir": str(self.upload_dir),
            "chromadb_dir": str(self.chromadb_dir),
            "max_file_size_mb": self.max_file_size // (1024 * 1024),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_model": self.embedding_model,
            "top_k": self.top_k
        }


# Global settings instance
settings = Settings()

# Development helper
if __name__ == "__main__":
    import json
    print("🔧 RagFlow Configuration:")
    print(json.dumps(settings.summary(), indent=2))
    
    # Check if all directories exist
    print(f"\n📁 Directory Check:")
    print(f"   Data dir exists: {settings.data_dir.exists()}")
    print(f"   Upload dir exists: {settings.upload_dir.exists()}")
    print(f"   ChromaDB dir exists: {settings.chromadb_dir.exists()}")
    
    # Check API keys
    print(f"\n🔑 API Keys:")
    print(f"   Google API: {'✅ Configured' if settings.google_api_key else '❌ Missing'}")
    print(f"   OpenAI API: {'✅ Configured' if settings.openai_api_key else '⚠️ Optional'}")