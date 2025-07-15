# backend/app/config.py
"""
RagFlow Backend Configuration
Zentrale Konfiguration für das Backend-System
"""

import os
from pathlib import Path
from typing import Optional
from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    """Application settings"""
    
    # API Configuration
    API_VERSION: str = "v1"
    API_PREFIX: str = f"/api/{API_VERSION}"
    
    # Google AI Configuration
    GOOGLE_API_KEY: Optional[str] = Field(default=None, env="GOOGLE_API_KEY")
    GEMINI_MODEL: str = Field(default="gemini-1.5-flash", env="GEMINI_MODEL")
    
    # File Upload Configuration
    MAX_FILE_SIZE: int = Field(default=50 * 1024 * 1024, env="MAX_FILE_SIZE")  # 50MB
    ALLOWED_FILE_TYPES: list = [".pdf", ".docx", ".doc", ".txt", ".md"]
    UPLOAD_DIR: str = Field(default="./uploads", env="UPLOAD_DIR")
    
    # Data Storage Configuration
    DATA_DIR: str = Field(default="./data", env="DATA_DIR")
    
    # RAG Configuration
    RAG_CHUNK_SIZE: int = Field(default=500, env="RAG_CHUNK_SIZE")
    RAG_CHUNK_OVERLAP: int = Field(default=50, env="RAG_CHUNK_OVERLAP")
    RAG_TOP_K: int = Field(default=5, env="RAG_TOP_K")
    RAG_MIN_SIMILARITY: float = Field(default=0.1, env="RAG_MIN_SIMILARITY")
    
    # Search Configuration
    TFIDF_MAX_FEATURES: int = Field(default=5000, env="TFIDF_MAX_FEATURES")
    SEMANTIC_MIN_SIMILARITY: float = Field(default=0.2, env="SEMANTIC_MIN_SIMILARITY")
    
    # Chat Configuration
    CHAT_MAX_CONTEXT_LENGTH: int = Field(default=8000, env="CHAT_MAX_CONTEXT_LENGTH")
    CHAT_TEMPERATURE: float = Field(default=0.7, env="CHAT_TEMPERATURE")
    
    # Logging Configuration
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FILE: str = Field(default="ragflow_backend.log", env="LOG_FILE")
    
    # CORS Configuration
    CORS_ORIGINS: list = Field(
        default=["http://localhost:3000", "http://localhost:5173"],
        env="CORS_ORIGINS"
    )
    
    # Development Configuration
    DEBUG: bool = Field(default=False, env="DEBUG")
    RELOAD: bool = Field(default=False, env="RELOAD")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

# Global settings instance
settings = Settings()

# Environment validation
def validate_environment():
    """Validate required environment variables and settings"""
    errors = []
    warnings = []
    
    # Check Google AI API Key
    if not settings.GOOGLE_API_KEY:
        warnings.append("GOOGLE_API_KEY not set - AI features will be limited")
    
    # Check directories
    upload_dir = Path(settings.UPLOAD_DIR)
    data_dir = Path(settings.DATA_DIR)
    
    try:
        upload_dir.mkdir(exist_ok=True)
        data_dir.mkdir(exist_ok=True)
    except Exception as e:
        errors.append(f"Cannot create directories: {e}")
    
    # Check file size limits
    if settings.MAX_FILE_SIZE > 100 * 1024 * 1024:  # 100MB
        warnings.append("MAX_FILE_SIZE is very large - may cause performance issues")
    
    # Check RAG settings
    if settings.RAG_CHUNK_SIZE < 100:
        warnings.append("RAG_CHUNK_SIZE is very small - may affect search quality")
    
    if settings.RAG_CHUNK_OVERLAP >= settings.RAG_CHUNK_SIZE:
        errors.append("RAG_CHUNK_OVERLAP must be smaller than RAG_CHUNK_SIZE")
    
    return errors, warnings

# Model and feature availability
class FeatureFlags:
    """Feature availability flags"""
    
    def __init__(self):
        self.google_ai_available = bool(settings.GOOGLE_API_KEY)
        self.sentence_transformers_available = self._check_sentence_transformers()
        self.pdf_processing_available = self._check_pdf_processing()
        self.docx_processing_available = self._check_docx_processing()
    
    def _check_sentence_transformers(self) -> bool:
        try:
            import sentence_transformers
            return True
        except ImportError:
            return False
    
    def _check_pdf_processing(self) -> bool:
        try:
            import PyPDF2
            return True
        except ImportError:
            try:
                import pdfplumber
                return True
            except ImportError:
                return False
    
    def _check_docx_processing(self) -> bool:
        try:
            import docx
            return True
        except ImportError:
            return False
    
    def get_status_dict(self) -> dict:
        return {
            "google_ai": self.google_ai_available,
            "sentence_transformers": self.sentence_transformers_available,
            "pdf_processing": self.pdf_processing_available,
            "docx_processing": self.docx_processing_available
        }

# Global feature flags
feature_flags = FeatureFlags()

# Development utilities
def get_development_config():
    """Get configuration for development environment"""
    return {
        "debug": True,
        "reload": True,
        "log_level": "DEBUG",
        "cors_origins": ["*"],  # Allow all origins in development
    }

def get_production_config():
    """Get configuration for production environment"""
    return {
        "debug": False,
        "reload": False,
        "log_level": "INFO",
        "cors_origins": settings.CORS_ORIGINS,
    }

# Configuration summary
def print_config_summary():
    """Print configuration summary for debugging"""
    print("\n" + "="*50)
    print("RagFlow Backend Configuration")
    print("="*50)
    
    print(f"Google AI Key: {'✓ Set' if settings.GOOGLE_API_KEY else '✗ Not set'}")
    print(f"Model: {settings.GEMINI_MODEL}")
    print(f"Upload Dir: {settings.UPLOAD_DIR}")
    print(f"Data Dir: {settings.DATA_DIR}")
    print(f"Max File Size: {settings.MAX_FILE_SIZE / (1024*1024):.1f} MB")
    print(f"RAG Chunk Size: {settings.RAG_CHUNK_SIZE}")
    print(f"Debug Mode: {settings.DEBUG}")
    
    print("\nFeature Availability:")
    status = feature_flags.get_status_dict()
    for feature, available in status.items():
        print(f"  {feature}: {'✓' if available else '✗'}")
    
    # Validation
    errors, warnings = validate_environment()
    
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  ⚠️  {warning}")
    
    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"  ❌ {error}")
    
    print("="*50 + "\n")

if __name__ == "__main__":
    print_config_summary()