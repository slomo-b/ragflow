# backend/app/features.py - Feature Management für RagFlow
"""
Feature flags and capabilities management for RagFlow
"""

import os
from typing import Dict, List, Any, Optional
from pathlib import Path


class FeatureManager:
    """Manages feature flags and system capabilities"""
    
    def __init__(self):
        self._features = {
            # === CORE FEATURES ===
            "chromadb": True,
            "vector_search": True,
            "document_processing": True,
            "chat_interface": True,
            "project_management": True,
            
            # === AI FEATURES ===
            "google_ai": self._check_google_ai(),
            "openai": self._check_openai(),
            "embeddings": True,
            "rag_search": True,
            
            # === DOCUMENT TYPES ===
            "pdf_support": self._check_pdf_support(),
            "docx_support": self._check_docx_support(),
            "text_support": True,
            "markdown_support": True,
            
            # === STORAGE FEATURES ===
            "file_upload": True,
            "persistent_storage": True,
            "backup_restore": True,
            
            # === ADVANCED FEATURES ===
            "semantic_search": True,
            "conversation_history": True,
            "multi_project": True,
            "admin_interface": True,
            
            # === DEVELOPMENT FEATURES ===
            "debug_mode": self._is_development(),
            "hot_reload": self._is_development(),
            "reset_database": self._is_development(),
        }
    
    def _check_google_ai(self) -> bool:
        """Check if Google AI is available"""
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("RAGFLOW_GOOGLE_API_KEY")
        if not api_key:
            return False
        
        try:
            import google.generativeai
            return True
        except ImportError:
            return False
    
    def _check_openai(self) -> bool:
        """Check if OpenAI is available"""
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("RAGFLOW_OPENAI_API_KEY")
        if not api_key:
            return False
        
        try:
            import openai
            return True
        except ImportError:
            return False
    
    def _check_pdf_support(self) -> bool:
        """Check if PDF processing is available"""
        try:
            import PyPDF2
            return True
        except ImportError:
            return False
    
    def _check_docx_support(self) -> bool:
        """Check if DOCX processing is available"""
        try:
            import docx
            return True
        except ImportError:
            return False
    
    def _is_development(self) -> bool:
        """Check if running in development mode"""
        env = os.getenv("RAGFLOW_ENVIRONMENT", "development").lower()
        return env in ["development", "dev", "local"]
    
    def is_enabled(self, feature_name: str) -> bool:
        """Check if a feature is enabled"""
        return self._features.get(feature_name, False)
    
    def enable_feature(self, feature_name: str) -> None:
        """Enable a feature"""
        self._features[feature_name] = True
    
    def disable_feature(self, feature_name: str) -> None:
        """Disable a feature"""
        self._features[feature_name] = False
    
    def get_enabled_features(self) -> List[str]:
        """Get list of enabled features"""
        return [name for name, enabled in self._features.items() if enabled]
    
    def get_disabled_features(self) -> List[str]:
        """Get list of disabled features"""
        return [name for name, enabled in self._features.items() if not enabled]
    
    def summary(self) -> Dict[str, Any]:
        """Get feature summary"""
        enabled = self.get_enabled_features()
        disabled = self.get_disabled_features()
        
        return {
            "total_features": len(self._features),
            "enabled_count": len(enabled),
            "disabled_count": len(disabled),
            "enabled_features": enabled,
            "disabled_features": disabled,
            "core_status": {
                "database": self.is_enabled("chromadb"),
                "ai_enabled": self.is_enabled("google_ai") or self.is_enabled("openai"),
                "document_processing": self.is_enabled("document_processing"),
                "vector_search": self.is_enabled("vector_search")
            }
        }
    
    def detailed_status(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed feature status with descriptions"""
        return {
            "core": {
                "chromadb": {
                    "enabled": self.is_enabled("chromadb"),
                    "description": "ChromaDB vector database integration"
                },
                "vector_search": {
                    "enabled": self.is_enabled("vector_search"),
                    "description": "Semantic vector search capabilities"
                },
                "document_processing": {
                    "enabled": self.is_enabled("document_processing"),
                    "description": "Document upload and processing"
                },
                "chat_interface": {
                    "enabled": self.is_enabled("chat_interface"),
                    "description": "AI chat interface"
                }
            },
            "ai": {
                "google_ai": {
                    "enabled": self.is_enabled("google_ai"),
                    "description": "Google Gemini AI integration"
                },
                "openai": {
                    "enabled": self.is_enabled("openai"),
                    "description": "OpenAI GPT integration"
                },
                "embeddings": {
                    "enabled": self.is_enabled("embeddings"),
                    "description": "Text embedding generation"
                },
                "rag_search": {
                    "enabled": self.is_enabled("rag_search"),
                    "description": "Retrieval Augmented Generation"
                }
            },
            "documents": {
                "pdf_support": {
                    "enabled": self.is_enabled("pdf_support"),
                    "description": "PDF document processing"
                },
                "docx_support": {
                    "enabled": self.is_enabled("docx_support"),
                    "description": "Microsoft Word document processing"
                },
                "text_support": {
                    "enabled": self.is_enabled("text_support"),
                    "description": "Plain text file processing"
                },
                "markdown_support": {
                    "enabled": self.is_enabled("markdown_support"),
                    "description": "Markdown file processing"
                }
            },
            "storage": {
                "file_upload": {
                    "enabled": self.is_enabled("file_upload"),
                    "description": "File upload capabilities"
                },
                "persistent_storage": {
                    "enabled": self.is_enabled("persistent_storage"),
                    "description": "Persistent data storage"
                },
                "backup_restore": {
                    "enabled": self.is_enabled("backup_restore"),
                    "description": "Backup and restore functionality"
                }
            },
            "advanced": {
                "semantic_search": {
                    "enabled": self.is_enabled("semantic_search"),
                    "description": "Advanced semantic search"
                },
                "conversation_history": {
                    "enabled": self.is_enabled("conversation_history"),
                    "description": "Chat conversation history"
                },
                "multi_project": {
                    "enabled": self.is_enabled("multi_project"),
                    "description": "Multiple project management"
                },
                "admin_interface": {
                    "enabled": self.is_enabled("admin_interface"),
                    "description": "Administrative interface"
                }
            }
        }
    
    def get_missing_dependencies(self) -> List[Dict[str, str]]:
        """Get list of missing dependencies for disabled features"""
        missing = []
        
        if not self.is_enabled("google_ai"):
            if not os.getenv("GOOGLE_API_KEY"):
                missing.append({
                    "feature": "google_ai",
                    "dependency": "GOOGLE_API_KEY environment variable",
                    "fix": "Set your Google AI API key in .env file"
                })
            else:
                missing.append({
                    "feature": "google_ai",
                    "dependency": "google-generativeai package",
                    "fix": "pip install google-generativeai"
                })
        
        if not self.is_enabled("openai"):
            if not os.getenv("OPENAI_API_KEY"):
                missing.append({
                    "feature": "openai",
                    "dependency": "OPENAI_API_KEY environment variable",
                    "fix": "Set your OpenAI API key in .env file"
                })
            else:
                missing.append({
                    "feature": "openai",
                    "dependency": "openai package",
                    "fix": "pip install openai"
                })
        
        if not self.is_enabled("pdf_support"):
            missing.append({
                "feature": "pdf_support",
                "dependency": "PyPDF2 package",
                "fix": "pip install PyPDF2"
            })
        
        if not self.is_enabled("docx_support"):
            missing.append({
                "feature": "docx_support", 
                "dependency": "python-docx package",
                "fix": "pip install python-docx"
            })
        
        return missing
    
    def validate_core_features(self) -> Dict[str, Any]:
        """Validate that core features are working"""
        validation = {
            "status": "healthy",
            "errors": [],
            "warnings": []
        }
        
        # Check ChromaDB
        if self.is_enabled("chromadb"):
            try:
                import chromadb
                validation["chromadb"] = "✅ Available"
            except ImportError:
                validation["errors"].append("ChromaDB not installed")
                validation["chromadb"] = "❌ Missing"
                validation["status"] = "error"
        
        # Check AI capabilities
        if not (self.is_enabled("google_ai") or self.is_enabled("openai")):
            validation["warnings"].append("No AI providers configured")
            validation["ai"] = "⚠️ No providers"
        else:
            providers = []
            if self.is_enabled("google_ai"):
                providers.append("Google AI")
            if self.is_enabled("openai"):
                providers.append("OpenAI")
            validation["ai"] = f"✅ {', '.join(providers)}"
        
        # Check document processing
        supported_formats = []
        if self.is_enabled("text_support"):
            supported_formats.append("TXT")
        if self.is_enabled("markdown_support"):
            supported_formats.append("MD")
        if self.is_enabled("pdf_support"):
            supported_formats.append("PDF")
        if self.is_enabled("docx_support"):
            supported_formats.append("DOCX")
        
        validation["document_formats"] = f"✅ {', '.join(supported_formats)}"
        
        return validation


# Global feature manager instance
features = FeatureManager()

# Convenience functions
def is_enabled(feature_name: str) -> bool:
    """Check if a feature is enabled"""
    return features.is_enabled(feature_name)

def get_enabled_features() -> List[str]:
    """Get list of enabled features"""
    return features.get_enabled_features()

def validate_features() -> Dict[str, Any]:
    """Validate core features"""
    return features.validate_core_features()

# Development helper
if __name__ == "__main__":
    import json
    
    print("🔧 RagFlow Feature Status:")
    print(json.dumps(features.summary(), indent=2))
    
    print("\n🔍 Detailed Status:")
    detailed = features.detailed_status()
    for category, features_dict in detailed.items():
        print(f"\n{category.upper()}:")
        for feature, info in features_dict.items():
            status = "✅" if info["enabled"] else "❌"
            print(f"  {status} {feature}: {info['description']}")
    
    print("\n🚨 Missing Dependencies:")
    missing = features.get_missing_dependencies()
    if missing:
        for dep in missing:
            print(f"  ❌ {dep['feature']}: {dep['dependency']}")
            print(f"     Fix: {dep['fix']}")
    else:
        print("  ✅ All dependencies satisfied!")
    
    print("\n🏥 Core Feature Validation:")
    validation = features.validate_core_features()
    print(json.dumps(validation, indent=2))