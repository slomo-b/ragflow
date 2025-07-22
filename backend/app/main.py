#!/usr/bin/env python3
"""
RagFlow Backend - Modern Version 2025
Mit FastAPI 0.115+, Pydantic V2, Python 3.13
"""

import asyncio
import json
import os
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Annotated

# Modern FastAPI imports
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Pydantic V2 imports
from pydantic import BaseModel, Field, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict

# Rich für schöne Ausgaben
try:
    from rich import print as rprint
    from rich.console import Console
    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    def rprint(*args, **kwargs):
        print(*args, **kwargs)

# === MODERNE KONFIGURATION ===
class Settings(BaseSettings):
    """Moderne App-Konfiguration mit Pydantic V2"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # API Configuration
    app_name: str = "RagFlow Backend"
    app_version: str = "3.0.0"
    debug: bool = False
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    
    # Google AI
    google_api_key: Optional[str] = Field(default=None, description="Google AI API Key")
    
    # File Upload
    max_file_size: int = Field(default=100_000_000, description="Max file size in bytes (100MB)")
    upload_dir: Path = Field(default=Path("uploads"), description="Upload directory")
    data_dir: Path = Field(default=Path("data"), description="Data directory")
    
    # RAG Configuration
    chunk_size: int = Field(default=500, description="Text chunk size for RAG")
    chunk_overlap: int = Field(default=50, description="Overlap between chunks")
    top_k: int = Field(default=5, description="Number of top results to return")

# Global settings
settings = Settings()

# Data directories
settings.data_dir.mkdir(exist_ok=True)
settings.upload_dir.mkdir(exist_ok=True)

# Global data stores
projects_db: Dict[str, Any] = {}
documents_db: Dict[str, Any] = {}
chats_db: Dict[str, Any] = {}

# === DEPENDENCY CHECKS ===
class FeatureFlags:
    """Modern feature detection"""
    
    def __init__(self):
        self.numpy = self._check_import("numpy")
        self.sklearn = self._check_import("sklearn")
        self.sentence_transformers = self._check_import("sentence_transformers")
        self.google_ai = self._check_import("google.generativeai") and bool(settings.google_api_key)
        self.pypdf = self._check_import("pypdf")
        self.docx = self._check_import("docx")
        self.chromadb = self._check_import("chromadb")
        
    def _check_import(self, module_name: str) -> bool:
        try:
            __import__(module_name)
            return True
        except ImportError:
            return False
    
    def summary(self) -> Dict[str, bool]:
        return {
            "numpy": self.numpy,
            "sklearn": self.sklearn,
            "sentence_transformers": self.sentence_transformers,
            "google_ai": self.google_ai,
            "pypdf": self.pypdf,
            "docx": self.docx,
            "chromadb": self.chromadb
        }

features = FeatureFlags()

# === PYDANTIC V2 MODELS ===
class ProjectCreate(BaseModel):
    """Project creation model"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    name: str = Field(..., min_length=1, max_length=100, description="Project name")
    description: Optional[str] = Field(default="", max_length=500, description="Project description")

class ProjectResponse(BaseModel):
    """Project response model"""
    id: str
    name: str
    description: str
    created_at: datetime
    document_count: int = 0

class DocumentResponse(BaseModel):
    """Document response model"""
    id: str
    filename: str
    file_size: int
    file_type: str
    processing_status: str
    created_at: datetime
    project_ids: List[str]

class ChatMessage(BaseModel):
    """Chat message model"""
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1)
    timestamp: datetime

class ChatRequest(BaseModel):
    """Chat request model"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    message: str = Field(..., min_length=1, max_length=10000)
    project_id: Optional[str] = None

class ChatResponse(BaseModel):
    """Chat response model"""
    id: str
    response: str
    sources: List[str]
    timestamp: datetime

# === MODERNE RAG SYSTEM ===
class ModernRAG:
    """Moderne RAG-Implementation mit optionalen Features"""
    
    def __init__(self):
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.vectorizer = None
        self.embeddings_model = None
        
        # Initialize TF-IDF if available
        if features.sklearn:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2),
                lowercase=True,
                strip_accents='unicode'
            )
            self.cosine_similarity = cosine_similarity
        
        # Initialize embeddings if available
        if features.sentence_transformers:
            try:
                from sentence_transformers import SentenceTransformer
                self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
                rprint("✅ Sentence Transformers loaded successfully")
            except Exception as e:
                rprint(f"⚠️ Failed to load embeddings: {e}")
    
    def add_document(self, doc_id: str, content: str, metadata: Dict[str, Any] = None, project_ids: List[str] = None):
        """Add document to RAG system"""
        self.documents[doc_id] = {
            'content': content,
            'metadata': metadata or {},
            'project_ids': project_ids or [],
            'chunks': self._create_chunks(content)
        }
        rprint(f"✅ Added document {doc_id} with {len(self._create_chunks(content))} chunks")
    
    def _create_chunks(self, text: str) -> List[str]:
        """Create text chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), settings.chunk_size - settings.chunk_overlap):
            chunk = ' '.join(words[i:i + settings.chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def search(self, query: str, project_id: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """Search documents"""
        if not self.documents:
            return []
        
        # Filter by project if specified
        candidate_docs = {}
        for doc_id, doc_data in self.documents.items():
            if project_id is None or project_id in doc_data.get('project_ids', []):
                candidate_docs[doc_id] = doc_data
        
        if not candidate_docs:
            return []
        
        # Use embeddings if available, otherwise TF-IDF
        if self.embeddings_model:
            return self._embedding_search(query, candidate_docs, limit)
        elif features.sklearn:
            return self._tfidf_search(query, candidate_docs, limit)
        else:
            return self._keyword_search(query, candidate_docs, limit)
    
    def _embedding_search(self, query: str, candidate_docs: Dict, limit: int) -> List[Dict[str, Any]]:
        """Embedding-based search"""
        try:
            query_embedding = self.embeddings_model.encode([query])
            results = []
            
            for doc_id, doc_data in candidate_docs.items():
                chunks = doc_data['chunks']
                if chunks:
                    chunk_embeddings = self.embeddings_model.encode(chunks)
                    similarities = self.embeddings_model.similarity(query_embedding, chunk_embeddings)[0]
                    
                    best_idx = similarities.argmax()
                    best_score = float(similarities[best_idx])
                    
                    if best_score > 0.3:  # Threshold
                        results.append({
                            'document_id': doc_id,
                            'content': chunks[best_idx],
                            'score': best_score,
                            'metadata': doc_data['metadata']
                        })
            
            return sorted(results, key=lambda x: x['score'], reverse=True)[:limit]
        
        except Exception as e:
            rprint(f"⚠️ Embedding search failed: {e}")
            return self._keyword_search(query, candidate_docs, limit)
    
    def _tfidf_search(self, query: str, candidate_docs: Dict, limit: int) -> List[Dict[str, Any]]:
        """TF-IDF based search"""
        try:
            all_chunks = []
            chunk_doc_mapping = []
            
            for doc_id, doc_data in candidate_docs.items():
                for chunk in doc_data['chunks']:
                    all_chunks.append(chunk)
                    chunk_doc_mapping.append(doc_id)
            
            if not all_chunks:
                return []
            
            # Fit TF-IDF
            tfidf_matrix = self.vectorizer.fit_transform(all_chunks + [query])
            query_vector = tfidf_matrix[-1]
            chunk_vectors = tfidf_matrix[:-1]
            
            # Calculate similarities
            similarities = self.cosine_similarity(query_vector, chunk_vectors).flatten()
            
            # Get top results
            top_indices = similarities.argsort()[-limit:][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:
                    doc_id = chunk_doc_mapping[idx]
                    results.append({
                        'document_id': doc_id,
                        'content': all_chunks[idx],
                        'score': float(similarities[idx]),
                        'metadata': candidate_docs[doc_id]['metadata']
                    })
            
            return results
        
        except Exception as e:
            rprint(f"⚠️ TF-IDF search failed: {e}")
            return self._keyword_search(query, candidate_docs, limit)
    
    def _keyword_search(self, query: str, candidate_docs: Dict, limit: int) -> List[Dict[str, Any]]:
        """Fallback keyword search"""
        query_words = set(query.lower().split())
        results = []
        
        for doc_id, doc_data in candidate_docs.items():
            for chunk in doc_data['chunks']:
                chunk_words = set(chunk.lower().split())
                score = len(query_words.intersection(chunk_words)) / len(query_words)
                
                if score > 0:
                    results.append({
                        'document_id': doc_id,
                        'content': chunk,
                        'score': score,
                        'metadata': doc_data['metadata']
                    })
        
        return sorted(results, key=lambda x: x['score'], reverse=True)[:limit]

# === MODERNE AI CHAT ===
class ModernAIChat:
    """Moderne AI Chat mit Google AI"""
    
    def __init__(self):
        self.enabled = False
        
        if features.google_ai:
            try:
                import google.generativeai as genai
                genai.configure(api_key=settings.google_api_key)
                self.model = genai.GenerativeModel('gemini-1.5-pro')
                self.enabled = True
                rprint("✅ Google AI configured successfully")
            except Exception as e:
                rprint(f"⚠️ Failed to configure Google AI: {e}")
    
    async def generate_response(self, query: str, context: str = "", project_id: Optional[str] = None) -> str:
        """Generate AI response"""
        if not self.enabled:
            return "AI chat is not available. Please configure GOOGLE_API_KEY."
        
        try:
            prompt = f"""Based on the following context from documents, answer the user's question.

Context:
{context}

Question: {query}

Please provide a helpful and accurate answer based on the context provided. If the context doesn't contain enough information to answer the question, say so."""
            
            response = await asyncio.to_thread(self.model.generate_content, prompt)
            return response.text
        
        except Exception as e:
            rprint(f"⚠️ AI generation failed: {e}")
            return f"Sorry, I encountered an error generating a response: {str(e)}"

# Initialize components
rag_system = ModernRAG()
ai_chat = ModernAIChat()

# === DATA MANAGEMENT ===
async def save_data():
    """Async data saving"""
    try:
        data_files = {
            "projects.json": projects_db,
            "documents.json": documents_db, 
            "chats.json": chats_db
        }
        
        for filename, data in data_files.items():
            file_path = settings.data_dir / filename
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        rprint("✅ Data saved successfully")
    
    except Exception as e:
        rprint(f"❌ Failed to save data: {e}")

async def load_data():
    """Async data loading"""
    global projects_db, documents_db, chats_db
    
    try:
        data_files = {
            "projects.json": lambda: projects_db,
            "documents.json": lambda: documents_db,
            "chats.json": lambda: chats_db
        }
        
        for filename, get_dict in data_files.items():
            file_path = settings.data_dir / filename
            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if filename == "projects.json":
                        projects_db.update(data)
                    elif filename == "documents.json":
                        documents_db.update(data)
                    elif filename == "chats.json":
                        chats_db.update(data)
        
        # Load documents into RAG
        await load_documents_into_rag()
        
        rprint(f"✅ Loaded {len(projects_db)} projects, {len(documents_db)} documents, {len(chats_db)} chats")
    
    except Exception as e:
        rprint(f"❌ Failed to load data: {e}")

async def load_documents_into_rag():
    """Load documents into RAG system"""
    loaded_count = 0
    
    for doc_id, doc_data in documents_db.items():
        if (doc_data.get("processing_status") == "completed" and 
            "extracted_text" in doc_data):
            
            project_ids = doc_data.get("project_ids", [])
            rag_system.add_document(
                doc_id=doc_id,
                content=doc_data["extracted_text"],
                metadata=doc_data.get("metadata", {}),
                project_ids=project_ids
            )
            loaded_count += 1
    
    rprint(f"✅ Loaded {loaded_count} documents into RAG system")

# === LIFESPAN MANAGEMENT ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern lifespan management"""
    rprint("🚀 Starting RagFlow Backend...")
    await load_data()
    rprint("✅ RagFlow Backend ready!")
    yield
    rprint("👋 Shutting down RagFlow Backend...")
    await save_data()

# === FASTAPI APP ===
app = FastAPI(
    title=settings.app_name,
    description="AI-powered document analysis with modern RAG (Latest FastAPI + Pydantic V2)",
    version=settings.app_version,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === DEPENDENCY INJECTION ===
async def get_settings() -> Settings:
    """Get settings dependency"""
    return settings

# === HEALTH ENDPOINTS ===
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "🚀 RagFlow Backend is running!",
        "version": settings.app_version,
        "docs": "/docs"
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "version": settings.app_version,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "features": features.summary()
    }

@app.get("/api/system/info")
async def system_info():
    """System information endpoint"""
    return {
        "app": {
            "name": settings.app_name,
            "version": settings.app_version,
            "python_version": sys.version
        },
        "features": features.summary(),
        "stats": {
            "projects": len(projects_db),
            "documents": len(documents_db),
            "chats": len(chats_db),
            "rag_documents": len(rag_system.documents)
        },
        "settings": {
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "top_k": settings.top_k
        }
    }

# === PROJECT ENDPOINTS ===
@app.get("/api/projects", response_model=List[ProjectResponse])
async def get_projects():
    """Get all projects"""
    projects = []
    for project_id, project_data in projects_db.items():
        doc_count = sum(1 for doc in documents_db.values() 
                       if project_id in doc.get("project_ids", []))
        
        projects.append(ProjectResponse(
            id=project_id,
            name=project_data["name"],
            description=project_data.get("description", ""),
            created_at=datetime.fromisoformat(project_data["created_at"]) 
                      if isinstance(project_data["created_at"], str) 
                      else project_data["created_at"],
            document_count=doc_count
        ))
    
    return projects

@app.post("/api/projects", response_model=ProjectResponse)
async def create_project(project: ProjectCreate):
    """Create new project"""
    project_id = str(uuid.uuid4())
    timestamp = datetime.utcnow()
    
    project_data = {
        "name": project.name,
        "description": project.description,
        "created_at": timestamp.isoformat()
    }
    
    projects_db[project_id] = project_data
    await save_data()
    
    return ProjectResponse(
        id=project_id,
        name=project_data["name"],
        description=project_data["description"],
        created_at=timestamp,
        document_count=0
    )

# === DOCUMENT ENDPOINTS ===
@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    project_id: str = Form(...)
):
    """Upload and process document"""
    if project_id not in projects_db:
        raise HTTPException(status_code=404, detail="Project not found")
    
    if file.size and file.size > settings.max_file_size:
        raise HTTPException(status_code=413, detail="File too large")
    
    # Save file
    file_id = str(uuid.uuid4())
    file_path = settings.upload_dir / f"{file_id}_{file.filename}"
    
    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Process document
        extracted_text = await process_document(file_path, file.content_type)
        
        # Store document metadata
        doc_data = {
            "filename": file.filename,
            "file_size": len(content),
            "file_type": file.content_type or "unknown",
            "file_path": str(file_path),
            "project_ids": [project_id],
            "processing_status": "completed" if extracted_text else "failed",
            "created_at": datetime.utcnow().isoformat(),
            "extracted_text": extracted_text,
            "metadata": {
                "upload_filename": file.filename,
                "content_type": file.content_type
            }
        }
        
        documents_db[file_id] = doc_data
        
        # Add to RAG system if processing succeeded
        if extracted_text:
            rag_system.add_document(
                doc_id=file_id,
                content=extracted_text,
                metadata=doc_data["metadata"],
                project_ids=[project_id]
            )
        
        await save_data()
        
        return {
            "document_id": file_id,
            "filename": file.filename,
            "status": doc_data["processing_status"],
            "message": "Document uploaded and processed successfully" if extracted_text else "Document uploaded but processing failed",
            "extracted_length": len(extracted_text) if extracted_text else 0
        }
        
    except Exception as e:
        rprint(f"❌ Document upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

async def process_document(file_path: Path, content_type: str) -> str:
    """Extract text from document using modern libraries"""
    try:
        if content_type == "text/plain":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        
        elif content_type == "application/pdf" and features.pypdf:
            try:
                from pypdf import PdfReader
                reader = PdfReader(file_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
            except Exception as e:
                rprint(f"⚠️ PDF processing failed: {e}")
                return ""
        
        elif "wordprocessingml.document" in (content_type or "") and features.docx:
            try:
                from docx import Document
                doc = Document(file_path)
                text = ""
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                return text
            except Exception as e:
                rprint(f"⚠️ DOCX processing failed: {e}")
                return ""
        
        else:
            # Try to read as text
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
    
    except Exception as e:
        rprint(f"❌ Document processing failed: {e}")
        return ""


# === KONFIGURATION ENDPOINT ===
@app.get("/api/config")
async def get_configuration():
    """Get backend configuration information"""
    return {
        "google_api_configured": bool(settings.google_api_key),
        "upload_dir": str(settings.upload_dir),
        "data_dir": str(settings.data_dir),
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "max_file_size": settings.max_file_size,
        "top_k": settings.top_k,
        "features": {
            "google_ai": bool(settings.google_api_key),
            "rag_search": True,
            "file_upload": True,
            "chat": True
        }
    }

# === AI INFO ENDPOINT ===
@app.get("/api/ai/info")
async def get_ai_info():
    """Get AI model information"""
    if not settings.google_api_key:
        raise HTTPException(status_code=503, detail="Google AI API key not configured")
    
    return {
        "model": getattr(settings, 'gemini_model', 'gemini-1.5-flash'),
        "provider": "Google AI",
        "features": ["chat", "text_generation", "rag_search"],
        "status": "available" if settings.google_api_key else "unavailable",
        "api_configured": bool(settings.google_api_key)
    }

# === FILE UPLOAD ENDPOINT ===
@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    project_name: str = Form(...)
):
    """Upload and process file"""
    try:
        # Check file size
        if file.size and file.size > settings.max_file_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max size: {settings.max_file_size / (1024*1024):.1f}MB"
            )
        
        # Check file type
        file_extension = Path(file.filename).suffix.lower()
        allowed_extensions = ['.pdf', '.docx', '.doc', '.txt', '.md']
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_extension} not supported. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Create project if it doesn't exist
        project_id = None
        for pid, pdata in projects_db.items():
            if pdata["name"] == project_name:
                project_id = pid
                break
        
        if not project_id:
            project_id = str(uuid.uuid4())
            projects_db[project_id] = {
                "name": project_name,
                "description": f"Project created via file upload: {file.filename}",
                "created_at": datetime.utcnow().isoformat()
            }
        
        # Save file
        upload_dir = Path(settings.upload_dir)
        upload_dir.mkdir(exist_ok=True)
        
        file_id = str(uuid.uuid4())
        file_path = upload_dir / f"{file_id}_{file.filename}"
        
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Extract text (simplified - you might want to use your existing text extraction)
        extracted_text = ""
        if file_extension == '.txt':
            extracted_text = content.decode('utf-8', errors='ignore')
        elif file_extension == '.md':
            extracted_text = content.decode('utf-8', errors='ignore')
        else:
            # For other formats, you'd need more sophisticated extraction
            extracted_text = f"File uploaded: {file.filename} (extraction not implemented for {file_extension})"
        
        # Store document
        doc_data = {
            "filename": file.filename,
            "file_path": str(file_path),
            "file_size": len(content),
            "file_type": file_extension,
            "project_ids": [project_id],
            "extracted_text": extracted_text,
            "processing_status": "completed",
            "created_at": datetime.utcnow().isoformat(),
            "metadata": {
                "original_filename": file.filename,
                "upload_timestamp": datetime.utcnow().isoformat(),
                "project_name": project_name
            }
        }
        
        documents_db[file_id] = doc_data
        
        # Add to RAG system
        rag_system.add_document(
            doc_id=file_id,
            content=extracted_text,
            metadata=doc_data["metadata"],
            project_ids=[project_id]
        )
        
        await save_data()
        
        return {
            "success": True,
            "file_id": file_id,
            "project_id": project_id,
            "filename": file.filename,
            "text_length": len(extracted_text),
            "message": "File uploaded and processed successfully"
        }
        
    except Exception as e:
        rprint(f"❌ Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# === RAG SEARCH ENDPOINT ===
@app.post("/api/search")
async def search_documents(request: dict):
    """Search documents using RAG"""
    try:
        query = request.get("query", "")
        project_id = request.get("project_id")
        top_k = request.get("top_k", settings.top_k)
        
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        # Perform search
        results = rag_system.search(
            query=query,
            top_k=top_k,
            project_ids=[project_id] if project_id else None
        )
        
        # Format results
        formatted_results = []
        for result in results:
            doc_data = documents_db.get(result.doc_id, {})
            formatted_results.append({
                "id": result.doc_id,
                "content": result.content,
                "score": result.score,
                "metadata": result.metadata,
                "filename": doc_data.get("filename", "Unknown"),
                "project_id": project_id
            })
        
        return {
            "query": query,
            "results": formatted_results,
            "total_found": len(formatted_results),
            "search_params": {
                "top_k": top_k,
                "project_id": project_id
            }
        }
        
    except Exception as e:
        rprint(f"❌ Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# === CHAT ENDPOINTS ===
# === CHAT ENDPOINT VERBESSERUNG ===
# Das existierende Chat-Endpoint sollte auch verbessert werden für bessere Fehlerbehandlung

@app.post("/api/chat")
async def chat_with_ai(request: dict):
    """Enhanced chat endpoint with better error handling"""
    try:
        message = request.get("message", "")
        project_id = request.get("project_id")
        
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")
        
        # Check if Google AI is configured
        if not settings.google_api_key:
            return {
                "response": "Sorry, the AI service is not properly configured. Please set the GOOGLE_API_KEY environment variable.",
                "error": "AI_SERVICE_UNAVAILABLE",
                "chat_id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Import Google AI
        try:
            import google.generativeai as genai
            genai.configure(api_key=settings.google_api_key)
            model = genai.GenerativeModel(getattr(settings, 'gemini_model', 'gemini-1.5-flash'))
        except ImportError:
            raise HTTPException(status_code=503, detail="Google AI library not available")
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"AI service error: {str(e)}")
        
        # Get context from RAG if project_id provided
        context = ""
        sources = []
        
        if project_id:
            search_results = rag_system.search(
                query=message,
                top_k=3,
                project_ids=[project_id]
            )
            
            if search_results:
                context = "\n\n".join([
                    f"Document: {result.metadata.get('original_filename', 'Unknown')}\n{result.content}"
                    for result in search_results
                ])
                
                sources = [{
                    "id": result.doc_id,
                    "name": result.metadata.get('original_filename', 'Unknown'),
                    "excerpt": result.content[:200] + "..." if len(result.content) > 200 else result.content,
                    "relevance_score": result.score
                } for result in search_results]
        
        # Prepare prompt
        if context:
            prompt = f"""Based on the following context from the user's documents, please answer their question:

Context:
{context}

Question: {message}

Please provide a helpful answer based on the context. If the context doesn't contain relevant information, please say so clearly."""
        else:
            prompt = message
        
        # Generate response
        response = model.generate_content(prompt)
        ai_response = response.text
        
        # Save chat
        chat_id = str(uuid.uuid4())
        chat_data = {
            "project_id": project_id,
            "messages": [
                {
                    "role": "user",
                    "content": message,
                    "timestamp": datetime.utcnow().isoformat()
                },
                {
                    "role": "assistant", 
                    "content": ai_response,
                    "timestamp": datetime.utcnow().isoformat()
                }
            ],
            "sources": sources,
            "created_at": datetime.utcnow().isoformat()
        }
        
        chats_db[chat_id] = chat_data
        await save_data()
        
        return {
            "response": ai_response,
            "chat_id": chat_id,
            "project_id": project_id,
            "timestamp": datetime.utcnow().isoformat(),
            "sources": sources,
            "model_info": {
                "model": getattr(settings, 'gemini_model', 'gemini-1.5-flash'),
                "provider": "Google AI"
            }
        }
        
    except Exception as e:
        error_msg = f"Sorry, I encountered an error generating a response: {str(e)}"
        rprint(f"❌ Chat error: {e}")
        
        # Return error but don't raise HTTP exception for better UX
        return {
            "response": error_msg,
            "error": "GENERATION_ERROR", 
            "chat_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "sources": []
        }
