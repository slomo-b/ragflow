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

# === CHAT ENDPOINTS ===
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with documents"""
    try:
        # Search for relevant documents
        search_results = rag_system.search(
            query=request.message,
            project_id=request.project_id,
            limit=settings.top_k
        )
        
        # Prepare context
        context = ""
        sources = []
        
        for result in search_results:
            context += f"Document: {result['document_id']}\n"
            context += f"Content: {result['content']}\n"
            context += f"Score: {result['score']:.3f}\n\n"
            sources.append(result['document_id'])
        
        # Generate AI response
        ai_response = await ai_chat.generate_response(
            query=request.message,
            context=context,
            project_id=request.project_id
        )
        
        # Store chat
        chat_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()
        
        chat_data = {
            "project_id": request.project_id,
            "messages": [
                {
                    "role": "user",
                    "content": request.message,
                    "timestamp": timestamp.isoformat()
                },
                {
                    "role": "assistant", 
                    "content": ai_response,
                    "timestamp": timestamp.isoformat()
                }
            ],
            "sources": sources,
            "created_at": timestamp.isoformat()
        }
        
        chats_db[chat_id] = chat_data
        await save_data()
        
        return ChatResponse(
            id=chat_id,
            response=ai_response,
            sources=sources,
            timestamp=timestamp
        )
        
    except Exception as e:
        rprint(f"❌ Chat failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level="info"
    )