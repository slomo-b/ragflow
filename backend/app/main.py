#!/usr/bin/env python3
"""
RagFlow Backend - Modern Version 2025 mit korrigiertem RAG System
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
    gemini_model: str = Field(default="gemini-1.5-flash", description="Gemini model name")
    
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

# === EINFACHES ABER FUNKTIONALES RAG SYSTEM ===
class SimpleRAGSystem:
    """Einfaches aber robustes RAG System"""
    
    def __init__(self):
        self.documents = {}
        self.project_docs = {}
        self.document_chunks = {}
        
        # Initialize TF-IDF if sklearn available
        self.vectorizer = None
        self.tfidf_matrix = None
        self.document_texts = []
        self.document_ids = []
        
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
            rprint("✅ TF-IDF RAG System initialized")
        else:
            rprint("⚠️ Using simple keyword-based RAG (install scikit-learn for better performance)")
    
    def add_document(self, doc_id: str, content: str, metadata: Dict[str, Any] = None, project_ids: List[str] = None):
        """Add document to RAG system"""
        try:
            if not content or not content.strip():
                rprint(f"⚠️ Empty content for document {doc_id}")
                return
            
            # Store document
            self.documents[doc_id] = {
                'content': content,
                'metadata': metadata or {},
                'project_ids': project_ids or []
            }
            
            # Index by project
            for project_id in (project_ids or []):
                if project_id not in self.project_docs:
                    self.project_docs[project_id] = []
                if doc_id not in self.project_docs[project_id]:
                    self.project_docs[project_id].append(doc_id)
            
            # Update search index
            self._update_search_index()
            
            rprint(f"✅ Document {doc_id} added to RAG system")
            
        except Exception as e:
            rprint(f"❌ Error adding document {doc_id}: {e}")
    
    def _update_search_index(self):
        """Update the search index with all documents"""
        try:
            if not self.vectorizer:
                return  # Skip if no sklearn
            
            # Collect all document texts
            self.document_texts = []
            self.document_ids = []
            
            for doc_id, doc_data in self.documents.items():
                self.document_texts.append(doc_data['content'])
                self.document_ids.append(doc_id)
            
            if self.document_texts:
                # Fit TF-IDF
                self.tfidf_matrix = self.vectorizer.fit_transform(self.document_texts)
                rprint(f"✅ Search index updated with {len(self.document_texts)} documents")
            
        except Exception as e:
            rprint(f"❌ Error updating search index: {e}")
    
    def search(self, query: str, top_k: int = 5, project_ids: List[str] = None):
        """Search documents using TF-IDF or simple keyword matching"""
        try:
            if not query or not query.strip():
                return []
            
            # Filter documents by project if specified
            search_docs = {}
            if project_ids:
                for project_id in project_ids:
                    if project_id in self.project_docs:
                        for doc_id in self.project_docs[project_id]:
                            if doc_id in self.documents:
                                search_docs[doc_id] = self.documents[doc_id]
            else:
                search_docs = self.documents
            
            if not search_docs:
                return []
            
            # Use TF-IDF search if available
            if self.vectorizer and self.tfidf_matrix is not None:
                return self._tfidf_search(query, search_docs, top_k)
            else:
                return self._keyword_search(query, search_docs, top_k)
                
        except Exception as e:
            rprint(f"❌ Search error: {e}")
            return []
    
    def _tfidf_search(self, query: str, search_docs: Dict, top_k: int):
        """TF-IDF based search"""
        try:
            # Transform query
            query_vector = self.vectorizer.transform([query])
            
            # Calculate similarities
            similarities = self.cosine_similarity(query_vector, self.tfidf_matrix)[0]
            
            # Get top results
            results = []
            for idx, similarity in enumerate(similarities):
                if similarity > 0:
                    doc_id = self.document_ids[idx]
                    if doc_id in search_docs:
                        doc_data = search_docs[doc_id]
                        results.append({
                            'doc_id': doc_id,
                            'content': doc_data['content'][:500],  # First 500 chars
                            'score': float(similarity),
                            'metadata': doc_data['metadata']
                        })
            
            # Sort by score and return top_k
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            rprint(f"❌ TF-IDF search error: {e}")
            return []
    
    def _keyword_search(self, query: str, search_docs: Dict, top_k: int):
        """Simple keyword-based search"""
        try:
            results = []
            query_lower = query.lower()
            query_words = query_lower.split()
            
            for doc_id, doc_data in search_docs.items():
                content = doc_data['content'].lower()
                
                # Count keyword matches
                matches = 0
                for word in query_words:
                    matches += content.count(word)
                
                if matches > 0:
                    # Simple scoring based on matches
                    score = matches / len(query_words)
                    
                    results.append({
                        'doc_id': doc_id,
                        'content': doc_data['content'][:500],  # First 500 chars
                        'score': score,
                        'metadata': doc_data['metadata']
                    })
            
            # Sort by score and return top_k
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            rprint(f"❌ Keyword search error: {e}")
            return []

# === AI CHAT SYSTEM ===
class ModernAIChat:
    """AI Chat mit Google Gemini"""
    
    def __init__(self):
        self.model = None
        self._initialize_ai()
    
    def _initialize_ai(self):
        """Initialize AI model"""
        if not settings.google_api_key or settings.google_api_key.strip() == "":
            rprint("⚠️ Google AI API key not configured")
            return
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=settings.google_api_key)
            
            # Fix model name if needed
            model_name = settings.gemini_model
            if model_name == 'gemini-2.5-flash':
                model_name = 'gemini-1.5-flash'
            
            self.model = genai.GenerativeModel(model_name)
            rprint(f"✅ Google AI initialized with model: {model_name}")
            
        except ImportError:
            rprint("❌ google-generativeai not installed")
        except Exception as e:
            rprint(f"❌ Failed to initialize Google AI: {e}")
    
    async def generate_response(self, query: str, context: str = "", project_id: str = None):
        """Generate AI response"""
        if not self.model:
            return "❌ AI service not configured. Please set GOOGLE_API_KEY in .env file."
        
        try:
            if context:
                prompt = f"""Based on the following context from documents, answer the user's question:

Context:
{context}

Question: {query}

Please provide a helpful answer based on the context. If the context doesn't contain relevant information, say so clearly."""
            else:
                prompt = query
            
            response = await asyncio.to_thread(self.model.generate_content, prompt)
            return response.text
        
        except Exception as e:
            rprint(f"⚠️ AI generation failed: {e}")
            return f"Sorry, I encountered an error generating a response: {str(e)}"

# Initialize components
rag_system = SimpleRAGSystem()
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

# === SIMPLE DOCUMENT PROCESSING ===
async def process_document(file_path: Path, content_type: str = None) -> str:
    """Simple document text extraction"""
    try:
        text = ""
        file_extension = file_path.suffix.lower()
        
        if file_extension == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        elif file_extension == '.md':
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        elif file_extension == '.pdf' and features.pypdf:
            try:
                import pypdf
                with open(file_path, 'rb') as f:
                    reader = pypdf.PdfReader(f)
                    text = "\n".join([page.extract_text() for page in reader.pages])
            except Exception as e:
                rprint(f"⚠️ PDF extraction failed: {e}")
                text = f"PDF file: {file_path.name} (extraction failed)"
        elif file_extension in ['.docx', '.doc'] and features.docx:
            try:
                import docx
                doc = docx.Document(file_path)
                text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            except Exception as e:
                rprint(f"⚠️ DOCX extraction failed: {e}")
                text = f"Word document: {file_path.name} (extraction failed)"
        else:
            text = f"File: {file_path.name} (text extraction not supported for {file_extension})"
        
        return text
    
    except Exception as e:
        rprint(f"❌ Document processing failed: {e}")
        return f"File: {file_path.name} (processing failed: {str(e)})"

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

@app.get("/api/config")
async def get_configuration():
    """Get backend configuration information"""
    return {
        "google_api_configured": bool(settings.google_api_key and settings.google_api_key.strip()),
        "upload_dir": str(settings.upload_dir),
        "data_dir": str(settings.data_dir),
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "max_file_size": settings.max_file_size,
        "top_k": settings.top_k,
        "features": features.summary()
    }

@app.get("/api/ai/info")
async def get_ai_info():
    """Get AI model information"""
    if not settings.google_api_key or settings.google_api_key.strip() == "":
        raise HTTPException(status_code=503, detail="Google AI API key not configured")
    
    return {
        "model": settings.gemini_model,
        "provider": "Google AI",
        "features": ["chat", "text_generation", "rag_search"],
        "status": "available" if settings.google_api_key else "unavailable",
        "api_configured": bool(settings.google_api_key and settings.google_api_key.strip())
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
        
        # Extract text
        extracted_text = await process_document(file_path, file.content_type)
        
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

# === SEARCH ENDPOINT ===
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
        search_results = rag_system.search(
            query=query,
            top_k=top_k,
            project_ids=[project_id] if project_id else None
        )
        
        # Format results
        formatted_results = []
        for result in search_results:
            doc_data = documents_db.get(result['doc_id'], {})
            formatted_results.append({
                "id": result['doc_id'],
                "content": result['content'],
                "score": result['score'],
                "metadata": result['metadata'],
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
        return {
            "query": request.get("query", ""),
            "results": [],
            "total_found": 0,
            "error": str(e),
            "message": "Search temporarily unavailable"
        }

# === CHAT ENDPOINT ===
@app.post("/api/chat")
async def chat_with_ai(request: dict):
    """Enhanced chat endpoint with RAG integration"""
    try:
        message = request.get("message", "")
        project_id = request.get("project_id")
        
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")
        
        # Check if Google AI is configured
        if not settings.google_api_key or settings.google_api_key.strip() == "":
            return {
                "response": "❌ AI service not configured. Please set GOOGLE_API_KEY in the .env file.",
                "error": "AI_SERVICE_UNAVAILABLE",
                "chat_id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow().isoformat(),
                "sources": []
            }
        
        # Get context from RAG if project_id provided
        context = ""
        sources = []
        
        if project_id:
            try:
                search_results = rag_system.search(
                    query=message,
                    top_k=3,
                    project_ids=[project_id]
                )
                
                if search_results:
                    context = "\n\n".join([
                        f"Document: {result['metadata'].get('original_filename', 'Unknown')}\n{result['content']}"
                        for result in search_results
                    ])
                    
                    sources = [{
                        "id": result['doc_id'],
                        "name": result['metadata'].get('original_filename', 'Unknown'),
                        "excerpt": result['content'][:200] + "..." if len(result['content']) > 200 else result['content'],
                        "relevance_score": result['score']
                    } for result in search_results]
            except Exception as e:
                rprint(f"⚠️ RAG search failed: {e}")
        
        # Generate AI response
        ai_response = await ai_chat.generate_response(
            query=message,
            context=context,
            project_id=project_id
        )
        
        # Ensure response is not empty
        if not ai_response or len(ai_response.strip()) < 5:
            ai_response = "I received your message but couldn't generate a proper response. Please try rephrasing your question."
        
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
                "model": settings.gemini_model,
                "provider": "Google AI"
            }
        }
        
    except Exception as e:
        error_msg = f"Sorry, I encountered an error: {str(e)}"
        rprint(f"❌ Chat error: {e}")
        
        return {
            "response": error_msg,
            "error": "GENERATION_ERROR", 
            "chat_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "sources": []
        }

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
            "status": "completed" if extracted_text else "failed",
            "text_length": len(extracted_text) if extracted_text else 0
        }
        
    except Exception as e:
        rprint(f"❌ Document upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/api/documents", response_model=List[DocumentResponse])
async def get_documents(project_id: Optional[str] = None):
    """Get documents, optionally filtered by project"""
    documents = []
    
    for doc_id, doc_data in documents_db.items():
        # Filter by project if specified
        if project_id and project_id not in doc_data.get("project_ids", []):
            continue
            
        documents.append(DocumentResponse(
            id=doc_id,
            filename=doc_data["filename"],
            file_size=doc_data["file_size"],
            file_type=doc_data["file_type"],
            processing_status=doc_data["processing_status"],
            created_at=datetime.fromisoformat(doc_data["created_at"]) 
                      if isinstance(doc_data["created_at"], str) 
                      else doc_data["created_at"],
            project_ids=doc_data.get("project_ids", [])
        ))
    
    return documents

@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete document"""
    if document_id not in documents_db:
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        # Remove from RAG system
        if document_id in rag_system.documents:
            del rag_system.documents[document_id]
            rag_system._update_search_index()
        
        # Remove file
        doc_data = documents_db[document_id]
        file_path = Path(doc_data["file_path"])
        if file_path.exists():
            file_path.unlink()
        
        # Remove from database
        del documents_db[document_id]
        await save_data()
        
        return {"message": "Document deleted successfully"}
        
    except Exception as e:
        rprint(f"❌ Document deletion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")

# === CHAT HISTORY ENDPOINTS ===
@app.get("/api/chats")
async def get_chats(project_id: Optional[str] = None):
    """Get chat history"""
    chats = []
    
    for chat_id, chat_data in chats_db.items():
        # Filter by project if specified
        if project_id and chat_data.get("project_id") != project_id:
            continue
            
        chats.append({
            "id": chat_id,
            "project_id": chat_data.get("project_id"),
            "created_at": chat_data["created_at"],
            "message_count": len(chat_data.get("messages", [])),
            "last_message": chat_data.get("messages", [])[-1]["content"][:100] + "..." 
                           if chat_data.get("messages") else ""
        })
    
    # Sort by creation time
    chats.sort(key=lambda x: x["created_at"], reverse=True)
    return chats

@app.get("/api/chats/{chat_id}")
async def get_chat(chat_id: str):
    """Get specific chat"""
    if chat_id not in chats_db:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    return chats_db[chat_id]

@app.delete("/api/chats/{chat_id}")
async def delete_chat(chat_id: str):
    """Delete chat"""
    if chat_id not in chats_db:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    del chats_db[chat_id]
    await save_data()
    
    return {"message": "Chat deleted successfully"}

# === ADMIN ENDPOINTS ===
@app.get("/api/admin/stats")
async def get_admin_stats():
    """Get system statistics for admin"""
    return {
        "projects": {
            "total": len(projects_db),
            "projects": [
                {
                    "id": pid,
                    "name": pdata["name"],
                    "document_count": sum(1 for doc in documents_db.values() 
                                        if pid in doc.get("project_ids", [])),
                    "created_at": pdata["created_at"]
                }
                for pid, pdata in projects_db.items()
            ]
        },
        "documents": {
            "total": len(documents_db),
            "by_status": {
                "completed": sum(1 for doc in documents_db.values() 
                               if doc.get("processing_status") == "completed"),
                "failed": sum(1 for doc in documents_db.values() 
                            if doc.get("processing_status") == "failed"),
                "pending": sum(1 for doc in documents_db.values() 
                             if doc.get("processing_status") == "pending")
            },
            "total_size": sum(doc.get("file_size", 0) for doc in documents_db.values())
        },
        "chats": {
            "total": len(chats_db),
            "total_messages": sum(len(chat.get("messages", [])) for chat in chats_db.values())
        },
        "rag": {
            "indexed_documents": len(rag_system.documents),
            "search_method": "TF-IDF" if rag_system.vectorizer else "Keyword-based"
        },
        "features": features.summary()
    }

@app.post("/api/admin/reindex")
async def reindex_documents():
    """Reindex all documents in RAG system"""
    try:
        # Clear current index
        rag_system.documents.clear()
        rag_system.project_docs.clear()
        
        # Reload all documents
        await load_documents_into_rag()
        
        return {
            "message": "Documents reindexed successfully",
            "indexed_count": len(rag_system.documents)
        }
        
    except Exception as e:
        rprint(f"❌ Reindexing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reindexing failed: {str(e)}")

# === ERROR HANDLERS ===
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    rprint(f"❌ Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )

# === MAIN EXECUTION ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level="info"
    )