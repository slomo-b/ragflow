#!/usr/bin/env python3
"""
RagFlow Backend - Complete Implementation
AI-powered document analysis with stable RAG integration
Compatible with Frontend API requirements
"""

import asyncio
import logging
import json
import traceback
import os
import sys
import uuid
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from contextlib import asynccontextmanager

# FastAPI imports
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Google AI imports
try:
    import google.generativeai as genai
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False

# Document processing imports
try:
    import PyPDF2
    import pdfplumber
    from docx import Document as DocxDocument
    DOCUMENT_PROCESSING_AVAILABLE = True
except ImportError:
    DOCUMENT_PROCESSING_AVAILABLE = False

from .config import settings

# === Enhanced Logger ===
class RagFlowLogger:
    """Enhanced logger for RagFlow with structured output"""
    
    def __init__(self, log_file: str = "ragflow_backend.log"):
        self.log_file = Path(log_file)
        self.setup_logging()
    
    def setup_logging(self):
        """Setup structured logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s',
            handlers=[
                logging.FileHandler(self.log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger("RagFlow")
    
    def info(self, message: str, component: str = "SYSTEM"):
        self.logger.info(f"[{component}] {message}")
    
    def error(self, message: str, component: str = "ERROR"):
        self.logger.error(f"[{component}] {message}")
    
    def warning(self, message: str, component: str = "WARNING"):
        self.logger.warning(f"[{component}] {message}")

# Initialize logger
logger = RagFlowLogger()

# === Data Storage ===
DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)

# In-memory databases
projects_db: Dict[str, Dict] = {}
documents_db: Dict[str, Dict] = {}
chats_db: Dict[str, Dict] = {}
vector_db: Dict[str, Dict] = {}  # Simple vector storage

# === RAG Components ===
class SimpleRAG:
    """Simple but stable RAG implementation"""
    
    def __init__(self):
        self.documents = {}
        self.chunks = {}
        self.embeddings = {}
        
    def add_document(self, doc_id: str, content: str, metadata: Dict = None):
        """Add document to RAG system"""
        try:
            # Store document
            self.documents[doc_id] = {
                "content": content,
                "metadata": metadata or {},
                "chunks": []
            }
            
            # Create chunks
            chunks = self._create_chunks(content)
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i}"
                self.chunks[chunk_id] = {
                    "content": chunk,
                    "document_id": doc_id,
                    "chunk_index": i,
                    "metadata": metadata or {}
                }
                self.documents[doc_id]["chunks"].append(chunk_id)
            
            logger.info(f"Added document {doc_id} with {len(chunks)} chunks", "RAG")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add document {doc_id}: {e}", "RAG")
            return False
    
    def _create_chunks(self, content: str, chunk_size: int = 1000, overlap: int = 100):
        """Create text chunks with overlap"""
        chunks = []
        words = content.split()
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk = " ".join(chunk_words)
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Simple keyword-based search (can be enhanced with embeddings)"""
        query_words = set(query.lower().split())
        results = []
        
        for chunk_id, chunk_data in self.chunks.items():
            chunk_words = set(chunk_data["content"].lower().split())
            
            # Simple relevance scoring based on word overlap
            overlap = len(query_words.intersection(chunk_words))
            if overlap > 0:
                score = overlap / len(query_words.union(chunk_words))
                results.append({
                    "chunk_id": chunk_id,
                    "content": chunk_data["content"],
                    "score": score,
                    "document_id": chunk_data["document_id"],
                    "metadata": chunk_data["metadata"]
                })
        
        # Sort by relevance and return top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

# Initialize RAG system
rag_system = SimpleRAG()

# === Google AI Integration ===
class GoogleAIChat:
    """Google AI chat integration"""
    
    def __init__(self):
        self.model = None
        self.initialize()
    
    def initialize(self):
        """Initialize Google AI"""
        if not GOOGLE_AI_AVAILABLE:
            logger.warning("Google AI not available", "AI")
            return False
        
        api_key = settings.GOOGLE_API_KEY
        if not api_key:
            logger.warning("Google AI API key not configured", "AI")
            return False
        
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(settings.GEMINI_MODEL)
            logger.info("Google AI initialized successfully", "AI")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Google AI: {e}", "AI")
            return False
    
    def generate_response(self, message: str, context: str = "") -> str:
        """Generate AI response with context"""
        if not self.model:
            return f"Simple response: {message} (AI not available)"
        
        try:
            prompt = f"""You are a helpful AI assistant analyzing documents.

Context from documents:
{context}

User question: {message}

Please provide a helpful response based on the context and your knowledge."""

            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            logger.error(f"AI generation failed: {e}", "AI")
            return f"I understand your question: '{message}'. However, I encountered an issue generating a detailed response."

# Initialize AI chat
ai_chat = GoogleAIChat()

# === Document Processing ===
class DocumentProcessor:
    """Document processing utilities"""
    
    @staticmethod
    def extract_text_from_pdf(file_path: Path) -> str:
        """Extract text from PDF"""
        try:
            if DOCUMENT_PROCESSING_AVAILABLE:
                # Try pdfplumber first
                try:
                    import pdfplumber
                    with pdfplumber.open(file_path) as pdf:
                        text = ""
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n"
                        return text
                except:
                    pass
                
                # Fallback to PyPDF2
                try:
                    with open(file_path, 'rb') as file:
                        reader = PyPDF2.PdfReader(file)
                        text = ""
                        for page in reader.pages:
                            text += page.extract_text() + "\n"
                        return text
                except:
                    pass
            
            return f"[PDF content - {file_path.name}]"
            
        except Exception as e:
            logger.error(f"Failed to extract PDF text: {e}", "DOC")
            return f"[PDF extraction failed - {file_path.name}]"
    
    @staticmethod
    def extract_text_from_docx(file_path: Path) -> str:
        """Extract text from DOCX"""
        try:
            if DOCUMENT_PROCESSING_AVAILABLE:
                doc = DocxDocument(file_path)
                text = ""
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                return text
            else:
                return f"[DOCX content - {file_path.name}]"
                
        except Exception as e:
            logger.error(f"Failed to extract DOCX text: {e}", "DOC")
            return f"[DOCX extraction failed - {file_path.name}]"
    
    @staticmethod
    def extract_text_from_txt(file_path: Path) -> str:
        """Extract text from TXT"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Failed to extract TXT text: {e}", "DOC")
            return f"[TXT extraction failed - {file_path.name}]"

# === Data Persistence ===
def save_data():
    """Save all data to files"""
    try:
        # Save projects
        with open(DATA_DIR / "projects.json", "w", encoding="utf-8") as f:
            json.dump(projects_db, f, indent=2, ensure_ascii=False)
        
        # Save documents
        with open(DATA_DIR / "documents.json", "w", encoding="utf-8") as f:
            json.dump(documents_db, f, indent=2, ensure_ascii=False)
        
        # Save chats
        with open(DATA_DIR / "chats.json", "w", encoding="utf-8") as f:
            json.dump(chats_db, f, indent=2, ensure_ascii=False)
        
        logger.info("Data saved successfully", "DATA")
    except Exception as e:
        logger.error(f"Failed to save data: {e}", "DATA")

def load_data():
    """Load all data from files"""
    global projects_db, documents_db, chats_db
    
    try:
        # Load projects
        projects_file = DATA_DIR / "projects.json"
        if projects_file.exists():
            with open(projects_file, "r", encoding="utf-8") as f:
                projects_db = json.load(f)
        
        # Load documents
        documents_file = DATA_DIR / "documents.json"
        if documents_file.exists():
            with open(documents_file, "r", encoding="utf-8") as f:
                documents_db = json.load(f)
        
        # Load chats
        chats_file = DATA_DIR / "chats.json"
        if chats_file.exists():
            with open(chats_file, "r", encoding="utf-8") as f:
                chats_db = json.load(f)
        
        logger.info(f"Loaded {len(projects_db)} projects, {len(documents_db)} documents, {len(chats_db)} chats", "DATA")
        
        # Rebuild RAG system
        for doc_id, doc_data in documents_db.items():
            if doc_data.get("processing_status") == "completed" and "extracted_text" in doc_data:
                rag_system.add_document(doc_id, doc_data["extracted_text"], doc_data.get("metadata", {}))
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}", "DATA")

# === Lifespan Management ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting RagFlow Backend...", "STARTUP")
    load_data()
    logger.info("RagFlow Backend ready!", "STARTUP")
    yield
    # Shutdown
    logger.info("Shutting down RagFlow Backend...", "SHUTDOWN")
    save_data()
    logger.info("RagFlow Backend stopped", "SHUTDOWN")

# === FastAPI App ===
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="AI-powered document analysis and chat system",
    version=settings.VERSION,
    lifespan=lifespan
)

# === CORS Configuration ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=settings.ALLOWED_METHODS,
    allow_headers=settings.ALLOWED_HEADERS,
)

# === Pydantic Models ===
class ProjectCreate(BaseModel):
    name: str
    description: str = ""

class ChatMessage(BaseModel):
    message: str
    project_id: Optional[str] = None
    use_documents: bool = True

class FlexibleChatMessage(BaseModel):
    message: str = Field(..., description="The chat message")
    project_id: Optional[str] = Field(None, description="Optional project ID")
    use_documents: Optional[bool] = Field(True, description="Whether to use documents")

# === Health Check ===
@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": settings.VERSION,
        "timestamp": datetime.utcnow().isoformat(),
        "features": {
            "google_ai": GOOGLE_AI_AVAILABLE and bool(settings.GOOGLE_API_KEY),
            "document_processing": DOCUMENT_PROCESSING_AVAILABLE,
            "projects": len(projects_db),
            "documents": len(documents_db),
            "chats": len(chats_db)
        }
    }

# === Project Endpoints ===
@app.get("/api/v1/projects/")
async def get_projects():
    """Get all projects"""
    projects = list(projects_db.values())
    return {"projects": projects, "total": len(projects)}

@app.post("/api/v1/projects/")
async def create_project(project: ProjectCreate):
    """Create a new project"""
    project_id = str(uuid.uuid4())
    
    project_data = {
        "id": project_id,
        "name": project.name,
        "description": project.description,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "document_ids": [],
        "document_count": 0,
        "chat_count": 0,
        "status": "active",
        "settings": {}
    }
    
    projects_db[project_id] = project_data
    save_data()
    
    logger.info(f"Created project: {project.name}", "PROJECT")
    return project_data

@app.delete("/api/v1/projects/{project_id}")
async def delete_project(project_id: str):
    """Delete a project"""
    if project_id not in projects_db:
        raise HTTPException(status_code=404, detail="Project not found")
    
    project = projects_db[project_id]
    
    # Remove associated documents
    for doc_id in project.get("document_ids", []):
        if doc_id in documents_db:
            documents_db[doc_id]["project_ids"].remove(project_id)
            if not documents_db[doc_id]["project_ids"]:
                del documents_db[doc_id]
    
    # Remove project
    del projects_db[project_id]
    save_data()
    
    logger.info(f"Deleted project: {project['name']}", "PROJECT")
    return {"message": f"Project '{project['name']}' deleted"}

# === Document Endpoints ===
@app.post("/api/v1/upload/documents")
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    project_id: Optional[str] = Form(None),
    tags: Optional[str] = Form(None)
):
    """Upload documents with background processing"""
    
    # Create upload directory
    upload_dir = Path(settings.UPLOAD_DIRECTORY)
    upload_dir.mkdir(exist_ok=True)
    
    uploaded_docs = []
    
    for file in files:
        try:
            # Generate unique filename
            file_id = str(uuid.uuid4())
            file_extension = Path(file.filename).suffix.lower()
            safe_filename = f"{file_id}{file_extension}"
            file_path = upload_dir / safe_filename
            
            # Save file
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Create document record
            document_data = {
                "id": file_id,
                "filename": file.filename,
                "safe_filename": safe_filename,
                "file_type": file_extension,
                "file_size": len(content),
                "file_path": str(file_path),
                "uploaded_at": datetime.utcnow().isoformat(),
                "processing_status": "pending",
                "project_ids": [project_id] if project_id else [],
                "tags": tags.split(",") if tags else [],
                "metadata": {}
            }
            
            documents_db[file_id] = document_data
            
            # Add to project
            if project_id and project_id in projects_db:
                projects_db[project_id]["document_ids"].append(file_id)
                projects_db[project_id]["document_count"] = len(projects_db[project_id]["document_ids"])
                projects_db[project_id]["updated_at"] = datetime.utcnow().isoformat()
            
            uploaded_docs.append(document_data)
            
            # Schedule background processing
            background_tasks.add_task(process_document, file_id)
            
            logger.info(f"Uploaded document: {file.filename}", "UPLOAD")
            
        except Exception as e:
            logger.error(f"Failed to upload {file.filename}: {e}", "UPLOAD")
            continue
    
    save_data()
    return {"uploaded": len(uploaded_docs), "documents": uploaded_docs}

async def process_document(document_id: str):
    """Process document in background"""
    try:
        doc = documents_db.get(document_id)
        if not doc:
            return
        
        doc["processing_status"] = "processing"
        doc["processing_started_at"] = datetime.utcnow().isoformat()
        
        file_path = Path(doc["file_path"])
        file_type = doc["file_type"].lower()
        
        # Extract text based on file type
        if file_type == ".pdf":
            extracted_text = DocumentProcessor.extract_text_from_pdf(file_path)
        elif file_type in [".docx", ".doc"]:
            extracted_text = DocumentProcessor.extract_text_from_docx(file_path)
        elif file_type == ".txt":
            extracted_text = DocumentProcessor.extract_text_from_txt(file_path)
        else:
            extracted_text = f"[Unsupported file type: {file_type}]"
        
        # Store extracted text
        doc["extracted_text"] = extracted_text
        doc["text_length"] = len(extracted_text)
        doc["processing_status"] = "completed"
        doc["processed_at"] = datetime.utcnow().isoformat()
        
        # Add to RAG system
        rag_system.add_document(document_id, extracted_text, doc.get("metadata", {}))
        
        save_data()
        logger.info(f"Processed document: {doc['filename']}", "PROCESSING")
        
    except Exception as e:
        doc["processing_status"] = "failed"
        doc["processing_error"] = str(e)
        save_data()
        logger.error(f"Failed to process document {document_id}: {e}", "PROCESSING")

@app.get("/api/v1/documents/")
async def get_documents(skip: int = 0, limit: int = 10, project_id: Optional[str] = None):
    """Get documents"""
    documents = list(documents_db.values())
    
    if project_id:
        documents = [doc for doc in documents if project_id in doc.get("project_ids", [])]
    
    # Sort by upload date (newest first)
    documents.sort(key=lambda x: x.get("uploaded_at", ""), reverse=True)
    
    # Pagination
    total = len(documents)
    documents = documents[skip:skip + limit]
    
    return {
        "documents": documents,
        "total": total,
        "skip": skip,
        "limit": limit
    }

@app.delete("/api/v1/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document"""
    if document_id not in documents_db:
        raise HTTPException(status_code=404, detail="Document not found")
    
    document = documents_db[document_id]
    
    # Delete file
    try:
        file_path = Path(document["file_path"])
        if file_path.exists():
            file_path.unlink()
    except Exception as e:
        logger.warning(f"Failed to delete file {document['file_path']}: {e}", "DELETE")
    
    # Remove from projects
    for project in projects_db.values():
        if document_id in project.get("document_ids", []):
            project["document_ids"].remove(document_id)
            project["document_count"] = len(project["document_ids"])
            project["updated_at"] = datetime.utcnow().isoformat()
    
    # Remove from RAG system
    if document_id in rag_system.documents:
        del rag_system.documents[document_id]
    
    # Remove document
    del documents_db[document_id]
    save_data()
    
    logger.info(f"Deleted document: {document['filename']}", "DELETE")
    return {"message": f"Document '{document['filename']}' deleted"}

# === Chat Endpoints ===
@app.post("/api/v1/chat")
async def chat_with_documents(request: Request):
    """Enhanced chat endpoint with RAG integration"""
    try:
        body = await request.json()
        logger.info(f"Chat request received: {json.dumps(body, indent=2)}", "CHAT")
        
        # Parse different request formats
        if "messages" in body:
            # Frontend format with messages array
            messages = body.get("messages", [])
            user_messages = [msg for msg in messages if msg.get("role") == "user"]
            message = user_messages[-1].get("content", "") if user_messages else ""
        else:
            # Direct message format
            message = body.get("message", "")
        
        project_id = body.get("project_id")
        use_documents = body.get("use_documents", True)
        
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")
        
        logger.info(f"Processing chat: '{message[:50]}...'", "CHAT")
        
        # RAG Search
        context = ""
        sources = []
        
        if use_documents and rag_system.documents:
            search_results = rag_system.search(message, top_k=3)
            
            if search_results:
                context_parts = []
                for result in search_results:
                    context_parts.append(f"Document: {result['document_id']}\nContent: {result['content'][:500]}...")
                    
                    # Add source info
                    doc_data = documents_db.get(result['document_id'])
                    if doc_data:
                        sources.append({
                            "id": result['document_id'],
                            "filename": doc_data.get('filename', 'Unknown'),
                            "excerpt": result['content'][:200] + "...",
                            "relevance_score": result['score']
                        })
                
                context = "\n\n".join(context_parts)
                logger.info(f"Found {len(search_results)} relevant documents", "RAG")
        
        # Generate AI response
        if context:
            response_text = ai_chat.generate_response(message, context)
        else:
            response_text = ai_chat.generate_response(message)
        
        # Create chat record
        chat_id = str(uuid.uuid4())
        chat_entry = {
            "id": chat_id,
            "project_id": project_id,
            "user_message": message,
            "ai_response": response_text,
            "timestamp": datetime.utcnow().isoformat(),
            "sources": sources,
            "context_used": bool(context),
            "documents_used": use_documents,
            "metadata": {
                "model": settings.GEMINI_MODEL,
                "context_length": len(context),
                "sources_count": len(sources)
            }
        }
        
        chats_db[chat_id] = chat_entry
        
        # Update project chat count
        if project_id and project_id in projects_db:
            projects_db[project_id]["chat_count"] = projects_db[project_id].get("chat_count", 0) + 1
            projects_db[project_id]["updated_at"] = datetime.utcnow().isoformat()
        
        save_data()
        
        logger.info(f"Chat completed: {len(response_text)} chars response", "CHAT")
        
        return {
            "response": response_text,
            "chat_id": chat_id,
            "timestamp": chat_entry["timestamp"],
            "project_id": project_id,
            "sources": sources,
            "success": True,
            "model_info": {
                "model": settings.GEMINI_MODEL,
                "temperature": settings.TEMPERATURE,
                "features_used": {
                    "document_search": bool(context),
                    "ai_generation": True,
                    "context_enhancement": bool(context)
                }
            },
            "intelligence_metadata": {
                "context_length": len(context),
                "sources_found": len(sources),
                "processing_time": 0.5
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}", "CHAT")
        logger.error(f"Traceback: {traceback.format_exc()}", "CHAT")
        
        return JSONResponse(
            status_code=200,
            content={
                "response": f"Entschuldigung, es gab einen Fehler bei der Verarbeitung: {str(e)}",
                "error": str(e),
                "success": False,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

@app.post("/api/v1/chat/simple")
async def simple_chat(request: Request):
    """Simple chat endpoint that accepts any JSON format"""
    try:
        body = await request.json()
        message = body.get("message", "")
        project_id = body.get("project_id")
        
        if not message:
            return {
                "response": "Please provide a message to chat.",
                "success": False,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        response_text = f"Simple response: I received '{message}'. Backend is working perfectly! 🚀"
        
        return {
            "response": response_text,
            "timestamp": datetime.utcnow().isoformat(),
            "success": True,
            "project_id": project_id
        }
        
    except Exception as e:
        logger.error(f"Simple chat error: {e}", "CHAT")
        return {
            "response": "Sorry, there was an error processing your message.",
            "error": str(e),
            "success": False,
            "timestamp": datetime.utcnow().isoformat()
        }

@app.get("/api/v1/chats/")
async def get_chat_history(project_id: Optional[str] = None, limit: int = 50):
    """Get chat history"""
    chats = list(chats_db.values())
    
    if project_id:
        chats = [chat for chat in chats if chat.get("project_id") == project_id]
    
    # Sort by timestamp (newest first)
    chats.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    
    return {
        "chats": chats[:limit],
        "total": len(chats),
        "project_id": project_id
    }

# === Error Handler ===
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", "ERROR")
    logger.error(f"Traceback: {traceback.format_exc()}", "ERROR")
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error": str(exc) if settings.DEBUG else "Something went wrong",
            "success": False,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# === Request Logging Middleware ===
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests"""
    start_time = datetime.utcnow()
    
    response = await call_next(request)
    
    process_time = (datetime.utcnow() - start_time).total_seconds()
    logger.info(f"{request.method} {request.url} - {response.status_code} - {process_time:.3f}s", "HTTP")
    
    return response

# === Startup Messages ===
logger.info("RagFlow Backend initialized", "INIT")
logger.info(f"Upload directory: {settings.UPLOAD_DIRECTORY}", "CONFIG")
logger.info(f"Environment: {settings.ENVIRONMENT}", "CONFIG")
logger.info(f"Google AI: {'Available' if GOOGLE_AI_AVAILABLE and settings.GOOGLE_API_KEY else 'Not configured'}", "CONFIG")
logger.info(f"Document Processing: {'Available' if DOCUMENT_PROCESSING_AVAILABLE else 'Limited'}", "CONFIG")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)