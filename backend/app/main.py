"""
RagFlow Backend - Main FastAPI Application
Enhanced document processing with AI-powered features
"""

import asyncio
import logging
import json
import traceback
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .config import settings

# === Windows-Compatible Logger Class ===
class WindowsCompatibleLogger:
    """Windows-compatible logger without emoji characters"""
    
    def __init__(self, log_file: str = "backend_debug.log"):
        self.log_file = Path(log_file)
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging with Windows-compatible encoding"""
        # Ensure UTF-8 encoding for Python output
        if sys.platform == "win32":
            # Force UTF-8 for Windows console
            try:
                os.system('chcp 65001 >nul 2>&1')  # Set console to UTF-8
            except:
                pass
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def log(self, message: str, level: str = "info"):
        """Log a message without emojis for Windows compatibility"""
        # Remove emojis and replace with text equivalents
        clean_message = self._clean_message(message)
        getattr(self.logger, level.lower())(clean_message)
    
    def _clean_message(self, message: str) -> str:
        """Clean message from emojis for Windows compatibility"""
        emoji_replacements = {
            '🚀': '[ROCKET]',
            '✅': '[OK]',
            '⚠️': '[WARNING]',
            '❌': '[ERROR]',
            '📁': '[FOLDER]',
            '🔧': '[CONFIG]',
            '🤖': '[ROBOT]',
            '💾': '[SAVE]',
            '👋': '[WAVE]',
            '📄': '[DOC]',
            '🗑️': '[DELETE]',
            '💬': '[CHAT]',
            '📋': '[CLIPBOARD]'
        }
        
        for emoji, replacement in emoji_replacements.items():
            message = message.replace(emoji, replacement)
        
        return message

# === Initialize Logger ===
debug_logger = WindowsCompatibleLogger("backend_debug.log")

# === Initialize Data Storage ===
projects_db: Dict[str, Dict[str, Any]] = {}
documents_db: Dict[str, Dict[str, Any]] = {}
chats_db: Dict[str, Dict[str, Any]] = {}

# === Data File Paths ===
DATA_DIR = Path("./data")
PROJECTS_FILE = DATA_DIR / "projects.json"
DOCUMENTS_FILE = DATA_DIR / "documents.json"
CHATS_FILE = DATA_DIR / "chats.json"

def load_data():
    """Load data from JSON files"""
    global projects_db, documents_db, chats_db
    
    try:
        if PROJECTS_FILE.exists():
            with open(PROJECTS_FILE, 'r', encoding='utf-8') as f:
                projects_db = json.load(f)
                debug_logger.log(f"[OK] Loaded {len(projects_db)} projects")
        
        if DOCUMENTS_FILE.exists():
            with open(DOCUMENTS_FILE, 'r', encoding='utf-8') as f:
                documents_db = json.load(f)
                debug_logger.log(f"[OK] Loaded {len(documents_db)} documents")
        
        if CHATS_FILE.exists():
            with open(CHATS_FILE, 'r', encoding='utf-8') as f:
                chats_db = json.load(f)
                debug_logger.log(f"[OK] Loaded {len(chats_db)} chats")
                
    except Exception as e:
        debug_logger.log(f"[WARNING] Error loading data: {e}", "warning")

def save_data():
    """Save data to JSON files"""
    try:
        DATA_DIR.mkdir(exist_ok=True)
        
        with open(PROJECTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(projects_db, f, indent=2, ensure_ascii=False)
        
        with open(DOCUMENTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(documents_db, f, indent=2, ensure_ascii=False)
        
        with open(CHATS_FILE, 'w', encoding='utf-8') as f:
            json.dump(chats_db, f, indent=2, ensure_ascii=False)
            
        debug_logger.log("[SAVE] Data saved successfully")
        
    except Exception as e:
        debug_logger.log(f"[ERROR] Error saving data: {e}", "error")

# === Document Processor Import with Absolute Import ===
try:
    # Try absolute import first
    try:
        from document_processor import process_uploaded_document, DocumentProcessor
        debug_logger.log("[OK] Document Processor loaded (absolute import)")
        PROCESSOR_AVAILABLE = True
    except ImportError:
        # Try relative import as fallback
        from .document_processor import process_uploaded_document, DocumentProcessor
        debug_logger.log("[OK] Document Processor loaded (relative import)")
        PROCESSOR_AVAILABLE = True
except ImportError as e:
    debug_logger.log(f"[WARNING] Document Processor not available - using fallback: {e}", "warning")
    PROCESSOR_AVAILABLE = False
    
    # Fallback Document Processor
    class FallbackDocumentProcessor:
        def __init__(self):
            self.supported_types = {'.txt': 'text', '.md': 'markdown'}
        
        async def process_document(self, file_path: str, file_type: str):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                return {
                    "success": True,
                    "text": text,
                    "chunks": [{"id": "1", "text": text, "start_pos": 0, "end_pos": len(text)}],
                    "metadata": {"word_count": len(text.split()), "char_count": len(text)}
                }
            except Exception as e:
                return {"success": False, "error": str(e)}
    
    DocumentProcessor = FallbackDocumentProcessor
    
    async def process_uploaded_document(file_path: str, file_type: str, document_id: str, documents_db: dict):
        processor = DocumentProcessor()
        result = await processor.process_document(file_path, file_type)
        
        if document_id in documents_db:
            doc = documents_db[document_id]
            if result["success"]:
                doc["processing_status"] = "completed"
                doc["extracted_text"] = result["text"]
                doc["text_chunks"] = result["chunks"]
                doc["text_metadata"] = result["metadata"]
                doc["processing_error"] = None
                debug_logger.log(f"[OK] Document processed: {doc.get('filename', 'unknown')}")
            else:
                doc["processing_status"] = "failed"
                doc["processing_error"] = result["error"]
                debug_logger.log(f"[ERROR] Document processing failed: {result['error']}")
            
            doc["processed_at"] = datetime.utcnow().isoformat()

# === Lifespan Event Handler ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    debug_logger.log("[ROCKET] Starting RagFlow Backend...")
    load_data()
    yield
    # Shutdown
    debug_logger.log("[SAVE] Saving data before shutdown...")
    save_data()
    debug_logger.log("[WAVE] RagFlow Backend stopped")

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
            "document_processor": PROCESSOR_AVAILABLE,
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
    import uuid
    
    project_id = str(uuid.uuid4())
    new_project = {
        "id": project_id,
        "name": project.name,
        "description": project.description,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "document_ids": [],
        "document_count": 0,
        "last_activity": datetime.utcnow().isoformat()
    }
    
    projects_db[project_id] = new_project
    save_data()
    
    debug_logger.log(f"[OK] Project created: {project.name} (ID: {project_id})")
    return {"message": "Project created", "project": new_project}

@app.delete("/api/v1/projects/{project_id}")
async def delete_project(project_id: str):
    """Delete a project"""
    if project_id not in projects_db:
        raise HTTPException(status_code=404, detail="Project not found")
    
    deleted_project = projects_db.pop(project_id)
    
    # Remove project association from documents
    for doc in documents_db.values():
        if project_id in doc.get("project_ids", []):
            doc["project_ids"].remove(project_id)
    
    save_data()
    debug_logger.log(f"[DELETE] Project deleted: {deleted_project['name']}")
    return {"message": f"Project '{deleted_project['name']}' deleted", "project": deleted_project}

# === Document Upload Endpoint ===
@app.post("/api/v1/upload/documents")
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    project_id: Optional[str] = Form(None),
    tags: Optional[str] = Form(None)
):
    """Upload documents with background processing"""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    if project_id and project_id not in projects_db:
        raise HTTPException(status_code=404, detail="Project not found")
    
    uploaded_documents = []
    
    for file in files:
        if not file.filename:
            continue
            
        # Validate file type
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400, 
                detail=f"File type {file_extension} not supported. Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}"
            )
        
        # Create document ID
        import uuid
        document_id = str(uuid.uuid4())
        
        # Save file
        upload_dir = Path(settings.UPLOAD_DIRECTORY)
        upload_dir.mkdir(exist_ok=True)
        
        file_path = upload_dir / f"{document_id}_{file.filename}"
        
        try:
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
        
        # Create document entry
        document_entry = {
            "id": document_id,
            "filename": file.filename,
            "file_type": file_extension[1:],  # Remove the dot
            "file_size": len(content),
            "file_path": str(file_path),
            "project_ids": [project_id] if project_id else [],
            "tags": tags.split(",") if tags else [],
            "uploaded_at": datetime.utcnow().isoformat(),
            "processing_status": "pending",
            "chunk_count": 0,
            "total_tokens": None,
            "summary": None,
            "extracted_text": None,
            "text_chunks": []
        }
        
        documents_db[document_id] = document_entry
        
        # Update project
        if project_id:
            projects_db[project_id]["document_ids"].append(document_id)
            projects_db[project_id]["document_count"] = len(projects_db[project_id]["document_ids"])
            projects_db[project_id]["updated_at"] = datetime.utcnow().isoformat()
        
        uploaded_documents.append(document_entry)
        
        # Schedule background processing
        background_tasks.add_task(
            process_uploaded_document,
            str(file_path),
            file_extension[1:],
            document_id,
            documents_db
        )
        
        debug_logger.log(f"[DOC] Document uploaded: {file.filename} (ID: {document_id})")
    
    save_data()
    
    return {
        "message": f"Successfully uploaded {len(uploaded_documents)} document(s)",
        "documents": uploaded_documents,
        "processing_started": True
    }

# === Document Endpoints ===
@app.get("/api/v1/documents/")
async def get_documents(skip: int = 0, limit: int = 10, project_id: Optional[str] = None):
    """Get documents"""
    documents = list(documents_db.values())
    
    if project_id:
        documents = [
            doc for doc in documents 
            if project_id in doc.get("project_ids", [])
        ]
    
    total = len(documents)
    documents = documents[skip:skip+limit]
    
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
    
    document = documents_db.pop(document_id)
    
    # Delete physical file
    try:
        file_path = Path(document["file_path"])
        if file_path.exists():
            file_path.unlink()
    except Exception as e:
        debug_logger.log(f"[WARNING] Failed to delete file {document['file_path']}: {e}", "warning")
    
    # Remove from projects
    for project in projects_db.values():
        if document_id in project.get("document_ids", []):
            project["document_ids"].remove(document_id)
            project["document_count"] = len(project["document_ids"])
            project["updated_at"] = datetime.utcnow().isoformat()
    
    save_data()
    debug_logger.log(f"[DELETE] Document deleted: {document['filename']}")
    return {"message": f"Document '{document['filename']}' deleted", "document": document}

# === Chat Endpoints (Multiple versions for compatibility) ===
@app.post("/api/v1/chat")
async def chat_with_documents(request: Union[ChatMessage, dict, FlexibleChatMessage]):
    """Enhanced chat endpoint with flexible input handling"""
    try:
        # Handle different input types
        if isinstance(request, dict):
            message = request.get("message", "")
            project_id = request.get("project_id")
            use_documents = request.get("use_documents", True)
        else:
            message = request.message
            project_id = getattr(request, 'project_id', None)
            use_documents = getattr(request, 'use_documents', True)
        
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")
        
        # Simple response for now
        response_text = f"I received your message: '{message}'. Backend is working perfectly! 🚀"
        
        # Create chat entry
        import uuid
        chat_id = str(uuid.uuid4())
        
        chat_entry = {
            "id": chat_id,
            "project_id": project_id,
            "user_message": message,
            "ai_response": response_text,
            "timestamp": datetime.utcnow().isoformat(),
            "documents_used": use_documents,
            "metadata": {
                "model": "simple-response",
                "processing_time": 0.1,
                "status": "success"
            }
        }
        
        chats_db[chat_id] = chat_entry
        save_data()
        
        debug_logger.log(f"[CHAT] Chat processed: {message[:50]}...")
        
        return {
            "response": response_text,
            "chat_id": chat_id,
            "timestamp": chat_entry["timestamp"],
            "metadata": chat_entry["metadata"],
            "success": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        debug_logger.log(f"[ERROR] Chat error: {e}", "error")
        debug_logger.log(f"[ERROR] Chat error traceback: {traceback.format_exc()}", "error")
        
        return JSONResponse(
            status_code=200,  # Return 200 but with error in response
            content={
                "response": "Sorry, I encountered an error processing your message. Please try again.",
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
        
        response_text = f"Simple chat response: I received '{message}'. Backend is working great!"
        
        debug_logger.log(f"[SIMPLE-CHAT] Processed: {message[:50]}...")
        
        return {
            "response": response_text,
            "timestamp": datetime.utcnow().isoformat(),
            "success": True,
            "project_id": project_id
        }
        
    except Exception as e:
        debug_logger.log(f"[ERROR] Simple chat error: {e}", "error")
        return {
            "response": "Sorry, there was an error processing your message.",
            "error": str(e),
            "success": False,
            "timestamp": datetime.utcnow().isoformat()
        }

# === Chat History Endpoint ===
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

# === Error Handlers ===
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    debug_logger.log(f"[ERROR] Unhandled exception: {exc}", "error")
    debug_logger.log(f"Traceback: {traceback.format_exc()}", "error")
    
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
    """Log all requests for debugging"""
    start_time = datetime.utcnow()
    
    # Log request
    debug_logger.log(f"[REQUEST] {request.method} {request.url}")
    
    response = await call_next(request)
    
    # Log response
    process_time = (datetime.utcnow() - start_time).total_seconds()
    debug_logger.log(f"[RESPONSE] {response.status_code} - {process_time:.3f}s")
    
    return response

# === Startup Messages ===
debug_logger.log("[ROCKET] RagFlow Backend initialized")
debug_logger.log(f"[FOLDER] Upload directory: {settings.UPLOAD_DIRECTORY}")
debug_logger.log(f"[CONFIG] Environment: {settings.ENVIRONMENT}")
debug_logger.log(f"[ROBOT] Document processor: {'Available' if PROCESSOR_AVAILABLE else 'Fallback mode'}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)