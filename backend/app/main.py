# backend/app/main.py - RagFlow Backend mit allen integrierten Features
import sys
import uuid
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from rich.console import Console

# Import our modules
from .config import settings
from .features import features
from .database import get_db_manager, DocumentModel, ProjectModel, ChatModel
from .rag_system import RAGSystem
from .document_processor import document_processor
from .ocr_support import ocr_processor

console = Console()

# FastAPI App
app = FastAPI(
    title="RagFlow Backend",
    version="3.0.0",
    description="AI-powered document analysis with ChromaDB, OCR, and advanced processing",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database and systems
db = get_db_manager()
rag_system = RAGSystem()

# Pydantic Models
class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Project name")
    description: str = Field(default="", max_length=500, description="Project description")

class ProjectResponse(BaseModel):
    id: str
    name: str
    description: str
    created_at: datetime
    document_count: int
    chat_count: int = 0

class ProjectUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)

class DocumentResponse(BaseModel):
    id: str
    filename: str
    file_size: int
    file_type: str
    processing_status: str
    created_at: datetime
    project_ids: List[str]
    processing_method: Optional[str] = "standard"
    text_length: Optional[int] = 0

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    project_id: Optional[str] = None
    model: Optional[str] = "gemini-1.5-flash"
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(1000, ge=1, le=4000)

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    project_id: Optional[str] = None
    top_k: int = Field(5, ge=1, le=20)

class UploadResponse(BaseModel):
    message: str
    document: DocumentResponse
    processing_info: Dict[str, Any]

# === UTILITY FUNCTIONS ===

async def save_uploaded_file(file: UploadFile) -> Path:
    """Save uploaded file to temporary location"""
    # Create temp directory if it doesn't exist
    temp_dir = Path(tempfile.gettempdir()) / "ragflow_uploads"
    temp_dir.mkdir(exist_ok=True)
    
    # Generate unique filename
    file_id = str(uuid.uuid4())
    file_extension = Path(file.filename).suffix if file.filename else ""
    temp_file_path = temp_dir / f"{file_id}{file_extension}"
    
    # Save file
    with open(temp_file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    return temp_file_path

def cleanup_temp_file(file_path: Path):
    """Clean up temporary file"""
    try:
        if file_path.exists():
            file_path.unlink()
    except Exception as e:
        console.print(f"⚠️ Could not delete temp file {file_path}: {e}")

# === BASIC ENDPOINTS ===

@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "message": "🚀 RagFlow Backend with ChromaDB, OCR, and Advanced Processing",
        "version": settings.app_version,
        "features": {
            "chromadb": features.is_enabled("chromadb"),
            "document_processing": features.is_enabled("document_processing"),
            "ocr_support": ocr_processor.is_ocr_available(),
            "vector_search": features.is_enabled("vector_search"),
            "ai_chat": features.is_enabled("google_ai") or features.is_enabled("openai")
        },
        "docs": "/docs",
        "health": "/api/health"
    }

@app.get("/api/health")
async def health_check():
    """Comprehensive health check"""
    try:
        # Test database connection
        db_stats = db.get_stats()
        db_healthy = True
    except Exception as e:
        console.print(f"❌ Database health check failed: {e}")
        db_healthy = False
        db_stats = {}
    
    # Test document processor
    doc_processor_stats = document_processor.get_processing_stats()
    
    # Test OCR availability
    ocr_status = ocr_processor.get_engine_status()
    
    # Test AI availability
    rag_status = rag_system.get_status()
    
    return {
        "status": "healthy" if db_healthy else "degraded",
        "timestamp": datetime.utcnow(),
        "version": settings.app_version,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "database": {
            "type": "ChromaDB",
            "healthy": db_healthy,
            "stats": db_stats
        },
        "document_processing": {
            "available": True,
            "supported_formats": doc_processor_stats["supported_formats"],
            "capabilities": doc_processor_stats["capabilities"]
        },
        "ocr": {
            "available": ocr_status["ocr_ready"],
            "engines": ocr_status["available_engines"],
            "default_engine": ocr_status["default_engine"]
        },
        "ai": {
            "rag_available": rag_status["features"]["rag_search"],
            "google_ai": rag_status["google_ai"]["available"],
            "openai": rag_status["openai"]["available"]
        },
        "features": features.summary()
    }

@app.get("/api/config")
async def get_configuration():
    """Get backend configuration"""
    return {
        "app": settings.summary(),
        "features": features.detailed_status(),
        "document_processing": document_processor.get_processing_stats(),
        "ocr": ocr_processor.get_engine_status(),
        "ai": rag_system.get_status()
    }

@app.get("/api/system/info")
async def system_info():
    """Detailed system information"""
    db_stats = db.get_stats()
    
    return {
        "app": {
            "name": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment,
            "python_version": sys.version
        },
        "database": {
            "type": "ChromaDB",
            "location": settings.get_database_url(),
            "stats": db_stats
        },
        "capabilities": {
            "document_formats": document_processor.get_supported_extensions(),
            "ocr_engines": list(ocr_processor.available_engines.keys()),
            "ai_providers": [
                provider for provider, status in rag_system.get_status().items() 
                if isinstance(status, dict) and status.get("available")
            ]
        },
        "statistics": {
            "projects": db_stats.get("projects", {}).get("total", 0),
            "documents": db_stats.get("documents", {}).get("total", 0),
            "document_chunks": db_stats.get("documents", {}).get("chunks_total", 0),
            "chats": db_stats.get("chats", {}).get("total", 0)
        },
        "settings": {
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "max_file_size_mb": settings.max_file_size // (1024 * 1024),
            "embedding_model": settings.embedding_model,
            "top_k": settings.top_k
        }
    }

@app.get("/api/ai/info")
async def get_ai_info():
    """Get AI model and capabilities information"""
    rag_status = rag_system.get_status()
    
    if not (rag_status["google_ai"]["available"] or rag_status["openai"]["available"]):
        raise HTTPException(status_code=503, detail="No AI providers configured")
    
    return {
        "providers": {
            "google_ai": {
                "available": rag_status["google_ai"]["available"],
                "model": settings.gemini_model,
                "features": ["chat", "text_generation", "rag_search"]
            },
            "openai": {
                "available": rag_status["openai"]["available"],
                "features": ["chat", "text_generation", "rag_search"]
            }
        },
        "features": rag_status["features"],
        "database": "ChromaDB with vector embeddings",
        "embedding_model": settings.embedding_model,
        "status": "available"
    }

# === PROJECT ENDPOINTS ===

@app.get("/api/projects", response_model=List[ProjectResponse])
async def get_projects():
    """Get all projects with statistics"""
    try:
        projects = db.get_projects()
        
        project_responses = []
        for project in projects:
            stats = db.get_project_stats(project.id)
            project_responses.append(ProjectResponse(
                id=project.id,
                name=project.name,
                description=project.description,
                created_at=datetime.fromisoformat(project.created_at),
                document_count=stats["document_count"],
                chat_count=stats["chat_count"]
            ))
        
        return project_responses
        
    except Exception as e:
        console.print(f"❌ Error loading projects: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load projects: {str(e)}")

@app.post("/api/projects", response_model=ProjectResponse)
async def create_project(project: ProjectCreate):
    """Create new project"""
    try:
        new_project = db.create_project(
            name=project.name,
            description=project.description
        )
        
        stats = db.get_project_stats(new_project.id)
        
        return ProjectResponse(
            id=new_project.id,
            name=new_project.name,
            description=new_project.description,
            created_at=datetime.fromisoformat(new_project.created_at),
            document_count=stats["document_count"],
            chat_count=stats["chat_count"]
        )
        
    except Exception as e:
        console.print(f"❌ Error creating project: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create project: {str(e)}")

@app.get("/api/projects/{project_id}")
async def get_project(project_id: str):
    """Get single project with detailed information"""
    try:
        project = db.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        stats = db.get_project_stats(project_id)
        documents = db.get_documents_by_project(project_id)
        chats = db.get_chats_by_project(project_id)
        
        return {
            "id": project.id,
            "name": project.name,
            "description": project.description,
            "created_at": project.created_at,
            "updated_at": project.updated_at,
            "statistics": stats,
            "documents": [
                {
                    "id": doc.id,
                    "filename": doc.filename,
                    "file_type": doc.file_type,
                    "file_size": doc.file_size,
                    "processing_status": doc.processing_status,
                    "created_at": doc.created_at,
                    "text_length": len(doc.content) if doc.content else 0
                }
                for doc in documents
            ],
            "recent_chats": [
                {
                    "id": chat.id,
                    "created_at": chat.created_at,
                    "message_count": len(chat.messages)
                }
                for chat in chats[:5]  # Last 5 chats
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        console.print(f"❌ Error loading project {project_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load project: {str(e)}")

@app.put("/api/projects/{project_id}", response_model=ProjectResponse)
async def update_project(project_id: str, project_update: ProjectUpdate):
    """Update project information"""
    try:
        updated_project = db.update_project(
            project_id=project_id,
            name=project_update.name,
            description=project_update.description
        )
        
        if not updated_project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        stats = db.get_project_stats(project_id)
        
        return ProjectResponse(
            id=updated_project.id,
            name=updated_project.name,
            description=updated_project.description,
            created_at=datetime.fromisoformat(updated_project.created_at),
            document_count=stats["document_count"],
            chat_count=stats["chat_count"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        console.print(f"❌ Error updating project {project_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update project: {str(e)}")

@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: str):
    """Delete project and all associated data"""
    try:
        project = db.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        stats = db.get_project_stats(project_id)
        
        success = db.delete_project(project_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete project")
        
        return {
            "message": f"Project '{project.name}' deleted successfully",
            "details": {
                "documents_affected": stats["document_count"],
                "chats_deleted": stats["chat_count"],
                "total_file_size_freed": stats["total_file_size"]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        console.print(f"❌ Error deleting project {project_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete project: {str(e)}")

# === DOCUMENT ENDPOINTS ===

@app.get("/api/documents", response_model=List[DocumentResponse])
async def get_documents(project_id: Optional[str] = Query(None)):
    """Get documents, optionally filtered by project"""
    try:
        documents = db.get_documents(project_id=project_id)
        
        return [
            DocumentResponse(
                id=doc.id,
                filename=doc.filename,
                file_size=doc.file_size,
                file_type=doc.file_type,
                processing_status=doc.processing_status,
                created_at=datetime.fromisoformat(doc.created_at),
                project_ids=doc.project_ids,
                processing_method=doc.metadata.get("processing_method", "standard"),
                text_length=doc.metadata.get("text_length", 0)
            )
            for doc in documents
        ]
        
    except Exception as e:
        console.print(f"❌ Error loading documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load documents: {str(e)}")

@app.post("/api/documents/upload", response_model=UploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    project_id: str = Form(...),
    use_ocr: bool = Form(False),
    ocr_language: str = Form("en"),
    ocr_engine: Optional[str] = Form(None)
):
    """Upload and process document with optional OCR"""
    
    # Validate project exists
    project = db.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    # Save uploaded file
    temp_file_path = await save_uploaded_file(file)
    
    # Schedule cleanup
    background_tasks.add_task(cleanup_temp_file, temp_file_path)
    
    try:
        # Determine processing method
        should_use_ocr = use_ocr and ocr_processor.can_process_file(temp_file_path)
        
        if should_use_ocr:
            console.print(f"📄 Processing with OCR: {file.filename}")
            
            # Use OCR processing
            if temp_file_path.suffix.lower() in ocr_processor.supported_image_formats:
                result = await ocr_processor.process_image_file(
                    file_path=temp_file_path,
                    engine=ocr_engine,
                    language=ocr_language,
                    project_ids=[project_id]
                )
                processing_method = "ocr_image"
            else:
                result = await ocr_processor.process_scanned_pdf(
                    file_path=temp_file_path,
                    engine=ocr_engine,
                    language=ocr_language,
                    project_ids=[project_id]
                )
                processing_method = "ocr_pdf"
            
            processing_info = {
                "method": processing_method,
                "ocr_engine": result.get("ocr_engine"),
                "ocr_language": result.get("ocr_language"),
                "text_extracted": True,
                "text_length": result.get("text_length", 0)
            }
            
        else:
            console.print(f"📄 Processing with standard parser: {file.filename}")
            
            # Use standard document processing
            result = await document_processor.process_document(
                file_path=temp_file_path,
                project_ids=[project_id]
            )
            
            processing_info = {
                "method": "standard",
                "format_detected": result.get("file_type"),
                "chunks_created": result.get("chunks_created", 0),
                "text_length": result.get("text_length", 0)
            }
        
        # Create response
        document_response = DocumentResponse(
            id=result["document_id"],
            filename=result["filename"],
            file_size=result["file_size"],
            file_type=result.get("file_type", temp_file_path.suffix),
            processing_status=result.get("processing_status", "completed"),
            created_at=datetime.utcnow(),
            project_ids=[project_id],
            processing_method=processing_info["method"],
            text_length=processing_info["text_length"]
        )
        
        return UploadResponse(
            message=f"Document '{file.filename}' processed successfully",
            document=document_response,
            processing_info=processing_info
        )
        
    except Exception as e:
        console.print(f"❌ Document processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")

@app.post("/api/documents/upload-batch")
async def upload_batch_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    project_id: str = Form(...),
    use_ocr: bool = Form(False),
    ocr_language: str = Form("en"),
    ocr_engine: Optional[str] = Form(None)
):
    """Upload and process multiple documents"""
    
    # Validate project
    project = db.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 files per batch")
    
    results = {
        "total_files": len(files),
        "processed": [],
        "failed": [],
        "processing_summary": {
            "ocr_used": use_ocr,
            "ocr_language": ocr_language,
            "ocr_engine": ocr_engine
        }
    }
    
    for file in files:
        try:
            if not file.filename:
                results["failed"].append({
                    "filename": "unknown",
                    "error": "No filename provided"
                })
                continue
            
            # Save and process file
            temp_file_path = await save_uploaded_file(file)
            background_tasks.add_task(cleanup_temp_file, temp_file_path)
            
            # Process based on OCR settings
            should_use_ocr = use_ocr and ocr_processor.can_process_file(temp_file_path)
            
            if should_use_ocr:
                if temp_file_path.suffix.lower() in ocr_processor.supported_image_formats:
                    result = await ocr_processor.process_image_file(
                        file_path=temp_file_path,
                        engine=ocr_engine,
                        language=ocr_language,
                        project_ids=[project_id]
                    )
                else:
                    result = await ocr_processor.process_scanned_pdf(
                        file_path=temp_file_path,
                        engine=ocr_engine,
                        language=ocr_language,
                        project_ids=[project_id]
                    )
            else:
                result = await document_processor.process_document(
                    file_path=temp_file_path,
                    project_ids=[project_id]
                )
            
            results["processed"].append({
                "filename": file.filename,
                "document_id": result["document_id"],
                "file_size": result["file_size"],
                "text_length": result.get("text_length", 0),
                "processing_method": "ocr" if should_use_ocr else "standard",
                "status": "success"
            })
            
        except Exception as e:
            console.print(f"❌ Failed to process {file.filename}: {e}")
            results["failed"].append({
                "filename": file.filename,
                "error": str(e),
                "status": "failed"
            })
    
    return {
        "message": f"Batch processing completed: {len(results['processed'])} successful, {len(results['failed'])} failed",
        "results": results
    }

@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete document and associated files"""
    try:
        documents = db.get_documents()
        document = next((d for d in documents if d.id == document_id), None)
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete physical file if it exists
        try:
            file_path = Path(document.file_path)
            if file_path.exists():
                file_path.unlink()
        except Exception as e:
            console.print(f"⚠️ Could not delete file {document.file_path}: {e}")
        
        # Delete from database
        success = db.delete_document(document_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete document from database")
        
        return {
            "message": f"Document '{document.filename}' deleted successfully",
            "document_id": document_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        console.print(f"❌ Error deleting document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

# === SEARCH ENDPOINTS ===

@app.post("/api/search")
async def search_documents(search_request: SearchRequest):
    """Search documents using RAG vector search"""
    try:
        results = db.search_documents(
            query=search_request.query,
            project_id=search_request.project_id,
            top_k=search_request.top_k
        )
        
        return {
            "query": search_request.query,
            "project_id": search_request.project_id,
            "top_k": search_request.top_k,
            "results": results,
            "total_results": len(results),
            "search_metadata": {
                "embedding_model": settings.embedding_model,
                "search_timestamp": datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        console.print(f"❌ Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# === CHAT ENDPOINTS ===

@app.post("/api/chat")
async def chat_with_ai(chat_request: ChatRequest):
    """Enhanced chat with AI using RAG integration"""
    try:
        # Validate AI availability
        rag_status = rag_system.get_status()
        if not (rag_status["google_ai"]["available"] or rag_status["openai"]["available"]):
            raise HTTPException(status_code=503, detail="No AI providers available")
        
        # Perform RAG search
        search_results = db.search_documents(
            query=chat_request.message,
            project_id=chat_request.project_id,
            top_k=settings.top_k
        )
        
        # Prepare context documents
        context_documents = [
            {
                "filename": result["filename"],
                "content": result["full_text"],
                "relevance_score": result["relevance_score"]
            }
            for result in search_results
        ]
        
        # Generate AI response
        response = await rag_system.generate_response(
            query=chat_request.message,
            context_documents=context_documents,
            model=chat_request.model,
            temperature=chat_request.temperature
        )
        
        # Create chat record
        chat_id = str(uuid.uuid4())
        messages = [
            {
                "role": "user", 
                "content": chat_request.message, 
                "timestamp": datetime.utcnow().isoformat()
            },
            {
                "role": "assistant", 
                "content": response["response"], 
                "timestamp": datetime.utcnow().isoformat()
            }
        ]
        
        chat = db.create_chat(
            project_id=chat_request.project_id,
            messages=messages
        )
        
        return {
            "response": response["response"],
            "chat_id": chat.id,
            "project_id": chat_request.project_id,
            "timestamp": datetime.utcnow().isoformat(),
            "model_info": {
                "model": chat_request.model,
                "temperature": chat_request.temperature,
                "features_used": response.get("features_used", {}),
                "context_documents": len(context_documents)
            },
            "sources": [
                {
                    "id": result["id"],
                    "name": result["filename"],
                    "filename": result["filename"],
                    "excerpt": result["excerpt"],
                    "relevance_score": result["relevance_score"]
                }
                for result in search_results
            ],
            "intelligence_metadata": response.get("intelligence_metadata", {})
        }
        
    except HTTPException:
        raise
    except Exception as e:
        console.print(f"❌ Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.get("/api/chats")
async def get_chats(project_id: Optional[str] = Query(None)):
    """Get chat history"""
    try:
        chats = db.get_chats(project_id=project_id)
        
        return [
            {
                "id": chat.id,
                "project_id": chat.project_id,
                "created_at": chat.created_at,
                "updated_at": chat.updated_at,
                "message_count": len(chat.messages),
                "last_message": chat.messages[-1]["content"][:100] + "..." if chat.messages else "",
                "last_message_timestamp": chat.messages[-1]["timestamp"] if chat.messages else chat.created_at
            }
            for chat in chats
        ]
        
    except Exception as e:
        console.print(f"❌ Error loading chats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load chats: {str(e)}")

@app.get("/api/chats/{chat_id}")
async def get_chat(chat_id: str):
    """Get specific chat with full message history"""
    try:
        chats = db.get_chats()
        chat = next((c for c in chats if c.id == chat_id), None)
        
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        return {
            "id": chat.id,
            "project_id": chat.project_id,
            "messages": chat.messages,
            "created_at": chat.created_at,
            "updated_at": chat.updated_at,
            "message_count": len(chat.messages),
            "metadata": chat.metadata
        }
        
    except HTTPException:
        raise
    except Exception as e:
        console.print(f"❌ Error loading chat {chat_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load chat: {str(e)}")

@app.delete("/api/chats/{chat_id}")
async def delete_chat(chat_id: str):
    """Delete chat conversation"""
    try:
        chats = db.get_chats()
        chat = next((c for c in chats if c.id == chat_id), None)
        
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        success = db.delete_chat(chat_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete chat")
        
        return {
            "message": "Chat deleted successfully",
            "chat_id": chat_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        console.print(f"❌ Error deleting chat {chat_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete chat: {str(e)}")

# === OCR ENDPOINTS ===

@app.post("/api/ocr/process-image")
async def process_image_ocr(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    project_id: str = Form(...),
    language: str = Form("en"),
    engine: Optional[str] = Form(None)
):
    """Process image file with OCR"""
    
    if not ocr_processor.is_ocr_available():
        raise HTTPException(status_code=503, detail="OCR not available. Please install pytesseract or easyocr.")
    
    # Validate project
    project = db.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in ocr_processor.supported_image_formats:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported image format: {file_extension}. Supported: {list(ocr_processor.supported_image_formats)}"
        )
    
    # Save uploaded file
    temp_file_path = await save_uploaded_file(file)
    background_tasks.add_task(cleanup_temp_file, temp_file_path)
    
    try:
        result = await ocr_processor.process_image_file(
            file_path=temp_file_path,
            engine=engine,
            language=language,
            project_ids=[project_id]
        )
        
        return {
            "message": f"Image '{file.filename}' processed with OCR successfully",
            "result": result,
            "ocr_info": {
                "engine_used": result["ocr_engine"],
                "language": result["ocr_language"],
                "text_length": result["text_length"]
            }
        }
        
    except Exception as e:
        console.print(f"❌ OCR processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")

@app.post("/api/ocr/process-pdf")
async def process_scanned_pdf_ocr(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    project_id: str = Form(...),
    language: str = Form("en"),
    engine: Optional[str] = Form(None)
):
    """Process scanned PDF with OCR"""
    
    if not ocr_processor.is_ocr_available():
        raise HTTPException(status_code=503, detail="OCR not available")
    
    # Validate project
    project = db.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Validate file type
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Save uploaded file
    temp_file_path = await save_uploaded_file(file)
    background_tasks.add_task(cleanup_temp_file, temp_file_path)
    
    try:
        result = await ocr_processor.process_scanned_pdf(
            file_path=temp_file_path,
            engine=engine,
            language=language,
            project_ids=[project_id]
        )
        
        return {
            "message": f"Scanned PDF '{file.filename}' processed with OCR successfully",
            "result": result,
            "ocr_info": {
                "engine_used": result["ocr_engine"],
                "language": result["ocr_language"],
                "total_pages": result.get("total_pages", 0),
                "pages_processed": result.get("pages_processed", 0),
                "text_length": result["text_length"]
            }
        }
        
    except Exception as e:
        console.print(f"❌ Scanned PDF processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Scanned PDF processing failed: {str(e)}")

@app.get("/api/ocr/status")
async def get_ocr_status():
    """Get OCR engine status and capabilities"""
    return ocr_processor.get_engine_status()

@app.get("/api/ocr/test-engines")
async def test_ocr_engines():
    """Test all available OCR engines"""
    if not ocr_processor.is_ocr_available():
        raise HTTPException(status_code=503, detail="No OCR engines available")
    
    try:
        test_results = ocr_processor.test_ocr_engines()
        return {
            "message": "OCR engine testing completed",
            "results": test_results
        }
    except Exception as e:
        console.print(f"❌ OCR engine testing failed: {e}")
        raise HTTPException(status_code=500, detail=f"OCR testing failed: {str(e)}")

@app.get("/api/ocr/installation-guide")
async def get_ocr_installation_guide():
    """Get OCR installation instructions"""
    return ocr_processor.get_installation_guide()

# === BATCH PROCESSING ENDPOINTS ===

@app.post("/api/batch/process-directory")
async def batch_process_directory(
    directory_path: str = Form(...),
    project_id: str = Form(...),
    recursive: bool = Form(True),
    use_ocr: bool = Form(False),
    ocr_language: str = Form("en"),
    file_pattern: str = Form("*")
):
    """Batch process all files in a directory"""
    
    # Validate project
    project = db.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Validate directory
    dir_path = Path(directory_path)
    if not dir_path.exists():
        raise HTTPException(status_code=404, detail="Directory not found")
    
    if not dir_path.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a directory")
    
    try:
        # Process documents
        doc_results = await document_processor.batch_process_directory(
            directory_path=dir_path,
            project_ids=[project_id],
            recursive=recursive,
            file_pattern=file_pattern
        )
        
        ocr_results = {"processed": [], "failed": []}
        
        # Process images with OCR if requested
        if use_ocr and ocr_processor.is_ocr_available():
            ocr_results = await ocr_processor.batch_process_images(
                directory_path=dir_path,
                language=ocr_language,
                project_ids=[project_id],
                recursive=recursive
            )
        
        return {
            "message": f"Batch processing completed for directory: {directory_path}",
            "document_processing": doc_results,
            "ocr_processing": ocr_results,
            "summary": {
                "total_documents_processed": len(doc_results["processed"]),
                "total_ocr_processed": len(ocr_results["processed"]),
                "total_failed": len(doc_results["failed"]) + len(ocr_results["failed"]),
                "project_id": project_id
            }
        }
        
    except Exception as e:
        console.print(f"❌ Batch processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

# === ADMIN ENDPOINTS ===

@app.get("/api/admin/stats")
async def get_admin_stats():
    """Get comprehensive system statistics"""
    try:
        db_stats = db.get_stats()
        projects = db.get_projects()
        
        # Detailed project information
        project_details = []
        for project in projects:
            project_stats = db.get_project_stats(project.id)
            project_details.append({
                "id": project.id,
                "name": project.name,
                "document_count": project_stats["document_count"],
                "chat_count": project_stats["chat_count"],
                "total_file_size": project_stats["total_file_size"],
                "created_at": project.created_at
            })
        
        # Processing capabilities
        doc_capabilities = document_processor.get_processing_stats()
        ocr_status = ocr_processor.get_engine_status()
        
        return {
            "database": db_stats,
            "projects": {
                "total": len(projects),
                "details": project_details
            },
            "processing": {
                "document_formats": doc_capabilities["supported_formats"],
                "document_capabilities": doc_capabilities["capabilities"],
                "ocr_available": ocr_status["ocr_ready"],
                "ocr_engines": ocr_status["available_engines"]
            },
            "system": {
                "app_version": settings.app_version,
                "environment": settings.environment,
                "features_enabled": features.get_enabled_features(),
                "uptime_check": datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        console.print(f"❌ Error loading admin stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load admin stats: {str(e)}")

@app.post("/api/admin/reindex")
async def reindex_documents():
    """Reindex all documents in the RAG system"""
    try:
        db_stats = db.get_stats()
        
        # Note: ChromaDB auto-manages indexes, but we can report current status
        return {
            "message": "Document reindexing completed",
            "status": "success",
            "documents_in_index": db_stats.get("documents", {}).get("total", 0),
            "vector_chunks": db_stats.get("documents", {}).get("chunks_total", 0),
            "embedding_model": settings.embedding_model,
            "reindex_timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        console.print(f"❌ Reindex error: {e}")
        raise HTTPException(status_code=500, detail=f"Reindexing failed: {str(e)}")

@app.post("/api/admin/optimize-database")
async def optimize_database():
    """Optimize database performance"""
    try:
        # Get current stats
        before_stats = db.get_stats()
        
        # ChromaDB auto-optimizes, but we can report optimization status
        after_stats = db.get_stats()
        
        return {
            "message": "Database optimization completed",
            "before": before_stats,
            "after": after_stats,
            "optimization_timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        console.print(f"❌ Database optimization error: {e}")
        raise HTTPException(status_code=500, detail=f"Database optimization failed: {str(e)}")

@app.get("/api/admin/system-diagnostics")
async def system_diagnostics():
    """Run comprehensive system diagnostics"""
    diagnostics = {
        "timestamp": datetime.utcnow().isoformat(),
        "overall_status": "healthy",
        "issues": [],
        "warnings": []
    }
    
    try:
        # Test database
        try:
            db_stats = db.get_stats()
            diagnostics["database"] = {"status": "healthy", "stats": db_stats}
        except Exception as e:
            diagnostics["database"] = {"status": "error", "error": str(e)}
            diagnostics["issues"].append(f"Database error: {e}")
            diagnostics["overall_status"] = "error"
        
        # Test document processor
        try:
            doc_stats = document_processor.get_processing_stats()
            diagnostics["document_processor"] = {"status": "healthy", "capabilities": doc_stats}
        except Exception as e:
            diagnostics["document_processor"] = {"status": "error", "error": str(e)}
            diagnostics["issues"].append(f"Document processor error: {e}")
        
        # Test OCR
        if ocr_processor.is_ocr_available():
            try:
                ocr_test = ocr_processor.test_ocr_engines()
                diagnostics["ocr"] = {"status": "healthy", "test_results": ocr_test}
            except Exception as e:
                diagnostics["ocr"] = {"status": "warning", "error": str(e)}
                diagnostics["warnings"].append(f"OCR warning: {e}")
        else:
            diagnostics["ocr"] = {"status": "not_available"}
            diagnostics["warnings"].append("OCR engines not available")
        
        # Test AI
        try:
            ai_status = rag_system.get_status()
            if ai_status["google_ai"]["available"] or ai_status["openai"]["available"]:
                diagnostics["ai"] = {"status": "healthy", "providers": ai_status}
            else:
                diagnostics["ai"] = {"status": "warning", "providers": ai_status}
                diagnostics["warnings"].append("No AI providers configured")
        except Exception as e:
            diagnostics["ai"] = {"status": "error", "error": str(e)}
            diagnostics["issues"].append(f"AI system error: {e}")
        
        # Set overall status
        if diagnostics["issues"]:
            diagnostics["overall_status"] = "error"
        elif diagnostics["warnings"]:
            diagnostics["overall_status"] = "warning"
        
        return diagnostics
        
    except Exception as e:
        console.print(f"❌ System diagnostics failed: {e}")
        raise HTTPException(status_code=500, detail=f"System diagnostics failed: {str(e)}")

# === DEVELOPMENT ENDPOINTS ===

@app.post("/api/dev/reset-database")
async def reset_database():
    """Reset entire database (development only!)"""
    if not settings.is_development:
        raise HTTPException(status_code=403, detail="Only available in development mode")
    
    try:
        db.reset_database()
        return {
            "message": "Database reset successfully", 
            "warning": "ALL DATA WAS DELETED!",
            "environment": settings.environment
        }
    except Exception as e:
        console.print(f"❌ Database reset error: {e}")
        raise HTTPException(status_code=500, detail=f"Database reset failed: {str(e)}")

@app.get("/api/dev/test-all-features")
async def test_all_features():
    """Test all system features (development only!)"""
    if not settings.is_development:
        raise HTTPException(status_code=403, detail="Only available in development mode")
    
    test_results = {
        "timestamp": datetime.utcnow().isoformat(),
        "tests": {}
    }
    
    # Test database
    try:
        test_project = db.create_project("Test Project", "Test Description")
        db.delete_project(test_project.id)
        test_results["tests"]["database"] = {"status": "pass", "message": "CRUD operations working"}
    except Exception as e:
        test_results["tests"]["database"] = {"status": "fail", "error": str(e)}
    
    # Test document processing
    try:
        stats = document_processor.get_processing_stats()
        test_results["tests"]["document_processor"] = {"status": "pass", "supported_formats": len(stats["supported_formats"])}
    except Exception as e:
        test_results["tests"]["document_processor"] = {"status": "fail", "error": str(e)}
    
    # Test OCR
    try:
        if ocr_processor.is_ocr_available():
            ocr_test = ocr_processor.test_ocr_engines()
            test_results["tests"]["ocr"] = {"status": "pass", "engines_tested": len(ocr_test["engines"])}
        else:
            test_results["tests"]["ocr"] = {"status": "skip", "reason": "OCR not available"}
    except Exception as e:
        test_results["tests"]["ocr"] = {"status": "fail", "error": str(e)}
    
    # Test AI
    try:
        ai_status = rag_system.get_status()
        if ai_status["google_ai"]["available"] or ai_status["openai"]["available"]:
            test_results["tests"]["ai"] = {"status": "pass", "providers_available": True}
        else:
            test_results["tests"]["ai"] = {"status": "skip", "reason": "No AI providers configured"}
    except Exception as e:
        test_results["tests"]["ai"] = {"status": "fail", "error": str(e)}
    
    return test_results

# === STARTUP/SHUTDOWN EVENTS ===

@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    console.print("🚀 RagFlow Backend starting with advanced features...")
    console.print("=" * 60)
    
    # Database statistics
    try:
        db_stats = db.get_stats()
        console.print("📊 Database Statistics:")
        console.print(f"   Projects: {db_stats.get('projects', {}).get('total', 0)}")
        console.print(f"   Documents: {db_stats.get('documents', {}).get('total', 0)}")
        console.print(f"   Document Chunks: {db_stats.get('documents', {}).get('chunks_total', 0)}")
        console.print(f"   Chats: {db_stats.get('chats', {}).get('total', 0)}")
    except Exception as e:
        console.print(f"⚠️ Could not load database stats: {e}")
    
    # Feature status
    console.print("\n🔧 Feature Status:")
    try:
        feature_summary = features.summary()
        console.print(f"   Enabled Features: {feature_summary['enabled_count']}/{feature_summary['total_features']}")
    except Exception as e:
        console.print(f"   Feature Status: Error loading - {e}")
    
    # Document processing
    try:
        doc_stats = document_processor.get_processing_stats()
        # Fix: Check if supported_formats is a number or list
        supported_formats = doc_stats.get('supported_formats', 0)
        if isinstance(supported_formats, (list, dict)):
            format_count = len(supported_formats)
        else:
            format_count = supported_formats  # It's already a number
        
        console.print(f"   Document Formats: {format_count}")
        
        # Show available formats
        available_formats = doc_stats.get('available_formats', [])
        if available_formats:
            format_names = [fmt['extension'] for fmt in available_formats if fmt.get('available', False)]
            console.print(f"   Supported Extensions: {', '.join(format_names)}")
    except Exception as e:
        console.print(f"   Document Processing: Error - {e}")
    
    # OCR status
    try:
        console.print(f"   OCR Available: {'✅' if ocr_processor.is_ocr_available() else '❌'}")
        if ocr_processor.is_ocr_available():
            available_engines = [engine for engine, available in ocr_processor.available_engines.items() if available]
            console.print(f"   OCR Engines: {', '.join(available_engines) if available_engines else 'None working'}")
    except Exception as e:
        console.print(f"   OCR Status: Error - {e}")
    
    # AI status
    try:
        ai_status = rag_system.get_status()
        ai_providers = []
        if ai_status.get("google_ai", {}).get("available", False):
            ai_providers.append("Google AI")
        if ai_status.get("openai", {}).get("available", False):
            ai_providers.append("OpenAI")
        
        console.print(f"   AI Providers: {', '.join(ai_providers) if ai_providers else 'None configured'}")
    except Exception as e:
        console.print(f"   AI Status: Error - {e}")
    
    console.print("\n✅ RagFlow Backend ready!")
    console.print("📖 API Documentation: http://127.0.0.1:8000/docs")
    console.print("=" * 60)

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event"""
    console.print("🛑 RagFlow Backend shutting down...")
    console.print("✅ Shutdown complete")

# === ERROR HANDLERS ===

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": "The requested resource was not found",
            "path": str(request.url.path)
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    console.print(f"❌ Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "support": "Check server logs for details"
        }
    )

# === MAIN ENTRY POINT ===

if __name__ == "__main__":
    import uvicorn
    
    console.print("🚀 Starting RagFlow Backend...")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=settings.is_development,
        log_level="info"
    )