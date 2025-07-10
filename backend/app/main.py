#!/usr/bin/env python3
"""
RAG-Enhanced RagFlow Backend - Mit vollständiger Dokumentenanalyse
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import os
import json
from pathlib import Path
from datetime import datetime
import uuid
import asyncio
from dotenv import load_dotenv
from document_processor import process_uploaded_document, RAGChatEnhancer, DocumentProcessor


# Load environment variables
load_dotenv()

# Create directories
Path("uploads").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)

# Simple data storage (in-memory with file persistence)
projects_db = {}
documents_db = {}
chats_db = {}

# === Pydantic Models ===
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    project_id: Optional[str] = None

class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None

class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None

# === FastAPI App ===
app = FastAPI(
    title="RagFlow Backend",
    description="AI-powered document analysis mit RAG-Integration",
    version="2.2.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Helper Functions ===
async def get_gemini_service():
    """Get configured Gemini service"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "your_google_api_key_here":
        return None
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        print(f"❌ Failed to initialize Gemini: {e}")
        return None

def get_project_context(project_id: str, projects_db: dict, documents_db: dict) -> str:
    """Erstelle projektbezogenen Kontext für das LLM"""
    if not project_id or project_id not in projects_db:
        return ""
    
    project = projects_db[project_id]
    project_name = project.get("name", "Unbenanntes Projekt")
    project_description = project.get("description", "")
    
    # Sammle Dokument-Informationen mit Verarbeitungsstatus
    project_docs = [
        doc for doc in documents_db.values() 
        if project_id in doc.get("project_ids", [])
    ]
    
    # Baue den Kontext auf
    context_parts = [
        f"Du hilfst im Projekt '{project_name}'."
    ]
    
    if project_description:
        context_parts.append(f"Projektbeschreibung: {project_description}")
    
    if project_docs:
        processed_docs = [doc for doc in project_docs if doc.get("processing_status") == "completed"]
        context_parts.append(f"Das Projekt enthält {len(project_docs)} Dokument(e), davon {len(processed_docs)} verarbeitet:")
        
        for doc in project_docs[:3]:  # Zeige max. 3 Dokumente
            doc_name = doc.get("filename", "Unbekanntes Dokument")
            doc_type = doc.get("file_type", "").upper()
            status = doc.get("processing_status", "uploaded")
            status_emoji = "✅" if status == "completed" else "🔄" if status == "processing" else "❌" if status == "failed" else "📄"
            context_parts.append(f"- {status_emoji} {doc_name} ({doc_type})")
        
        if len(project_docs) > 3:
            context_parts.append(f"- ... und {len(project_docs) - 3} weitere Dokumente")
    else:
        context_parts.append("Das Projekt enthält noch keine Dokumente.")
    
    return " ".join(context_parts)

def get_enhanced_system_prompt(project_context: str = "") -> str:
    """Erstelle einen erweiterten System-Prompt"""
    base_prompt = """Du bist ein hilfsreicher AI-Assistent für RagFlow, eine Dokumentenanalyse-App. 

Du bist spezialisiert auf:
- Analyse und Extraktion von Informationen aus Dokumenten
- Beantwortung von Fragen zu hochgeladenen Dateien
- Zusammenfassung und Strukturierung von Inhalten
- Hilfe bei der Dokumentenverwaltung

Antworte immer:
- Präzise und hilfreich
- Auf Deutsch (außer wenn explizit anders gewünscht)
- Projektbezogen und kontextbewusst
- Mit konkreten Handlungsempfehlungen wenn möglich

Wenn du auf Dokumenteninhalte verweist, gib immer den Dateinamen an und zitiere relevante Passagen."""

    if project_context:
        base_prompt += f"\n\nAktueller Projektkontext: {project_context}"
        base_prompt += "\n\nBeziehe dich in deinen Antworten auf diesen Projektkontext und die verfügbaren Dokumente."
    
    return base_prompt

# === API Routes ===

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "🚀 RagFlow Backend läuft (RAG-Enhanced Version)!",
        "version": "2.2.0",
        "features": [
            "✅ Vollständige RAG-Integration",
            "✅ PDF/DOCX/TXT Textextraktion",
            "✅ Intelligente Dokumentensuche", 
            "✅ Projektbezogener AI-Kontext",
            "✅ Chunk-basierte Analyse"
        ],
        "endpoints": {
            "health": "/api/health",
            "chat": "/api/v1/chat",
            "projects": "/api/v1/projects",
            "upload": "/api/v1/upload/documents",
            "docs": "/docs"
        }
    }

@app.get("/api/health")
async def health_check():
    """Basic health check"""
    gemini = await get_gemini_service()
    
    # Zähle verarbeitete Dokumente
    processed_docs = sum(1 for doc in documents_db.values() if doc.get("processing_status") == "completed")
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.2.0",
        "services": {
            "gemini_ai": "connected" if gemini else "disconnected",
            "database": "operational",
            "file_storage": "operational",
            "document_processor": "operational"
        },
        "stats": {
            "projects": len(projects_db),
            "documents": len(documents_db),
            "processed_documents": processed_docs,
            "chats": len(chats_db)
        }
    }

@app.get("/api/health/detailed")
async def detailed_health():
    """Detailed health information"""
    gemini = await get_gemini_service()
    api_key = os.getenv("GOOGLE_API_KEY")
    
    # Dokumentenstatistiken
    doc_stats = {
        "total": len(documents_db),
        "uploaded": sum(1 for doc in documents_db.values() if doc.get("processing_status") == "uploaded"),
        "processing": sum(1 for doc in documents_db.values() if doc.get("processing_status") == "processing"),
        "completed": sum(1 for doc in documents_db.values() if doc.get("processing_status") == "completed"),
        "failed": sum(1 for doc in documents_db.values() if doc.get("processing_status") == "failed")
    }
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.2.0",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "services": {
            "gemini_ai": {
                "status": "connected" if gemini else "disconnected",
                "api_key_configured": bool(api_key and api_key != "your_google_api_key_here"),
                "model": "gemini-1.5-flash"
            },
            "database": {
                "status": "operational",
                "type": "in-memory",
                "persistence": "file-based"
            },
            "file_storage": {
                "status": "operational",
                "upload_path": str(Path("uploads").absolute()),
                "data_path": str(Path("data").absolute())
            },
            "document_processor": {
                "status": "operational",
                "supported_formats": ["pdf", "docx", "txt", "md"],
                "libraries": ["pdfplumber", "PyPDF2", "python-docx"]
            }
        },
        "stats": {
            "projects": len(projects_db),
            "documents": doc_stats,
            "chats": len(chats_db)
        },
        "system": {
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
            "fastapi_version": "0.104.1",
            "environment": os.getenv("ENVIRONMENT", "development")
        }
    }

# === Chat Endpoints ===

@app.post("/api/v1/chat/test")
async def test_gemini():
    """Test Gemini API connection"""
    gemini = await get_gemini_service()
    
    if not gemini:
        return {
            "status": "error",
            "message": "Google API Key nicht konfiguriert oder ungültig",
            "hint": "Setze GOOGLE_API_KEY in der .env Datei"
        }
    
    try:
        response = gemini.generate_content("Antworte nur mit: 'RagFlow Backend funktioniert perfekt mit RAG!'")
        
        return {
            "status": "success",
            "message": "✅ Gemini API funktioniert einwandfrei!",
            "gemini_response": response.text if response else "Keine Antwort",
            "timestamp": datetime.utcnow().isoformat(),
            "model": "gemini-1.5-flash"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Gemini API Fehler: {str(e)}",
            "hint": "Überprüfe deinen API-Schlüssel unter https://ai.google.dev"
        }

@app.post("/api/v1/chat")
async def chat_endpoint(request: ChatRequest):
    """Chat with Gemini AI - RAG-Enhanced Version"""
    try:
        gemini = await get_gemini_service()
        
        if not gemini:
            return {
                "response": "⚠️ Google API Key nicht konfiguriert. Setze GOOGLE_API_KEY in der .env Datei.",
                "status": "error",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        if not request.messages:
            raise HTTPException(status_code=400, detail="Keine Nachrichten übermittelt")
        
        # Hole die letzte Benutzernachricht
        user_message = None
        for msg in reversed(request.messages):
            if msg.role == "user":
                user_message = msg.content
                break
        
        if not user_message:
            raise HTTPException(status_code=400, detail="Keine Benutzernachricht gefunden")
        
        # Erstelle projektbezogenen Kontext
        project_context = ""
        project_name = "einem Projekt"
        rag_sources = []
        
        if request.project_id:
            project_context = get_project_context(request.project_id, projects_db, documents_db)
            if request.project_id in projects_db:
                project_name = f"dem Projekt '{projects_db[request.project_id].get('name', 'Unbenannt')}'"
        
        # *** RAG-Enhancement: Suche relevante Dokumenteninhalte ***
        enhanced_prompt = get_enhanced_system_prompt(project_context)
        
        if request.project_id:
            try:
                rag_enhancer = RAGChatEnhancer(documents_db)
                enhanced_prompt = rag_enhancer.enhance_prompt_with_context(
                    user_message, 
                    request.project_id, 
                    enhanced_prompt
                )
                
                # Sammle Quellen für die Antwort
                relevant_content = rag_enhancer.find_relevant_content(user_message, request.project_id)
                rag_sources = [
                    {
                        "type": "document_chunk",
                        "content": content[:200] + "...",  # Kurze Vorschau
                        "relevance": "high"
                    }
                    for content in relevant_content[:3]
                ]
                
                print(f"🔍 RAG: {len(relevant_content)} relevante Inhalte gefunden")
                
            except Exception as rag_error:
                print(f"⚠️ RAG Fehler: {rag_error}")
                # Fallback ohne RAG
        
        # Baue die vollständige Prompt zusammen
        conversation_history = ""
        if len(request.messages) > 1:
            # Füge Gesprächsverlauf hinzu (außer der letzten Nachricht)
            for msg in request.messages[:-1]:
                role_german = "Benutzer" if msg.role == "user" else "Assistent"
                conversation_history += f"{role_german}: {msg.content}\n"
        
        full_prompt = f"""{enhanced_prompt}

{conversation_history}

Benutzer: {user_message}"""
        
        # Debug: Logge die Prompt-Länge
        print(f"📝 Prompt-Länge: {len(full_prompt)} Zeichen")
        
        # Generiere Antwort
        response = gemini.generate_content(full_prompt)
        
        # Verarbeite die Antwort
        ai_response = response.text if response else "Entschuldigung, ich konnte keine Antwort generieren."
        
        # Speichere Chat mit RAG-Metadaten
        chat_id = str(uuid.uuid4())
        chats_db[chat_id] = {
            "id": chat_id,
            "project_id": request.project_id,
            "project_name": projects_db.get(request.project_id, {}).get("name", "") if request.project_id else "",
            "messages": [msg.dict() for msg in request.messages],
            "response": ai_response,
            "timestamp": datetime.utcnow().isoformat(),
            "model": "gemini-1.5-flash",
            "context_used": bool(project_context),
            "rag_enhanced": len(rag_sources) > 0,
            "rag_sources_count": len(rag_sources)
        }
        
        # Update Chat-Zähler des Projekts
        if request.project_id and request.project_id in projects_db:
            projects_db[request.project_id]["chat_count"] = projects_db[request.project_id].get("chat_count", 0) + 1
            projects_db[request.project_id]["updated_at"] = datetime.utcnow().isoformat()
        
        return {
            "response": ai_response,
            "chat_id": chat_id,
            "project_id": request.project_id,
            "project_name": project_name,
            "timestamp": datetime.utcnow().isoformat(),
            "model_info": {
                "model": "gemini-1.5-flash",
                "temperature": 0.7,
                "context_enhanced": bool(project_context),
                "rag_enhanced": len(rag_sources) > 0
            },
            "sources": rag_sources
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Chat Fehler: {str(e)}")
        return {
            "response": f"Fehler bei der Chat-Anfrage: {str(e)}",
            "status": "error",
            "timestamp": datetime.utcnow().isoformat()
        }

@app.get("/api/v1/chat/models")
async def get_models():
    """Get available AI models"""
    return {
        "models": [
            {
                "id": "gemini-1.5-flash",
                "name": "Gemini 1.5 Flash",
                "description": "Schnelles und effizientes Modell für die meisten Aufgaben mit RAG-Unterstützung",
                "max_tokens": 1024,
                "capabilities": ["text-generation", "conversation", "document-analysis", "rag-enhanced"]
            },
            {
                "id": "gemini-1.5-pro", 
                "name": "Gemini 1.5 Pro",
                "description": "Erweiterte Modell für komplexe Analysen mit RAG-Unterstützung",
                "max_tokens": 2048,
                "capabilities": ["text-generation", "conversation", "document-analysis", "complex-reasoning", "rag-enhanced"]
            }
        ],
        "default": "gemini-1.5-flash"
    }

# === Project Endpoints ===

@app.post("/api/v1/projects/")
async def create_project(project: ProjectCreate):
    """Create a new project"""
    try:
        project_id = str(uuid.uuid4())
        
        new_project = {
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
        
        projects_db[project_id] = new_project
        
        return new_project
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create project: {str(e)}")

@app.get("/api/v1/projects/")
async def get_projects(skip: int = 0, limit: int = 10, search: Optional[str] = None):
    """Get all projects"""
    projects = list(projects_db.values())
    
    # Simple search
    if search:
        search_lower = search.lower()
        projects = [
            p for p in projects 
            if search_lower in p["name"].lower() or 
               (p.get("description") and search_lower in p["description"].lower())
        ]
    
    # Simple pagination
    total = len(projects)
    projects = projects[skip:skip+limit]
    
    return {
        "projects": projects,
        "total": total,
        "skip": skip,
        "limit": limit
    }

@app.get("/api/v1/projects/{project_id}")
async def get_project(project_id: str):
    """Get specific project with documents"""
    if project_id not in projects_db:
        raise HTTPException(status_code=404, detail="Project not found")
    
    project = projects_db[project_id].copy()
    
    # Add associated documents
    project_docs = [
        doc for doc in documents_db.values() 
        if project_id in doc.get("project_ids", [])
    ]
    
    project["documents"] = project_docs
    return project

@app.put("/api/v1/projects/{project_id}")
async def update_project(project_id: str, project_update: ProjectUpdate):
    """Update a project"""
    if project_id not in projects_db:
        raise HTTPException(status_code=404, detail="Project not found")
    
    project = projects_db[project_id]
    
    # Update fields
    update_data = project_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        project[field] = value
    
    project["updated_at"] = datetime.utcnow().isoformat()
    
    return project

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
    
    return {"message": f"Project '{deleted_project['name']}' deleted", "project": deleted_project}

# === Document Endpoints ===

@app.post("/api/v1/upload/documents")
async def upload_documents(
    files: List[UploadFile] = File(...),
    project_id: Optional[str] = Form(None),
    tags: Optional[str] = Form(None)
):
    """Upload one or more documents mit automatischer RAG-Verarbeitung"""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Check project exists
    if project_id and project_id not in projects_db:
        raise HTTPException(status_code=404, detail="Project not found")
    
    uploaded_documents = []
    
    for file in files:
        if not file.filename:
            continue
            
        # Validate file type
        allowed_extensions = {'.pdf', '.docx', '.txt', '.md'}
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"File type {file_extension} not supported. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Generate unique filename
        doc_id = str(uuid.uuid4())
        safe_filename = f"{doc_id}_{file.filename}"
        file_path = Path("uploads") / safe_filename
        
        try:
            # Save file
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)
            
            # Create document metadata
            document = {
                "id": doc_id,
                "filename": file.filename,
                "safe_filename": safe_filename,
                "file_path": str(file_path),
                "file_type": file_extension[1:],  # Remove dot
                "file_size": len(content),
                "uploaded_at": datetime.utcnow().isoformat(),
                "processing_status": "uploaded",  # Will be updated by background task
                "project_ids": [project_id] if project_id else [],
                "tags": tags.split(",") if tags else [],
                "content_preview": None,
                "extracted_text": "",
                "text_chunks": [],
                "text_metadata": {},
                "processing_error": None,
                "processed_at": None
            }
            
            documents_db[doc_id] = document
            uploaded_documents.append(document)
            
            # *** RAG-Integration: Starte Hintergrundverarbeitung ***
            print(f"📄 Starte Verarbeitung von {file.filename}...")
            asyncio.create_task(
                process_uploaded_document(
                    file_path=str(file_path),
                    file_type=file_extension[1:],
                    document_id=doc_id,
                    documents_db=documents_db
                )
            )
            
            # Update project document count
            if project_id:
                project = projects_db[project_id]
                project["document_ids"].append(doc_id)
                project["document_count"] = len(project["document_ids"])
                project["updated_at"] = datetime.utcnow().isoformat()
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to upload {file.filename}: {str(e)}")
    
    return {
        "message": f"Successfully uploaded {len(uploaded_documents)} document(s). Processing started in background.",
        "documents": uploaded_documents,
        "processing_note": "Documents are being processed for text extraction. Check status via document endpoints."
    }

@app.get("/api/v1/documents/")
async def get_documents(skip: int = 0, limit: int = 10, project_id: Optional[str] = None):
    """Get all documents, optionally filtered by project"""
    documents = list(documents_db.values())
    
    # Filter by project if specified
    if project_id:
        documents = [
            doc for doc in documents 
            if project_id in doc.get("project_ids", [])
        ]
    
    # Simple pagination
    total = len(documents)
    documents = documents[skip:skip+limit]
    
    return {
        "documents": documents,
        "total": total,
        "skip": skip,
        "limit": limit
    }

@app.get("/api/v1/documents/{document_id}")
async def get_document(document_id: str):
    """Get specific document"""
    if document_id not in documents_db:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return documents_db[document_id]

@app.get("/api/v1/documents/{document_id}/status")
async def get_document_processing_status(document_id: str):
    """Prüfe den Verarbeitungsstatus eines Dokuments"""
    if document_id not in documents_db:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc = documents_db[document_id]
    
    return {
        "document_id": document_id,
        "filename": doc.get("filename"),
        "processing_status": doc.get("processing_status", "uploaded"),
        "processing_error": doc.get("processing_error"),
        "text_extracted": bool(doc.get("extracted_text")),
        "chunk_count": len(doc.get("text_chunks", [])),
        "word_count": doc.get("text_metadata", {}).get("word_count", 0),
        "processed_at": doc.get("processed_at")
    }

@app.get("/api/v1/documents/{document_id}/content")
async def get_document_content(document_id: str, chunk_id: Optional[str] = None):
    """Hole den extrahierten Inhalt eines Dokuments"""
    if document_id not in documents_db:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc = documents_db[document_id]
    
    if doc.get("processing_status") != "completed":
        raise HTTPException(status_code=400, detail="Document not yet processed")
    
    if chunk_id:
        # Spezifischer Chunk
        chunks = doc.get("text_chunks", [])
        chunk = next((c for c in chunks if c.get("id") == chunk_id), None)
        if not chunk:
            raise HTTPException(status_code=404, detail="Chunk not found")
        return {"chunk": chunk}
    else:
        # Vollständiger Text
        return {
            "document_id": document_id,
            "filename": doc.get("filename"),
            "full_text": doc.get("extracted_text"),
            "chunks": doc.get("text_chunks", []),
            "metadata": doc.get("text_metadata", {})
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
        print(f"⚠️ Failed to delete file {document['file_path']}: {e}")
    
    # Remove from projects
    for project in projects_db.values():
        if document_id in project.get("document_ids", []):
            project["document_ids"].remove(document_id)
            project["document_count"] = len(project["document_ids"])
            project["updated_at"] = datetime.utcnow().isoformat()
    
    return {"message": f"Document '{document['filename']}' deleted", "document": document}

# === Data Persistence ===

@app.on_event("startup")
async def load_data():
    """Load data from files on startup"""
    global projects_db, documents_db, chats_db
    
    try:
        # Load projects
        if Path("data/projects.json").exists():
            with open("data/projects.json", "r", encoding="utf-8") as f:
                projects_db = json.load(f)
        
        # Load documents
        if Path("data/documents.json").exists():
            with open("data/documents.json", "r", encoding="utf-8") as f:
                documents_db = json.load(f)
        
        # Load chats
        if Path("data/chats.json").exists():
            with open("data/chats.json", "r", encoding="utf-8") as f:
                chats_db = json.load(f)
        
        processed_docs = sum(1 for doc in documents_db.values() if doc.get("processing_status") == "completed")
        print(f"📊 Daten geladen: {len(projects_db)} Projekte, {len(documents_db)} Dokumente ({processed_docs} verarbeitet), {len(chats_db)} Chats")
    except Exception as e:
        print(f"⚠️ Fehler beim Laden der Daten: {e}")

@app.on_event("shutdown")
async def save_data():
    """Save data to files on shutdown"""
    try:
        with open("data/projects.json", "w", encoding="utf-8") as f:
            json.dump(projects_db, f, indent=2, default=str, ensure_ascii=False)
        
        with open("data/documents.json", "w", encoding="utf-8") as f:
            json.dump(documents_db, f, indent=2, default=str, ensure_ascii=False)
        
        with open("data/chats.json", "w", encoding="utf-8") as f:
            json.dump(chats_db, f, indent=2, default=str, ensure_ascii=False)
        
        print("💾 Daten gespeichert")
    except Exception as e:
        print(f"⚠️ Fehler beim Speichern der Daten: {e}")

# === Main Entry Point ===

if __name__ == "__main__":
    print("🚀 RagFlow Backend wird gestartet... (RAG-Enhanced Version)")
    print("=" * 70)
    print(f"📍 Haupt-URL: http://localhost:8000")
    print(f"🏥 Gesundheit: http://localhost:8000/api/health")
    print(f"🤖 Chat-Test: http://localhost:8000/api/v1/chat/test")
    print(f"📚 API-Docs: http://localhost:8000/docs")
    print(f"📋 ReDoc: http://localhost:8000/redoc")
    print("=" * 70)
    print()
    print("🎯 RAG-Features:")
    print("   ✅ PDF/DOCX/TXT Textextraktion")
    print("   ✅ Automatische Chunk-Erstellung")
    print("   ✅ Intelligente Dokumentensuche")
    print("   ✅ Kontextbasierte AI-Antworten")
    print("   ✅ Projektbezogene Dokumentenanalyse")
    print()
    print("💡 Weitere Features:")
    print("   ✅ Projektbezogener AI-Kontext")
    print("   ✅ Intelligentere Unterhaltungen")
    print("   ✅ Verbesserte Dokumenten-Integration")
    print("   ✅ Natürlichere Antworten")
    print("   ✅ Erweiterte Chat-Speicherung")
    print()
    
    # Check environment
    api_key = os.getenv("GOOGLE_API_KEY")
    env = os.getenv("ENVIRONMENT", "development")
    
    print(f"🔧 Umgebung: {env}")
    
    if not api_key or api_key == "your_google_api_key_here":
        print("⚠️  WARNUNG: GOOGLE_API_KEY nicht konfiguriert!")
        print("   📝 Zur .env Datei hinzufügen: GOOGLE_API_KEY=dein_echter_schlüssel")
        print("   🔑 Schlüssel erhalten unter: https://ai.google.dev")
    else:
        print(f"✅ Google API Key konfiguriert (Länge: {len(api_key)})")
    
    # Test document processor
    try:
        processor = DocumentProcessor()
        print("✅ Document Processor bereit")
    except Exception as e:
        print(f"⚠️ Document Processor Fehler: {e}")
        print("💡 Stelle sicher, dass alle Pakete installiert sind:")
        print("   pip install pdfplumber PyPDF2 python-docx")
    
    print()
    print("🚀 Server startet auf http://localhost:8000")
    print("   Drücke Strg+C zum Stoppen")
    print("   📄 Lade PDFs hoch und stelle Fragen zu deren Inhalt!")
    print()
    
    # Run the server
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )