#!/usr/bin/env python3
"""
RagFlow Backend - Optimierte Implementation
AI-powered document analysis mit verbesserter RAG-Integration
"""

import asyncio
import logging
import json
import traceback
import os
import sys
import uuid
import hashlib
import re
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from contextlib import asynccontextmanager

# FastAPI imports
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ML/NLP imports
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Google AI imports
try:
    import google.generativeai as genai
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False

# Sentence Transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

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
    """Enhanced logger for RagFlow"""
    
    def __init__(self, log_file: str = "ragflow_backend.log"):
        self.log_file = Path(log_file)
        self.setup_logging()
    
    def setup_logging(self):
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
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# In-memory databases
projects_db: Dict[str, Dict] = {}
documents_db: Dict[str, Dict] = {}
chats_db: Dict[str, Dict] = {}

# === Optimized RAG System ===
class OptimizedRAG:
    """Verbesserte RAG-Implementierung mit hybrider Suche"""
    
    def __init__(self):
        self.documents = {}
        self.chunks = {}
        self.chunk_index = {}  # chunk_id -> document_id
        self.project_index = {}  # project_id -> [document_ids]
        
        # TF-IDF für Keyword-Suche
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        self.tfidf_matrix = None
        self.chunk_texts = []
        
        # Sentence Transformer für semantische Suche
        self.sentence_model = None
        self.semantic_embeddings = None
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                logger.info("Sentence Transformer loaded successfully", "RAG")
            except Exception as e:
                logger.warning(f"Could not load Sentence Transformer: {e}", "RAG")
        else:
            logger.warning("Sentence Transformers not available - using TF-IDF only", "RAG")
    
    def add_document(self, doc_id: str, content: str, metadata: Dict = None, project_ids: List[str] = None):
        """Füge Dokument zum RAG-System hinzu"""
        try:
            metadata = metadata or {}
            project_ids = project_ids or []
            
            if not content or not content.strip():
                logger.warning(f"Empty content for document {doc_id}", "RAG")
                return
            
            # Dokument speichern
            self.documents[doc_id] = {
                "content": content,
                "metadata": metadata,
                "project_ids": project_ids,
                "chunks": []
            }
            
            # Chunks erstellen
            chunks = self._create_chunks(content)
            chunk_ids = []
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i}"
                self.chunks[chunk_id] = {
                    "content": chunk,
                    "document_id": doc_id,
                    "chunk_index": i,
                    "metadata": metadata
                }
                chunk_ids.append(chunk_id)
                self.chunk_index[chunk_id] = doc_id
            
            self.documents[doc_id]["chunks"] = chunk_ids
            
            # Projekt-Index aktualisieren
            for project_id in project_ids:
                if project_id not in self.project_index:
                    self.project_index[project_id] = []
                if doc_id not in self.project_index[project_id]:
                    self.project_index[project_id].append(doc_id)
            
            # Suchindizes neu aufbauen
            self._rebuild_search_indexes()
            
            logger.info(f"Document added to RAG: {doc_id} with {len(chunks)} chunks", "RAG")
            
        except Exception as e:
            logger.error(f"Error adding document {doc_id}: {e}", "RAG")
    
    def _create_chunks(self, content: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Erstelle überlappende Textchunks"""
        if not content:
            return []
        
        # Text bereinigen
        content = re.sub(r'\s+', ' ', content.strip())
        
        # Sätze splitten für bessere Chunk-Grenzen
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Prüfe ob Hinzufügen des Satzes die Chunk-Größe überschreitet
            if len(current_chunk) + len(sentence) + 1 > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Overlap: Behalte letzten Teil des Chunks
                words = current_chunk.split()
                overlap_words = words[-overlap//10:] if len(words) > overlap//10 else []
                current_chunk = " ".join(overlap_words) + " " + sentence
            else:
                current_chunk = current_chunk + " " + sentence if current_chunk else sentence
        
        # Letzten Chunk hinzufügen
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [content[:chunk_size]]
    
    def _rebuild_search_indexes(self):
        """Baue Suchindizes neu auf"""
        if not self.chunks:
            return
        
        try:
            # Sammle alle Chunk-Texte
            self.chunk_texts = [chunk["content"] for chunk in self.chunks.values()]
            
            if not self.chunk_texts:
                return
            
            # TF-IDF Matrix erstellen
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.chunk_texts)
            
            # Semantische Embeddings erstellen
            if self.sentence_model:
                self.semantic_embeddings = self.sentence_model.encode(self.chunk_texts)
            
            logger.info(f"Search indexes rebuilt: {len(self.chunk_texts)} chunks", "RAG")
            
        except Exception as e:
            logger.error(f"Error rebuilding search indexes: {e}", "RAG")
    
    def search(self, query: str, project_id: str = None, top_k: int = 5) -> List[Dict]:
        """Hybride Suche mit Projekt-Filterung"""
        if not query or not self.chunks:
            return []
        
        try:
            # Filtere Chunks nach Projekt
            relevant_chunks = []
            
            for chunk_id, chunk_data in self.chunks.items():
                doc_id = chunk_data["document_id"]
                if doc_id in self.documents:
                    doc_project_ids = self.documents[doc_id].get("project_ids", [])
                    if not project_id or project_id in doc_project_ids:
                        relevant_chunks.append(chunk_id)
            
            if not relevant_chunks:
                logger.warning(f"No chunks found for project {project_id}", "RAG")
                return []
            
            # TF-IDF Suche
            tfidf_results = self._tfidf_search(query, relevant_chunks, top_k)
            
            # Semantische Suche
            semantic_results = []
            if self.sentence_model and self.semantic_embeddings is not None:
                semantic_results = self._semantic_search(query, relevant_chunks, top_k)
            
            # Kombiniere Ergebnisse
            combined_results = self._combine_results(tfidf_results, semantic_results, top_k)
            
            logger.info(f"Search '{query}' in project {project_id}: {len(combined_results)} results", "RAG")
            return combined_results
            
        except Exception as e:
            logger.error(f"Search error: {e}", "RAG")
            return []
    
    def _tfidf_search(self, query: str, relevant_chunks: List[str], top_k: int) -> List[Dict]:
        """TF-IDF basierte Keyword-Suche"""
        try:
            if not self.tfidf_matrix or not self.chunk_texts:
                return []
            
            # Query-Vektor erstellen
            query_vector = self.tfidf_vectorizer.transform([query])
            
            # Relevante Chunk-Indizes finden
            chunk_indices = []
            chunk_ids_list = list(self.chunks.keys())
            
            for chunk_id in relevant_chunks:
                try:
                    idx = chunk_ids_list.index(chunk_id)
                    chunk_indices.append(idx)
                except ValueError:
                    continue
            
            if not chunk_indices:
                return []
            
            # Cosine Similarity berechnen
            similarities = cosine_similarity(query_vector, self.tfidf_matrix[chunk_indices]).flatten()
            
            # Top-K Ergebnisse
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for i in top_indices:
                if similarities[i] > 0.05:  # Minimum threshold
                    chunk_idx = chunk_indices[i]
                    chunk_id = chunk_ids_list[chunk_idx]
                    chunk_data = self.chunks[chunk_id]
                    
                    results.append({
                        "document_id": chunk_data["document_id"],
                        "content": chunk_data["content"],
                        "score": float(similarities[i]),
                        "method": "tfidf",
                        "chunk_id": chunk_id
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"TF-IDF search error: {e}", "RAG")
            return []
    
    def _semantic_search(self, query: str, relevant_chunks: List[str], top_k: int) -> List[Dict]:
        """Semantische Suche mit Sentence Transformers"""
        try:
            if not self.sentence_model or self.semantic_embeddings is None:
                return []
            
            # Query Embedding
            query_embedding = self.sentence_model.encode([query])
            
            # Relevante Chunk-Indizes
            chunk_indices = []
            chunk_ids_list = list(self.chunks.keys())
            
            for chunk_id in relevant_chunks:
                try:
                    idx = chunk_ids_list.index(chunk_id)
                    chunk_indices.append(idx)
                except ValueError:
                    continue
            
            if not chunk_indices:
                return []
            
            # Semantische Ähnlichkeit berechnen
            relevant_embeddings = self.semantic_embeddings[chunk_indices]
            similarities = cosine_similarity(query_embedding, relevant_embeddings).flatten()
            
            # Top-K Ergebnisse
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for i in top_indices:
                if similarities[i] > 0.2:  # Höherer threshold für semantische Suche
                    chunk_idx = chunk_indices[i]
                    chunk_id = chunk_ids_list[chunk_idx]
                    chunk_data = self.chunks[chunk_id]
                    
                    results.append({
                        "document_id": chunk_data["document_id"],
                        "content": chunk_data["content"],
                        "score": float(similarities[i]),
                        "method": "semantic",
                        "chunk_id": chunk_id
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Semantic search error: {e}", "RAG")
            return []
    
    def _combine_results(self, tfidf_results: List[Dict], semantic_results: List[Dict], top_k: int) -> List[Dict]:
        """Kombiniere TF-IDF und semantische Ergebnisse"""
        combined = {}
        
        # TF-IDF Ergebnisse (Gewichtung 0.6)
        for result in tfidf_results:
            chunk_id = result["chunk_id"]
            combined[chunk_id] = result.copy()
            combined[chunk_id]["score"] = result["score"] * 0.6
            combined[chunk_id]["methods"] = ["tfidf"]
        
        # Semantische Ergebnisse (Gewichtung 0.4)
        for result in semantic_results:
            chunk_id = result["chunk_id"]
            if chunk_id in combined:
                # Kombiniere Scores
                combined[chunk_id]["score"] += result["score"] * 0.4
                combined[chunk_id]["methods"].append("semantic")
            else:
                combined[chunk_id] = result.copy()
                combined[chunk_id]["score"] = result["score"] * 0.4
                combined[chunk_id]["methods"] = ["semantic"]
        
        # Sortiere und gib Top-K zurück
        results = list(combined.values())
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results[:top_k]

# Initialize optimized RAG system
optimized_rag = OptimizedRAG()

# === Document Processor ===
class DocumentProcessor:
    """Enhanced document processing"""
    
    @staticmethod
    def extract_text_from_pdf(file_path: Path) -> str:
        """Extract text from PDF"""
        try:
            text = ""
            
            # Try pdfplumber first
            if 'pdfplumber' in sys.modules:
                import pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            
            # Fallback to PyPDF2
            elif 'PyPDF2' in sys.modules:
                import PyPDF2
                with open(file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"PDF extraction error: {e}", "PROCESSING")
            return f"[Error extracting PDF: {e}]"
    
    @staticmethod
    def extract_text_from_docx(file_path: Path) -> str:
        """Extract text from DOCX"""
        try:
            if 'docx' in sys.modules:
                from docx import Document
                doc = Document(file_path)
                text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
                return text.strip()
            else:
                return "[DOCX processing not available]"
                
        except Exception as e:
            logger.error(f"DOCX extraction error: {e}", "PROCESSING")
            return f"[Error extracting DOCX: {e}]"
    
    @staticmethod
    def extract_text_from_txt(file_path: Path) -> str:
        """Extract text from TXT"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except Exception as e:
            logger.error(f"TXT extraction error: {e}", "PROCESSING")
            return f"[Error extracting TXT: {e}]"

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
    
    def generate_response(self, prompt: str) -> str:
        """Generate AI response"""
        if not self.model:
            return f"AI response: {prompt[:100]}... (AI not available)"
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            logger.error(f"AI generation failed: {e}", "AI")
            return f"Entschuldigung, bei der Generierung der Antwort ist ein Fehler aufgetreten: {str(e)}"

# Initialize AI chat
ai_chat = GoogleAIChat()

# === Data Management ===
def save_data():
    """Save all data to files"""
    try:
        with open(DATA_DIR / "projects.json", "w", encoding="utf-8") as f:
            json.dump(projects_db, f, indent=2, ensure_ascii=False)
        
        with open(DATA_DIR / "documents.json", "w", encoding="utf-8") as f:
            json.dump(documents_db, f, indent=2, ensure_ascii=False)
        
        with open(DATA_DIR / "chats.json", "w", encoding="utf-8") as f:
            json.dump(chats_db, f, indent=2, ensure_ascii=False)
        
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
        load_documents_into_rag()
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}", "DATA")

def load_documents_into_rag():
    """Load existing documents into RAG system"""
    loaded_count = 0
    
    for doc_id, doc_data in documents_db.items():
        if (doc_data.get("processing_status") == "completed" and 
            "extracted_text" in doc_data):
            
            project_ids = doc_data.get("project_ids", [])
            optimized_rag.add_document(
                doc_id=doc_id,
                content=doc_data["extracted_text"],
                metadata=doc_data.get("metadata", {}),
                project_ids=project_ids
            )
            loaded_count += 1
    
    logger.info(f"Loaded {loaded_count} documents into RAG system", "RAG")

# === FastAPI App ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting RagFlow Backend...", "STARTUP")
    load_data()
    logger.info("RagFlow Backend ready!", "STARTUP")
    yield
    # Shutdown
    logger.info("Shutting down RagFlow Backend...", "SHUTDOWN")

app = FastAPI(
    title="RagFlow Backend",
    description="AI-powered document analysis with optimized RAG",
    version="2.1.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Health Endpoints ===
@app.get("/api/health")
async def health_check():
    """Basic health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.1.0"
    }

@app.get("/api/health/detailed")
async def detailed_health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.1.0",
        "components": {
            "database": "operational",
            "google_ai": "available" if ai_chat.model else "unavailable",
            "rag_system": "operational",
            "document_processing": "available" if DOCUMENT_PROCESSING_AVAILABLE else "limited"
        },
        "statistics": {
            "projects": len(projects_db),
            "documents": len(documents_db),
            "chats": len(chats_db),
            "rag_documents": len(optimized_rag.documents),
            "rag_chunks": len(optimized_rag.chunks)
        }
    }

# === Project Endpoints ===
@app.get("/api/v1/projects/")
async def get_projects(skip: int = 0, limit: int = 10, search: str = None):
    """Get projects"""
    projects = list(projects_db.values())
    
    if search:
        search_lower = search.lower()
        projects = [p for p in projects if search_lower in p.get("name", "").lower()]
    
    # Sort by updated date
    projects.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    
    total = len(projects)
    projects = projects[skip:skip + limit]
    
    return {
        "projects": projects,
        "total": total,
        "skip": skip,
        "limit": limit
    }

@app.post("/api/v1/projects/")
async def create_project(data: dict):
    """Create new project"""
    project_id = str(uuid.uuid4())[:8]
    
    project = {
        "id": project_id,
        "name": data.get("name", f"Project {project_id}"),
        "description": data.get("description", ""),
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "document_ids": [],
        "document_count": 0,
        "chat_count": 0,
        "settings": {}
    }
    
    projects_db[project_id] = project
    save_data()
    
    logger.info(f"Created project: {project['name']}", "PROJECT")
    return project

@app.get("/api/v1/projects/{project_id}")
async def get_project(project_id: str):
    """Get project details"""
    if project_id not in projects_db:
        raise HTTPException(status_code=404, detail="Project not found")
    
    project = projects_db[project_id].copy()
    
    # Add documents
    documents = [
        doc for doc in documents_db.values()
        if project_id in doc.get("project_ids", [])
    ]
    project["documents"] = documents
    
    return project

# === Document Endpoints ===
@app.post("/api/v1/documents/upload")
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    project_id: Optional[str] = Form(None)
):
    """Upload documents"""
    uploaded_docs = []
    
    for file in files:
        try:
            # Generate file ID
            file_id = str(uuid.uuid4())
            file_extension = Path(file.filename).suffix.lower()
            safe_filename = f"{file_id}{file_extension}"
            file_path = UPLOAD_DIR / safe_filename
            
            # Save file
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)
            
            # Create document record
            document_data = {
                "id": file_id,
                "filename": file.filename,
                "file_type": file_extension,
                "file_size": len(content),
                "file_path": str(file_path),
                "uploaded_at": datetime.utcnow().isoformat(),
                "processing_status": "pending",
                "project_ids": [project_id] if project_id else [],
                "tags": [],
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
            background_tasks.add_task(process_document_optimized, file_id)
            
            logger.info(f"Uploaded document: {file.filename}", "UPLOAD")
            
        except Exception as e:
            logger.error(f"Failed to upload {file.filename}: {e}", "UPLOAD")
            continue
    
    save_data()
    return {"uploaded": len(uploaded_docs), "documents": uploaded_docs}

async def process_document_optimized(document_id: str):
    """Optimized document processing"""
    try:
        doc = documents_db.get(document_id)
        if not doc:
            return
        
        doc["processing_status"] = "processing"
        doc["processing_started_at"] = datetime.utcnow().isoformat()
        
        file_path = Path(doc["file_path"])
        file_type = doc["file_type"].lower()
        
        # Extract text
        if file_type == ".pdf":
            extracted_text = DocumentProcessor.extract_text_from_pdf(file_path)
        elif file_type in [".docx", ".doc"]:
            extracted_text = DocumentProcessor.extract_text_from_docx(file_path)
        elif file_type == ".txt":
            extracted_text = DocumentProcessor.extract_text_from_txt(file_path)
        else:
            extracted_text = f"[Unsupported file type: {file_type}]"
        
        # Update document
        doc["extracted_text"] = extracted_text
        doc["text_length"] = len(extracted_text)
        doc["processing_status"] = "completed"
        doc["processed_at"] = datetime.utcnow().isoformat()
        
        # Add to RAG system
        project_ids = doc.get("project_ids", [])
        optimized_rag.add_document(
            doc_id=document_id,
            content=extracted_text,
            metadata=doc.get("metadata", {}),
            project_ids=project_ids
        )
        
        save_data()
        logger.info(f"Document processed and added to RAG: {doc['filename']}", "PROCESSING")
        
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
    
    # Sort by upload date
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
    if document_id in optimized_rag.documents:
        del optimized_rag.documents[document_id]
        # Remove chunks
        chunks_to_remove = [cid for cid, chunk in optimized_rag.chunks.items() 
                           if chunk["document_id"] == document_id]
        for chunk_id in chunks_to_remove:
            del optimized_rag.chunks[chunk_id]
            if chunk_id in optimized_rag.chunk_index:
                del optimized_rag.chunk_index[chunk_id]
        
        # Rebuild search indexes
        optimized_rag._rebuild_search_indexes()
    
    # Remove document
    del documents_db[document_id]
    save_data()
    
    logger.info(f"Deleted document: {document['filename']}", "DELETE")
    return {"message": f"Document '{document['filename']}' deleted"}

# === Chat Endpoints ===
@app.post("/api/v1/chat")
async def optimized_chat_endpoint(request: Request):
    """Optimized chat endpoint with enhanced RAG"""
    try:
        body = await request.json()
        logger.info(f"Chat request received for project: {body.get('project_id')}", "CHAT")
        
        # Parse message
        if "messages" in body:
            messages = body.get("messages", [])
            user_messages = [msg for msg in messages if msg.get("role") == "user"]
            message = user_messages[-1].get("content", "") if user_messages else ""
        else:
            message = body.get("message", "")
        
        project_id = body.get("project_id")
        use_documents = body.get("use_documents", True)
        
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")
        
        logger.info(f"Processing: '{message[:100]}...' for project: {project_id}", "CHAT")
        
        # RAG search with project filtering
        context = ""
        sources = []
        context_length = 0
        
        if use_documents and optimized_rag.documents:
            # Use optimized RAG search
            search_results = optimized_rag.search(message, project_id=project_id, top_k=5)
            
            if search_results:
                context_parts = []
                for i, result in enumerate(search_results):
                    doc_id = result['document_id']
                    content = result['content']
                    score = result['score']
                    methods = result.get('methods', ['unknown'])
                    
                    # Get document metadata
                    doc_data = documents_db.get(doc_id, {})
                    filename = doc_data.get('filename', 'Unknown')
                    
                    context_part = f"""=== Dokument {i+1}: {filename} (Relevanz: {score:.3f}, Methoden: {', '.join(methods)}) ===
{content}
"""
                    context_parts.append(context_part)
                    
                    # Source information
                    sources.append({
                        "id": doc_id,
                        "filename": filename,
                        "excerpt": content[:300] + "..." if len(content) > 300 else content,
                        "relevance_score": score,
                        "search_methods": methods,
                        "type": "document"
                    })
                
                context = "\n".join(context_parts)
                context_length = len(context)
                
                logger.info(f"Found {len(search_results)} relevant documents, context length: {context_length}", "RAG")
            else:
                logger.warning(f"No relevant documents found for project {project_id}", "RAG")
        
        # Enhanced prompt for better AI responses
        if context:
            enhanced_prompt = f"""Du bist ein hilfreicher AI-Assistent, der Dokumente analysiert und Fragen dazu beantwortet.

WICHTIGE KONTEXTINFORMATIONEN AUS DOKUMENTEN:
{context}

BENUTZERANFRAGE: {message}

ANWEISUNGEN:
1. Analysiere die bereitgestellten Dokumentinhalte sorgfältig
2. Beantworte die Benutzeranfrage präzise basierend auf den Dokumentinhalten
3. Wenn die Antwort in den Dokumenten gefunden wird, zitiere relevante Passagen
4. Wenn die Information nicht vollständig in den Dokumenten enthalten ist, sage das deutlich
5. Gib eine strukturierte, hilfreiche und vollständige Antwort
6. Verwende eine freundliche, professionelle Sprache

Antwort:"""
            
            response_text = ai_chat.generate_response(enhanced_prompt)
        else:
            # Fallback when no documents found
            fallback_prompt = f"""Benutzeranfrage: {message}

Hinweis: Es wurden keine relevanten Dokumente für diese Anfrage gefunden oder es sind keine Dokumente im aktuellen Projekt verfügbar.

Bitte gib eine hilfreiche Antwort basierend auf deinem allgemeinen Wissen und weise freundlich darauf hin, dass zusätzliche Dokumentinformationen die Antwort verbessern könnten."""
            
            response_text = ai_chat.generate_response(fallback_prompt)
        
        # Create chat entry
        chat_id = str(uuid.uuid4())
        chat_entry = {
            "id": chat_id,
            "project_id": project_id,
            "user_message": message,
            "ai_response": response_text,
            "timestamp": datetime.utcnow().isoformat(),
            "sources": sources,
            "context_used": bool(context),
            "context_length": context_length,
            "documents_searched": len(optimized_rag.documents),
            "model": "gemini-1.5-flash",
            "metadata": {
                "context_enhanced": bool(context),
                "sources_count": len(sources),
                "project_id": project_id,
                "rag_version": "optimized"
            }
        }
        
        chats_db[chat_id] = chat_entry
        save_data()
        
        # Response
        response_data = {
            "response": response_text,
            "chat_id": chat_id,
            "project_id": project_id,
            "timestamp": chat_entry["timestamp"],
            "sources": sources,
            "success": True,
            "model_info": {
                "model": "gemini-1.5-flash",
                "context_used": bool(context),
                "context_length": context_length,
                "features_used": {
                    "intelligent_document_search": True,
                    "hybrid_search": True,
                    "project_filtering": True,
                    "context_enhancement": bool(context)
                }
            },
            "intelligence_metadata": {
                "sources_found": len(sources),
                "context_enhanced": bool(context),
                "processing_method": "optimized_rag",
                "search_methods_used": list(set([method for source in sources for method in source.get("search_methods", [])]))
            }
        }
        
        logger.info(f"Chat response generated: {len(response_text)} chars, {len(sources)} sources", "CHAT")
        return response_data
        
    except Exception as e:
        logger.error(f"Chat error: {e}\n{traceback.format_exc()}", "CHAT")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "response": "Es tut mir leid, bei der Verarbeitung Ihrer Anfrage ist ein Fehler aufgetreten."
            }
        )

@app.post("/api/v1/chat/test")
async def test_chat():
    """Test chat functionality"""
    try:
        test_response = ai_chat.generate_response("Hello, this is a test message.")
        return {
            "status": "success",
            "response": test_response,
            "ai_available": ai_chat.model is not None
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "ai_available": False
        }

@app.get("/api/v1/chat/models")
async def get_available_models():
    """Get available AI models"""
    return {
        "models": [
            {
                "id": "gemini-1.5-flash",
                "name": "Gemini 1.5 Flash",
                "provider": "Google",
                "available": ai_chat.model is not None
            }
        ],
        "default": "gemini-1.5-flash"
    }

# === Chat History Endpoints ===
@app.get("/api/v1/chats/")
async def get_chats(project_id: Optional[str] = None, skip: int = 0, limit: int = 20):
    """Get chat history"""
    chats = list(chats_db.values())
    
    if project_id:
        chats = [chat for chat in chats if chat.get("project_id") == project_id]
    
    # Sort by timestamp (newest first)
    chats.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    
    total = len(chats)
    chats = chats[skip:skip + limit]
    
    return {
        "chats": chats,
        "total": total,
        "skip": skip,
        "limit": limit
    }

@app.get("/api/v1/chats/{chat_id}")
async def get_chat(chat_id: str):
    """Get specific chat"""
    if chat_id not in chats_db:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    return chats_db[chat_id]

@app.delete("/api/v1/chats/{chat_id}")
async def delete_chat(chat_id: str):
    """Delete a chat"""
    if chat_id not in chats_db:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    del chats_db[chat_id]
    save_data()
    
    logger.info(f"Deleted chat: {chat_id}", "DELETE")
    return {"message": "Chat deleted"}

# === Search and Analytics Endpoints ===
@app.post("/api/v1/search")
async def search_documents(request: dict):
    """Search documents using RAG"""
    query = request.get("query", "")
    project_id = request.get("project_id")
    top_k = request.get("top_k", 10)
    
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    
    try:
        results = optimized_rag.search(query, project_id=project_id, top_k=top_k)
        
        # Add document metadata to results
        enriched_results = []
        for result in results:
            doc_id = result["document_id"]
            doc_data = documents_db.get(doc_id, {})
            
            enriched_result = {
                **result,
                "document_filename": doc_data.get("filename", "Unknown"),
                "document_type": doc_data.get("file_type", ""),
                "upload_date": doc_data.get("uploaded_at", ""),
                "file_size": doc_data.get("file_size", 0)
            }
            enriched_results.append(enriched_result)
        
        return {
            "query": query,
            "results": enriched_results,
            "total_results": len(enriched_results),
            "project_id": project_id
        }
        
    except Exception as e:
        logger.error(f"Search error: {e}", "SEARCH")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/api/v1/analytics/overview")
async def get_analytics_overview():
    """Get analytics overview"""
    try:
        # Project analytics
        project_stats = {
            "total_projects": len(projects_db),
            "projects_with_documents": len([p for p in projects_db.values() if p.get("document_count", 0) > 0]),
            "average_documents_per_project": sum(p.get("document_count", 0) for p in projects_db.values()) / max(len(projects_db), 1)
        }
        
        # Document analytics
        completed_docs = [d for d in documents_db.values() if d.get("processing_status") == "completed"]
        document_stats = {
            "total_documents": len(documents_db),
            "completed_documents": len(completed_docs),
            "processing_documents": len([d for d in documents_db.values() if d.get("processing_status") == "processing"]),
            "failed_documents": len([d for d in documents_db.values() if d.get("processing_status") == "failed"]),
            "total_text_length": sum(len(d.get("extracted_text", "")) for d in completed_docs),
            "file_types": {}
        }
        
        # File type distribution
        for doc in documents_db.values():
            file_type = doc.get("file_type", "unknown")
            document_stats["file_types"][file_type] = document_stats["file_types"].get(file_type, 0) + 1
        
        # Chat analytics
        chat_stats = {
            "total_chats": len(chats_db),
            "chats_with_context": len([c for c in chats_db.values() if c.get("context_used", False)]),
            "average_sources_per_chat": sum(len(c.get("sources", [])) for c in chats_db.values()) / max(len(chats_db), 1)
        }
        
        # RAG analytics
        rag_stats = {
            "documents_in_rag": len(optimized_rag.documents),
            "total_chunks": len(optimized_rag.chunks),
            "projects_indexed": len(optimized_rag.project_index),
            "semantic_search_available": optimized_rag.sentence_model is not None
        }
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "project_stats": project_stats,
            "document_stats": document_stats,
            "chat_stats": chat_stats,
            "rag_stats": rag_stats
        }
        
    except Exception as e:
        logger.error(f"Analytics error: {e}", "ANALYTICS")
        raise HTTPException(status_code=500, detail=f"Analytics failed: {str(e)}")

# === System Status ===
@app.get("/api/v1/system/status")
async def get_system_status():
    """Get detailed system status"""
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.1.0",
        "status": "operational",
        "components": {
            "rag_system": {
                "status": "operational",
                "documents": len(optimized_rag.documents),
                "chunks": len(optimized_rag.chunks),
                "semantic_search": optimized_rag.sentence_model is not None
            },
            "ai_chat": {
                "status": "operational" if ai_chat.model else "unavailable",
                "model": "gemini-1.5-flash" if ai_chat.model else None
            },
            "document_processing": {
                "status": "available" if DOCUMENT_PROCESSING_AVAILABLE else "limited",
                "pdf_support": 'pdfplumber' in sys.modules or 'PyPDF2' in sys.modules,
                "docx_support": 'docx' in sys.modules
            }
        },
        "data": {
            "projects": len(projects_db),
            "documents": len(documents_db),
            "chats": len(chats_db)
        },
        "performance": {
            "memory_usage": f"{sys.getsizeof(documents_db) + sys.getsizeof(chats_db) + sys.getsizeof(projects_db)} bytes",
            "uptime": "Running"
        }
    }

# === Error Handlers ===
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=404,
        content={"error": "Not found", "detail": str(exc.detail)}
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception):
    logger.error(f"Internal server error: {exc}", "ERROR")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": "An unexpected error occurred"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)