# backend/app/database.py - ChromaDB Integration
import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from pydantic import BaseModel
from rich.console import Console

console = Console()

# Pydantic Models für bessere Typisierung
class ProjectModel(BaseModel):
    id: str
    name: str
    description: str
    created_at: str
    updated_at: str
    metadata: Dict[str, Any] = {}

class DocumentModel(BaseModel):
    id: str
    filename: str
    file_type: str
    file_size: int
    file_path: str
    content: str
    processing_status: str
    created_at: str
    project_ids: List[str] = []
    metadata: Dict[str, Any] = {}

class ChatModel(BaseModel):
    id: str
    project_id: Optional[str]
    messages: List[Dict[str, Any]]
    created_at: str
    updated_at: str
    metadata: Dict[str, Any] = {}

class ChromaDBManager:
    """
    ChromaDB Manager für RagFlow
    
    Verwaltet:
    - Projects (Metadaten)
    - Documents (Metadaten + Volltext-Embeddings für RAG)
    - Chats (Conversation History)
    - RAG Search (Vektor-Suche)
    """
    
    def __init__(self, persist_directory: str = "./backend/data/chromadb"):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        console.print(f"🗄️ Initializing ChromaDB at: {self.persist_directory}")
        
        # ChromaDB Client mit Persistierung
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Embedding Function für RAG
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"  # Schnell und effizient
        )
        
        # Collections erstellen/laden
        self._initialize_collections()
        
        console.print("✅ ChromaDB initialized successfully")
    
    def _initialize_collections(self):
        """Erstelle/lade alle benötigten Collections"""
        
        # Projects Collection (nur Metadaten)
        try:
            self.projects_collection = self.client.get_collection("projects")
            console.print("📁 Projects collection loaded")
        except:
            self.projects_collection = self.client.create_collection(
                name="projects",
                metadata={"description": "Project metadata storage"}
            )
            console.print("📁 Projects collection created")
        
        # Documents Collection (Metadaten + RAG Embeddings)
        try:
            self.documents_collection = self.client.get_collection(
                "documents", 
                embedding_function=self.embedding_function
            )
            console.print("📄 Documents collection loaded")
        except:
            self.documents_collection = self.client.create_collection(
                name="documents",
                embedding_function=self.embedding_function,
                metadata={"description": "Documents with RAG embeddings"}
            )
            console.print("📄 Documents collection created")
        
        # Chats Collection (nur Metadaten)
        try:
            self.chats_collection = self.client.get_collection("chats")
            console.print("💬 Chats collection loaded")
        except:
            self.chats_collection = self.client.create_collection(
                name="chats",
                metadata={"description": "Chat conversations storage"}
            )
            console.print("💬 Chats collection created")
    
    # === PROJECT OPERATIONS ===
    
    def create_project(self, name: str, description: str = "") -> ProjectModel:
        """Erstelle neues Projekt"""
        project_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        project = ProjectModel(
            id=project_id,
            name=name,
            description=description,
            created_at=timestamp,
            updated_at=timestamp
        )
        
        # In ChromaDB speichern
        self.projects_collection.add(
            ids=[project_id],
            documents=[f"Project: {name} - {description}"],  # Für mögliche Suche
            metadatas=[project.dict()]
        )
        
        console.print(f"✅ Project created: {name} ({project_id})")
        return project
    
    def get_projects(self) -> List[ProjectModel]:
        """Alle Projekte laden"""
        try:
            results = self.projects_collection.get()
            projects = []
            
            for i, project_id in enumerate(results['ids']):
                metadata = results['metadatas'][i]
                projects.append(ProjectModel(**metadata))
            
            return sorted(projects, key=lambda x: x.created_at, reverse=True)
        except Exception as e:
            console.print(f"❌ Error loading projects: {e}")
            return []
    
    def get_project(self, project_id: str) -> Optional[ProjectModel]:
        """Einzelnes Projekt laden"""
        try:
            results = self.projects_collection.get(ids=[project_id])
            if results['ids']:
                metadata = results['metadatas'][0]
                return ProjectModel(**metadata)
            return None
        except Exception as e:
            console.print(f"❌ Error loading project {project_id}: {e}")
            return None
    
    def update_project(self, project_id: str, name: str = None, description: str = None) -> Optional[ProjectModel]:
        """Projekt aktualisieren"""
        project = self.get_project(project_id)
        if not project:
            return None
        
        # Update fields
        if name is not None:
            project.name = name
        if description is not None:
            project.description = description
        
        project.updated_at = datetime.utcnow().isoformat()
        
        # Update in ChromaDB
        self.projects_collection.update(
            ids=[project_id],
            documents=[f"Project: {project.name} - {project.description}"],
            metadatas=[project.dict()]
        )
        
        console.print(f"✅ Project updated: {project.name}")
        return project
    
    def delete_project(self, project_id: str) -> bool:
        """Projekt und alle zugehörigen Daten löschen"""
        try:
            project = self.get_project(project_id)
            if not project:
                return False
            
            # 1. Alle Dokumente dieses Projekts finden und project_id entfernen
            documents = self.get_documents_by_project(project_id)
            for doc in documents:
                if project_id in doc.project_ids:
                    doc.project_ids.remove(project_id)
                    if doc.project_ids:  # Noch andere Projekte verknüpft
                        self._update_document_metadata(doc.id, doc)
                    else:  # Keine Projekte mehr - Dokument löschen
                        self.delete_document(doc.id)
            
            # 2. Alle Chats dieses Projekts löschen
            chats = self.get_chats_by_project(project_id)
            for chat in chats:
                self.delete_chat(chat.id)
            
            # 3. Projekt selbst löschen
            self.projects_collection.delete(ids=[project_id])
            
            console.print(f"✅ Project deleted: {project.name} (+{len(documents)} docs, +{len(chats)} chats)")
            return True
            
        except Exception as e:
            console.print(f"❌ Error deleting project {project_id}: {e}")
            return False
    
    def get_project_stats(self, project_id: str) -> Dict[str, int]:
        """Projekt-Statistiken"""
        documents = self.get_documents_by_project(project_id)
        chats = self.get_chats_by_project(project_id)
        
        return {
            "document_count": len(documents),
            "chat_count": len(chats),
            "total_file_size": sum(doc.file_size for doc in documents),
            "completed_documents": len([d for d in documents if d.processing_status == "completed"])
        }
    
    # === DOCUMENT OPERATIONS ===
    
    def create_document(self, filename: str, file_type: str, file_size: int, 
                       file_path: str, content: str, project_ids: List[str] = None) -> DocumentModel:
        """Dokument erstellen mit RAG-Embeddings"""
        document_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        document = DocumentModel(
            id=document_id,
            filename=filename,
            file_type=file_type,
            file_size=file_size,
            file_path=file_path,
            content=content,
            processing_status="completed",
            created_at=timestamp,
            project_ids=project_ids or []
        )
        
        # Text in Chunks aufteilen für bessere RAG-Performance
        chunks = self._split_text_into_chunks(content, chunk_size=1000, overlap=100)
        
        # Chunks mit Embeddings in ChromaDB speichern
        chunk_ids = []
        chunk_texts = []
        chunk_metadatas = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{document_id}_chunk_{i}"
            chunk_ids.append(chunk_id)
            chunk_texts.append(chunk)
            chunk_metadatas.append({
                **document.dict(),
                "chunk_index": i,
                "chunk_count": len(chunks),
                "is_chunk": True
            })
        
        # Bulk-Insert in ChromaDB
        self.documents_collection.add(
            ids=chunk_ids,
            documents=chunk_texts,
            metadatas=chunk_metadatas
        )
        
        console.print(f"✅ Document created: {filename} ({len(chunks)} chunks)")
        return document
    
    def get_documents(self, project_id: str = None) -> List[DocumentModel]:
        """Dokumente laden (optional gefiltert nach Projekt)"""
        try:
            if project_id:
                return self.get_documents_by_project(project_id)
            
            # Alle Dokumente - nur Original-Dokumente, keine Chunks
            results = self.documents_collection.get(
                where={"is_chunk": {"$ne": True}}  # Chunks ausschließen
            )
            
            documents = []
            processed_ids = set()
            
            for i, doc_id in enumerate(results['ids']):
                # Document ID extrahieren (vor _chunk_)
                base_doc_id = doc_id.split('_chunk_')[0]
                if base_doc_id not in processed_ids:
                    metadata = results['metadatas'][i]
                    if not metadata.get('is_chunk', False):
                        documents.append(DocumentModel(**metadata))
                        processed_ids.add(base_doc_id)
            
            return sorted(documents, key=lambda x: x.created_at, reverse=True)
            
        except Exception as e:
            console.print(f"❌ Error loading documents: {e}")
            return []
    
    def get_documents_by_project(self, project_id: str) -> List[DocumentModel]:
        """Dokumente eines bestimmten Projekts"""
        try:
            # Verwende ChromaDB's where clause für effiziente Filterung
            results = self.documents_collection.get(
                where={
                    "$and": [
                        {"project_ids": {"$contains": project_id}},
                        {"is_chunk": {"$ne": True}}
                    ]
                }
            )
            
            documents = []
            processed_ids = set()
            
            for i, doc_id in enumerate(results['ids']):
                base_doc_id = doc_id.split('_chunk_')[0]
                if base_doc_id not in processed_ids:
                    metadata = results['metadatas'][i]
                    documents.append(DocumentModel(**metadata))
                    processed_ids.add(base_doc_id)
            
            return documents
            
        except Exception as e:
            console.print(f"❌ Error loading documents for project {project_id}: {e}")
            return []
    
    def delete_document(self, document_id: str) -> bool:
        """Dokument und alle Chunks löschen"""
        try:
            # Alle Chunks dieses Dokuments finden
            results = self.documents_collection.get(
                where={"id": document_id}
            )
            
            if not results['ids']:
                return False
            
            # Alle Chunk-IDs sammeln
            chunk_ids = [id for id in results['ids'] if id.startswith(document_id)]
            
            # Alle Chunks löschen
            self.documents_collection.delete(ids=chunk_ids)
            
            console.print(f"✅ Document deleted: {document_id} ({len(chunk_ids)} chunks)")
            return True
            
        except Exception as e:
            console.print(f"❌ Error deleting document {document_id}: {e}")
            return False
    
    def _update_document_metadata(self, document_id: str, document: DocumentModel):
        """Dokument-Metadaten aktualisieren"""
        try:
            # Alle Chunks dieses Dokuments finden und aktualisieren
            results = self.documents_collection.get(
                where={"id": document_id}
            )
            
            for chunk_id in results['ids']:
                if chunk_id.startswith(document_id):
                    # Chunk-spezifische Metadaten beibehalten
                    chunk_results = self.documents_collection.get(ids=[chunk_id])
                    chunk_metadata = chunk_results['metadatas'][0]
                    
                    # Document-Metadaten aktualisieren, Chunk-Metadaten beibehalten
                    updated_metadata = {
                        **document.dict(),
                        "chunk_index": chunk_metadata.get("chunk_index"),
                        "chunk_count": chunk_metadata.get("chunk_count"),
                        "is_chunk": chunk_metadata.get("is_chunk")
                    }
                    
                    self.documents_collection.update(
                        ids=[chunk_id],
                        metadatas=[updated_metadata]
                    )
                    
        except Exception as e:
            console.print(f"❌ Error updating document metadata {document_id}: {e}")
    
    # === RAG SEARCH ===
    
    def search_documents(self, query: str, project_id: str = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """RAG-Suche in Dokumenten"""
        try:
            where_clause = {}
            
            if project_id:
                where_clause["project_ids"] = {"$contains": project_id}
            
            # Vektor-Suche in ChromaDB
            results = self.documents_collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_clause if where_clause else None
            )
            
            search_results = []
            
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    chunk_id = results['ids'][0][i]
                    document_id = chunk_id.split('_chunk_')[0]
                    chunk_text = results['documents'][0][i]
                    distance = results['distances'][0][i]
                    metadata = results['metadatas'][0][i]
                    
                    search_results.append({
                        "id": document_id,
                        "chunk_id": chunk_id,
                        "filename": metadata.get("filename", "Unknown"),
                        "excerpt": chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text,
                        "full_text": chunk_text,
                        "relevance_score": 1.0 - distance,  # ChromaDB gibt Distance, wir wollen Similarity
                        "metadata": metadata
                    })
            
            console.print(f"🔍 RAG Search: '{query}' → {len(search_results)} results")
            return search_results
            
        except Exception as e:
            console.print(f"❌ Error in RAG search: {e}")
            return []
    
    # === CHAT OPERATIONS ===
    
    def create_chat(self, project_id: str = None, messages: List[Dict[str, Any]] = None) -> ChatModel:
        """Chat erstellen"""
        chat_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        chat = ChatModel(
            id=chat_id,
            project_id=project_id,
            messages=messages or [],
            created_at=timestamp,
            updated_at=timestamp
        )
        
        self.chats_collection.add(
            ids=[chat_id],
            documents=[f"Chat {chat_id}"],  # Minimal document für ChromaDB
            metadatas=[chat.dict()]
        )
        
        console.print(f"✅ Chat created: {chat_id}")
        return chat
    
    def get_chats(self, project_id: str = None) -> List[ChatModel]:
        """Chats laden"""
        try:
            where_clause = {"project_id": project_id} if project_id else None
            results = self.chats_collection.get(where=where_clause)
            
            chats = []
            for i, chat_id in enumerate(results['ids']):
                metadata = results['metadatas'][i]
                chats.append(ChatModel(**metadata))
            
            return sorted(chats, key=lambda x: x.updated_at, reverse=True)
            
        except Exception as e:
            console.print(f"❌ Error loading chats: {e}")
            return []
    
    def get_chats_by_project(self, project_id: str) -> List[ChatModel]:
        """Chats eines Projekts"""
        return self.get_chats(project_id=project_id)
    
    def update_chat(self, chat_id: str, messages: List[Dict[str, Any]]) -> Optional[ChatModel]:
        """Chat aktualisieren"""
        try:
            results = self.chats_collection.get(ids=[chat_id])
            if not results['ids']:
                return None
            
            metadata = results['metadatas'][0]
            chat = ChatModel(**metadata)
            chat.messages = messages
            chat.updated_at = datetime.utcnow().isoformat()
            
            self.chats_collection.update(
                ids=[chat_id],
                metadatas=[chat.dict()]
            )
            
            return chat
            
        except Exception as e:
            console.print(f"❌ Error updating chat {chat_id}: {e}")
            return None
    
    def delete_chat(self, chat_id: str) -> bool:
        """Chat löschen"""
        try:
            self.chats_collection.delete(ids=[chat_id])
            console.print(f"✅ Chat deleted: {chat_id}")
            return True
        except Exception as e:
            console.print(f"❌ Error deleting chat {chat_id}: {e}")
            return False
    
    # === UTILITY FUNCTIONS ===
    
    def _split_text_into_chunks(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Text in überlappende Chunks aufteilen"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Versuche an Satzende zu trennen
            if end < len(text):
                # Suche nach dem letzten Punkt, Ausrufezeichen oder Fragezeichen
                last_sentence_end = max(
                    text.rfind('.', start, end),
                    text.rfind('!', start, end),
                    text.rfind('?', start, end)
                )
                
                if last_sentence_end > start + chunk_size // 2:  # Mindestens halbe Chunk-Größe
                    end = last_sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            
            # Verhindere Endlos-Schleife
            if start >= len(text):
                break
        
        return chunks
    
    def get_stats(self) -> Dict[str, Any]:
        """Gesamtstatistiken"""
        try:
            projects = self.get_projects()
            documents = self.get_documents()
            chats = self.get_chats()
            
            # ChromaDB Collection Stats
            projects_count = self.projects_collection.count()
            documents_count = self.documents_collection.count()  # Includes chunks
            chats_count = self.chats_collection.count()
            
            return {
                "projects": {
                    "total": len(projects),
                    "collection_count": projects_count
                },
                "documents": {
                    "total": len(documents),
                    "chunks_total": documents_count,
                    "completed": len([d for d in documents if d.processing_status == "completed"]),
                    "total_size": sum(d.file_size for d in documents)
                },
                "chats": {
                    "total": len(chats),
                    "collection_count": chats_count
                },
                "database": {
                    "type": "ChromaDB",
                    "location": str(self.persist_directory),
                    "collections": ["projects", "documents", "chats"]
                }
            }
            
        except Exception as e:
            console.print(f"❌ Error getting stats: {e}")
            return {}
    
    def reset_database(self):
        """Komplette Datenbank zurücksetzen (nur für Development!)"""
        console.print("⚠️ Resetting ChromaDB - ALL DATA WILL BE LOST!")
        
        # Collections löschen
        try:
            self.client.delete_collection("projects")
        except:
            pass
        
        try:
            self.client.delete_collection("documents")
        except:
            pass
        
        try:
            self.client.delete_collection("chats")
        except:
            pass
        
        # Neu initialisieren
        self._initialize_collections()
        console.print("✅ Database reset complete")

# Global instance
db_manager: Optional[ChromaDBManager] = None

def get_db_manager() -> ChromaDBManager:
    """Singleton ChromaDB Manager"""
    global db_manager
    if db_manager is None:
        db_manager = ChromaDBManager()
    return db_manager