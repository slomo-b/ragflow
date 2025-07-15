#!/usr/bin/env python3
"""
Vector Embeddings System für semantische Dokumentensuche
Nutzt sentence-transformers für hochwertige Embeddings
"""

import os
import numpy as np
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import asyncio
from datetime import datetime
import pickle
import uuid

# Vector/ML libraries
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("⚠️ Embeddings nicht verfügbar. Installiere: pip install sentence-transformers faiss-cpu")

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ sklearn nicht verfügbar. Installiere: pip install scikit-learn")

class VectorEmbeddingManager:
    """Verwaltet Vector Embeddings für semantische Suche"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialisiert den Embedding Manager
        
        Args:
            model_name: SentenceTransformer Modell (klein aber gut)
        """
        self.model_name = model_name
        self.model = None
        self.vector_store = None
        self.chunk_metadata = []
        self.embeddings_cache = {}
        self.vector_store_path = Path("data/vector_store")
        self.vector_store_path.mkdir(exist_ok=True)
        
        # Lade Modell
        self._load_model()
        
        # Lade existierende Embeddings
        self._load_existing_embeddings()
    
    def _load_model(self):
        """Lade das Embedding-Modell"""
        if not EMBEDDINGS_AVAILABLE:
            print("❌ sentence-transformers nicht verfügbar")
            return
        
        try:
            print(f"📥 Lade Embedding-Modell: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"✅ Modell geladen (Dimension: {self.model.get_sentence_embedding_dimension()})")
        except Exception as e:
            print(f"❌ Fehler beim Laden des Modells: {e}")
            self.model = None
    
    def _load_existing_embeddings(self):
        """Lade existierende Embeddings von der Festplatte"""
        try:
            # Lade Metadata
            metadata_file = self.vector_store_path / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    self.chunk_metadata = json.load(f)
                print(f"📥 {len(self.chunk_metadata)} Chunk-Metadaten geladen")
            
            # Lade Vector Store
            faiss_file = self.vector_store_path / "vector_store.faiss"
            if faiss_file.exists() and EMBEDDINGS_AVAILABLE:
                import faiss
                self.vector_store = faiss.read_index(str(faiss_file))
                print(f"📥 FAISS Vector Store geladen ({self.vector_store.ntotal} Vektoren)")
            
        except Exception as e:
            print(f"⚠️ Fehler beim Laden der Embeddings: {e}")
            self.chunk_metadata = []
            self.vector_store = None
    
    def _save_embeddings(self):
        """Speichere Embeddings auf die Festplatte"""
        try:
            # Speichere Metadata
            metadata_file = self.vector_store_path / "metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.chunk_metadata, f, indent=2, ensure_ascii=False, default=str)
            
            # Speichere Vector Store
            if self.vector_store and EMBEDDINGS_AVAILABLE:
                import faiss
                faiss_file = self.vector_store_path / "vector_store.faiss"
                faiss.write_index(self.vector_store, str(faiss_file))
            
            print(f"💾 Embeddings gespeichert ({len(self.chunk_metadata)} Chunks)")
            
        except Exception as e:
            print(f"❌ Fehler beim Speichern der Embeddings: {e}")
    
    async def process_document_chunks(self, document_id: str, chunks: List[Dict[str, Any]], document_metadata: Dict[str, Any]):
        """
        Verarbeitet Chunks eines Dokuments und erstellt Embeddings
        
        Args:
            document_id: Dokument-ID
            chunks: Liste der Text-Chunks
            document_metadata: Metadaten des Dokuments
        """
        if not self.model:
            print("❌ Kein Embedding-Modell verfügbar")
            return
        
        print(f"🔄 Erstelle Embeddings für {len(chunks)} Chunks...")
        
        # Entferne alte Embeddings für dieses Dokument
        self._remove_document_embeddings(document_id)
        
        # Extrahiere Texte für Embedding
        chunk_texts = [chunk.get("text", "") for chunk in chunks]
        
        try:
            # Erstelle Embeddings
            embeddings = self.model.encode(chunk_texts, show_progress_bar=True)
            print(f"✅ {len(embeddings)} Embeddings erstellt")
            
            # Erstelle Chunk-Metadaten
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_metadata = {
                    "chunk_id": chunk.get("id", str(uuid.uuid4())),
                    "document_id": document_id,
                    "chunk_index": i,
                    "text": chunk.get("text", ""),
                    "text_preview": chunk.get("text", "")[:200],
                    "document_filename": document_metadata.get("filename", "Unknown"),
                    "document_type": document_metadata.get("file_type", "unknown"),
                    "created_at": datetime.utcnow().isoformat(),
                    "start_pos": chunk.get("start_pos", 0),
                    "end_pos": chunk.get("end_pos", 0)
                }
                
                self.chunk_metadata.append(chunk_metadata)
            
            # Füge zu Vector Store hinzu
            self._add_to_vector_store(embeddings)
            
            # Speichere
            self._save_embeddings()
            
            print(f"✅ Embeddings für Dokument {document_metadata.get('filename')} erstellt")
            
        except Exception as e:
            print(f"❌ Fehler beim Erstellen der Embeddings: {e}")
    
    def _remove_document_embeddings(self, document_id: str):
        """Entferne alle Embeddings für ein Dokument"""
        # Finde Indizes der zu entfernenden Chunks
        indices_to_remove = [
            i for i, meta in enumerate(self.chunk_metadata)
            if meta.get("document_id") == document_id
        ]
        
        if indices_to_remove:
            print(f"🗑️ Entferne {len(indices_to_remove)} alte Embeddings")
            
            # Entferne aus Metadaten (rückwärts um Indizes nicht zu verschieben)
            for i in reversed(indices_to_remove):
                del self.chunk_metadata[i]
            
            # Vector Store neu aufbauen (FAISS unterstützt kein selektives Löschen)
            if self.chunk_metadata:
                self._rebuild_vector_store()
            else:
                self.vector_store = None
    
    def _add_to_vector_store(self, embeddings: np.ndarray):
        """Füge Embeddings zum Vector Store hinzu"""
        if not EMBEDDINGS_AVAILABLE:
            return
        
        try:
            import faiss
            
            if self.vector_store is None:
                # Erstelle neuen Index
                dimension = embeddings.shape[1]
                self.vector_store = faiss.IndexFlatIP(dimension)  # Inner Product (ähnlich Cosine)
                print(f"🆕 Neuer FAISS Index erstellt (Dimension: {dimension})")
            
            # Normalisiere Embeddings für Cosine Similarity
            embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            # Füge hinzu
            self.vector_store.add(embeddings_normalized.astype('float32'))
            print(f"➕ {len(embeddings)} Embeddings zum Vector Store hinzugefügt")
            
        except Exception as e:
            print(f"❌ Fehler beim Vector Store Update: {e}")
    
    def _rebuild_vector_store(self):
        """Baue Vector Store komplett neu auf"""
        if not self.model or not self.chunk_metadata:
            return
        
        print(f"🔄 Baue Vector Store neu auf...")
        
        try:
            # Extrahiere alle Texte
            all_texts = [meta["text"] for meta in self.chunk_metadata]
            
            # Erstelle Embeddings
            embeddings = self.model.encode(all_texts, show_progress_bar=True)
            
            # Erstelle neuen Vector Store
            self.vector_store = None
            self._add_to_vector_store(embeddings)
            
            print(f"✅ Vector Store neu aufgebaut")
            
        except Exception as e:
            print(f"❌ Fehler beim Neuaufbau: {e}")
    
    async def semantic_search(self, query: str, top_k: int = 5, min_similarity: float = 0.3) -> List[Dict[str, Any]]:
        """
        Führe semantische Suche durch
        
        Args:
            query: Suchanfrage
            top_k: Anzahl der Ergebnisse
            min_similarity: Minimale Ähnlichkeit
            
        Returns:
            Liste der gefundenen Chunks mit Similarity Score
        """
        if not self.model or not self.vector_store or not self.chunk_metadata:
            print("❌ Vector Search nicht verfügbar")
            return []
        
        try:
            # Erstelle Query Embedding
            query_embedding = self.model.encode([query])
            query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
            
            # Suche ähnliche Vektoren
            similarities, indices = self.vector_store.search(query_embedding.astype('float32'), top_k)
            
            results = []
            for similarity, idx in zip(similarities[0], indices[0]):
                if idx < len(self.chunk_metadata) and similarity >= min_similarity:
                    chunk_meta = self.chunk_metadata[idx].copy()
                    chunk_meta["similarity_score"] = float(similarity)
                    chunk_meta["search_method"] = "semantic"
                    results.append(chunk_meta)
            
            print(f"🔍 Semantische Suche: {len(results)} Ergebnisse für '{query}'")
            return results
            
        except Exception as e:
            print(f"❌ Fehler bei semantischer Suche: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Gebe Statistiken über den Vector Store zurück"""
        return {
            "model_name": self.model_name,
            "model_available": self.model is not None,
            "total_chunks": len(self.chunk_metadata),
            "vector_store_size": self.vector_store.ntotal if self.vector_store else 0,
            "unique_documents": len(set(meta.get("document_id") for meta in self.chunk_metadata)),
            "embedding_dimension": self.model.get_sentence_embedding_dimension() if self.model else 0
        }

class HybridSearchManager:
    """Kombiniert keyword-basierte und semantische Suche"""
    
    def __init__(self, documents_db: dict, vector_manager: VectorEmbeddingManager):
        self.documents_db = documents_db
        self.vector_manager = vector_manager
    
    async def hybrid_search(self, query: str, project_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Führe hybride Suche durch (Keyword + Semantic)
        
        Args:
            query: Suchanfrage
            project_id: Projekt-ID
            top_k: Anzahl der Ergebnisse
            
        Returns:
            Kombinierte und gewichtete Ergebnisse
        """
        print(f"🔍 Hybride Suche für: '{query}' in Projekt: {project_id}")
        
        # 1. Keyword-basierte Suche (wie vorher)
        keyword_results = self._keyword_search(query, project_id, top_k)
        
        # 2. Semantische Suche
        semantic_results = await self.vector_manager.semantic_search(query, top_k * 2)
        
        # Filtere semantische Ergebnisse nach Projekt
        project_semantic_results = [
            result for result in semantic_results
            if result.get("document_id") in [
                doc_id for doc_id, doc in self.documents_db.items()
                if project_id in doc.get("project_ids", [])
            ]
        ]
        
        # 3. Kombiniere Ergebnisse
        combined_results = self._combine_search_results(keyword_results, project_semantic_results, top_k)
        
        print(f"✅ Hybrid-Suche: {len(keyword_results)} Keyword + {len(project_semantic_results)} Semantic = {len(combined_results)} Ergebnisse")
        
        return combined_results
    
    def _keyword_search(self, query: str, project_id: str, max_results: int) -> List[Dict[str, Any]]:
        """Einfache Keyword-Suche (wie im Original RAGChatEnhancer)"""
        results = []
        query_lower = query.lower()
        
        # Finde Dokumente des Projekts
        project_docs = [
            doc for doc in self.documents_db.values()
            if project_id in doc.get("project_ids", []) and 
               doc.get("processing_status") == "completed" and
               doc.get("extracted_text")
        ]
        
        for doc in project_docs:
            full_text = doc.get("extracted_text", "")
            if query_lower in full_text.lower():
                # Finde relevante Abschnitte
                text_lower = full_text.lower()
                positions = []
                start = 0
                while True:
                    pos = text_lower.find(query_lower, start)
                    if pos == -1:
                        break
                    positions.append(pos)
                    start = pos + 1
                
                # Extrahiere Kontext um die gefundenen Positionen
                for pos in positions[:max_results]:
                    context_start = max(0, pos - 200)
                    context_end = min(len(full_text), pos + 200)
                    context = full_text[context_start:context_end]
                    
                    results.append({
                        "chunk_id": str(uuid.uuid4()),
                        "document_id": doc.get("id"),
                        "text": context,
                        "text_preview": context[:200],
                        "document_filename": doc.get("filename", "Unknown"),
                        "similarity_score": 0.8,  # Fixer Wert für Keyword-Treffer
                        "search_method": "keyword",
                        "position": pos
                    })
        
        return results[:max_results]
    
    def _combine_search_results(self, keyword_results: List[Dict], semantic_results: List[Dict], top_k: int) -> List[Dict[str, Any]]:
        """Kombiniere und gewichte Keyword- und Semantic-Ergebnisse"""
        
        # Erstelle kombinierte Liste mit Gewichtung
        combined = []
        
        # Keyword-Ergebnisse (Gewichtung: 0.6)
        for result in keyword_results:
            result = result.copy()
            result["final_score"] = result["similarity_score"] * 0.6
            result["search_methods"] = ["keyword"]
            combined.append(result)
        
        # Semantic-Ergebnisse (Gewichtung: 0.4)
        for result in semantic_results:
            result = result.copy()
            result["final_score"] = result["similarity_score"] * 0.4
            result["search_methods"] = ["semantic"]
            
            # Prüfe ob bereits ähnlicher Keyword-Treffer existiert
            is_duplicate = False
            for existing in combined:
                if (existing.get("document_id") == result.get("document_id") and
                    self._text_similarity(existing.get("text", ""), result.get("text", "")) > 0.7):
                    # Kombiniere Scores
                    existing["final_score"] = max(existing["final_score"], existing["final_score"] + result["final_score"] * 0.5)
                    existing["search_methods"].append("semantic")
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                combined.append(result)
        
        # Sortiere nach finalem Score
        combined.sort(key=lambda x: x["final_score"], reverse=True)
        
        return combined[:top_k]
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Einfache Textähnlichkeit basierend auf Wörtern"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0

# Integration in das bestehende System
async def initialize_vector_system(documents_db: dict) -> Tuple[VectorEmbeddingManager, HybridSearchManager]:
    """Initialisiere das Vector-System für alle Dokumente"""
    
    print("🚀 Initialisiere Vector Embedding System...")
    
    # Erstelle Vector Manager
    vector_manager = VectorEmbeddingManager()
    
    if not vector_manager.model:
        print("❌ Vector System nicht verfügbar")
        return None, None
    
    # Verarbeite alle Dokumente die noch keine Embeddings haben
    processed_docs = set(meta.get("document_id") for meta in vector_manager.chunk_metadata)
    
    for doc_id, doc in documents_db.items():
        if (doc_id not in processed_docs and 
            doc.get("processing_status") == "completed" and 
            doc.get("text_chunks")):
            
            print(f"📄 Verarbeite Embeddings für: {doc.get('filename')}")
            await vector_manager.process_document_chunks(
                doc_id, 
                doc.get("text_chunks", []), 
                doc
            )
    
    # Erstelle Hybrid Search Manager
    hybrid_manager = HybridSearchManager(documents_db, vector_manager)
    
    print("✅ Vector System initialisiert")
    return vector_manager, hybrid_manager

if __name__ == "__main__":
    # Test des Vector Systems
    async def test_vector_system():
        print("🧪 Vector System Test")
        
        # Lade Dokumente
        doc_file = Path("data/documents.json")
        if doc_file.exists():
            with open(doc_file, 'r', encoding='utf-8') as f:
                documents_db = json.load(f)
            
            vector_manager, hybrid_manager = await initialize_vector_system(documents_db)
            
            if vector_manager:
                stats = vector_manager.get_statistics()
                print(f"📊 Vector Store Statistiken: {stats}")
                
                if hybrid_manager:
                    # Teste Suche
                    results = await hybrid_manager.hybrid_search("abendkarte", list(documents_db.keys())[0] if documents_db else "")
                    print(f"🔍 Test-Suche: {len(results)} Ergebnisse")
        else:
            print("❌ Keine Dokumente gefunden")
    
    asyncio.run(test_vector_system())