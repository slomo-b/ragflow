#!/usr/bin/env python3
"""
RAG Document Processor für RagFlow
Extrahiert Text aus PDFs und anderen Dokumenten für AI-Analyse
"""

import os
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any
import asyncio
from datetime import datetime

# Document processing libraries
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    print("⚠️ PyPDF2 nicht verfügbar")

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    print("⚠️ pdfplumber nicht verfügbar")

try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    print("⚠️ python-docx nicht verfügbar")

class DocumentProcessor:
    """Verarbeitet verschiedene Dokumenttypen und extrahiert Text"""
    
    def __init__(self):
        self.supported_types = {
            'pdf': self._extract_pdf_text,
            'docx': self._extract_docx_text,
            'txt': self._extract_txt_text,
            'md': self._extract_txt_text  # Markdown wie Text behandeln
        }
    
    async def process_document(self, file_path: str, file_type: str) -> Dict[str, Any]:
        """
        Hauptfunktion zur Dokumentenverarbeitung
        
        Args:
            file_path: Pfad zur Datei
            file_type: Dateityp (pdf, docx, txt, md)
            
        Returns:
            Dict mit extrahiertem Text und Metadaten
        """
        try:
            print(f"🔄 Verarbeite Dokument: {file_path} (Typ: {file_type})")
            
            if file_type not in self.supported_types:
                return {
                    "success": False,
                    "error": f"Dateityp '{file_type}' wird nicht unterstützt",
                    "text": "",
                    "chunks": [],
                    "metadata": {}
                }
            
            # Text extrahieren
            extractor = self.supported_types[file_type]
            text = await extractor(file_path)
            
            if not text or len(text.strip()) < 10:
                return {
                    "success": False,
                    "error": "Kein Text extrahiert oder Text zu kurz",
                    "text": "",
                    "chunks": [],
                    "metadata": {}
                }
            
            # Text in Chunks aufteilen (für bessere AI-Verarbeitung)
            chunks = self._chunk_text(text)
            
            # Metadaten erstellen
            metadata = {
                "word_count": len(text.split()),
                "char_count": len(text),
                "chunk_count": len(chunks),
                "extraction_method": file_type,
                "processed_at": datetime.utcnow().isoformat()
            }
            
            print(f"✅ Verarbeitung erfolgreich: {metadata['word_count']} Wörter, {len(chunks)} Chunks")
            
            return {
                "success": True,
                "error": None,
                "text": text,
                "chunks": chunks,
                "metadata": metadata
            }
            
        except Exception as e:
            print(f"❌ Fehler bei Verarbeitung: {str(e)}")
            return {
                "success": False,
                "error": f"Fehler bei der Verarbeitung: {str(e)}",
                "text": "",
                "chunks": [],
                "metadata": {}
            }
    
    async def _extract_pdf_text(self, file_path: str) -> str:
        """Extrahiert Text aus PDF-Dateien"""
        try:
            # Versuch 1: pdfplumber (bessere Ergebnisse)
            if PDFPLUMBER_AVAILABLE:
                try:
                    import pdfplumber
                    with pdfplumber.open(file_path) as pdf:
                        text_parts = []
                        for page_num, page in enumerate(pdf.pages):
                            try:
                                page_text = page.extract_text()
                                if page_text and page_text.strip():
                                    # Säubere den Text
                                    cleaned_text = self._clean_extracted_text(page_text)
                                    if cleaned_text:
                                        text_parts.append(f"[Seite {page_num + 1}]\n{cleaned_text}")
                            except Exception as page_error:
                                print(f"Fehler bei Seite {page_num + 1}: {page_error}")
                                continue
                        
                        full_text = "\n\n".join(text_parts)
                        if full_text.strip():
                            print(f"✅ pdfplumber: {len(text_parts)} Seiten extrahiert")
                            return full_text
                except Exception as e:
                    print(f"pdfplumber Fehler: {e}")
            
            # Versuch 2: PyPDF2 (Fallback)
            if PYPDF2_AVAILABLE:
                try:
                    import PyPDF2
                    with open(file_path, 'rb') as file:
                        reader = PyPDF2.PdfReader(file)
                        text_parts = []
                        
                        for page_num, page in enumerate(reader.pages):
                            try:
                                page_text = page.extract_text()
                                if page_text and page_text.strip():
                                    cleaned_text = self._clean_extracted_text(page_text)
                                    if cleaned_text:
                                        text_parts.append(f"[Seite {page_num + 1}]\n{cleaned_text}")
                            except Exception as page_error:
                                print(f"PyPDF2 Fehler bei Seite {page_num + 1}: {page_error}")
                                continue
                        
                        full_text = "\n\n".join(text_parts)
                        if full_text.strip():
                            print(f"✅ PyPDF2: {len(text_parts)} Seiten extrahiert")
                            return full_text
                except Exception as e:
                    print(f"PyPDF2 Fehler: {e}")
            
            print("❌ Keine funktionsfähige PDF-Bibliothek gefunden")
            return "PDF-Text konnte nicht extrahiert werden. Bitte installiere: pip install pdfplumber PyPDF2"
            
        except Exception as e:
            print(f"PDF Extraction Fehler: {e}")
            return ""
    
    async def _extract_docx_text(self, file_path: str) -> str:
        """Extrahiert Text aus DOCX-Dateien"""
        if not DOCX_AVAILABLE:
            return "DOCX-Verarbeitung nicht verfügbar. Installiere: pip install python-docx"
            
        try:
            from docx import Document
            doc = Document(file_path)
            
            text_parts = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_parts.append(paragraph.text.strip())
            
            # Extrahiere auch Text aus Tabellen
            for table in doc.tables:
                table_text = []
                for row in table.rows:
                    row_text = []
                    for cell in row.cells:
                        if cell.text.strip():
                            row_text.append(cell.text.strip())
                    if row_text:
                        table_text.append(" | ".join(row_text))
                
                if table_text:
                    text_parts.append("\n[Tabelle]\n" + "\n".join(table_text))
            
            full_text = "\n\n".join(text_parts)
            if full_text.strip():
                print(f"✅ DOCX: Text extrahiert ({len(text_parts)} Abschnitte)")
                return full_text
            
            return ""
            
        except Exception as e:
            print(f"DOCX Extraction Fehler: {e}")
            return ""
    
    async def _extract_txt_text(self, file_path: str) -> str:
        """Extrahiert Text aus TXT/MD-Dateien"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                print(f"✅ TXT: {len(text)} Zeichen extrahiert")
                return text
        except UnicodeDecodeError:
            # Fallback für andere Encodings
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    text = file.read()
                    print(f"✅ TXT (Latin-1): {len(text)} Zeichen extrahiert")
                    return text
            except Exception as e:
                print(f"TXT Extraction Fehler: {e}")
                return ""
        except Exception as e:
            print(f"TXT Extraction Fehler: {e}")
            return ""
    
    def _clean_extracted_text(self, text: str) -> str:
        """Säubert extrahierten Text von häufigen Problemen"""
        if not text:
            return ""
        
        # Entferne übermäßige Leerzeichen und Zeilenumbrüche
        import re
        
        # Ersetze mehrfache Leerzeichen durch einzelne
        text = re.sub(r' +', ' ', text)
        
        # Ersetze mehrfache Zeilenumbrüche durch maximal zwei
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Entferne Leerzeichen am Zeilenanfang und -ende
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        # Entferne leere Zeilen am Anfang und Ende
        text = text.strip()
        
        return text
    
    def _chunk_text(self, text: str, max_chunk_size: int = 1000, overlap: int = 100) -> List[Dict[str, Any]]:
        """
        Teilt Text in überlappende Chunks für bessere AI-Verarbeitung
        
        Args:
            text: Zu teilender Text
            max_chunk_size: Maximale Chunk-Größe in Zeichen
            overlap: Überlappung zwischen Chunks
            
        Returns:
            Liste von Chunks mit Metadaten
        """
        if len(text) <= max_chunk_size:
            return [{
                "id": str(uuid.uuid4()),
                "text": text,
                "start_pos": 0,
                "end_pos": len(text),
                "chunk_index": 0
            }]
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = min(start + max_chunk_size, len(text))
            
            # Versuche an Satzende zu teilen
            if end < len(text):
                # Suche nach dem letzten Punkt vor dem Ende
                last_sentence = text.rfind('.', start, end)
                if last_sentence > start + max_chunk_size * 0.5:
                    end = last_sentence + 1
                else:
                    # Suche nach Leerzeichen
                    last_space = text.rfind(' ', start, end)
                    if last_space > start + max_chunk_size * 0.5:
                        end = last_space
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "id": str(uuid.uuid4()),
                    "text": chunk_text,
                    "start_pos": start,
                    "end_pos": end,
                    "chunk_index": chunk_index
                })
                chunk_index += 1
            
            # Nächster Start mit Überlappung
            start = max(start + 1, end - overlap)
        
        return chunks

# RAG Integration für Chat
class RAGChatEnhancer:
    """Erweitert Chat-Anfragen um Dokumentenkontext"""
    
    def __init__(self, documents_db: dict):
        self.documents_db = documents_db
    
    def find_relevant_content(self, query: str, project_id: str, max_chunks: int = 3) -> List[str]:
        """
        Findet relevante Dokumenteninhalte für eine Anfrage
        
        Args:
            query: Benutzeranfrage
            project_id: Projekt-ID
            max_chunks: Maximale Anzahl relevanter Chunks
            
        Returns:
            Liste relevanter Textpassagen
        """
        relevant_content = []
        
        print(f"🔍 Suche relevante Inhalte für Query: '{query}' in Projekt: {project_id}")
        
        # Finde Dokumente des Projekts
        project_docs = [
            doc for doc in self.documents_db.values()
            if project_id in doc.get("project_ids", []) and 
               doc.get("processing_status") == "completed" and
               doc.get("extracted_text")
        ]
        
        print(f"📄 Gefundene verarbeitete Dokumente: {len(project_docs)}")
        
        if not project_docs:
            return []
        
        query_lower = query.lower()
        
        for doc in project_docs:
            print(f"🔍 Durchsuche: {doc.get('filename', 'Unknown')}")
            
            # Einfache Keyword-Suche im vollständigen Text
            full_text = doc.get("extracted_text", "")
            if query_lower in full_text.lower():
                # Finde relevante Abschnitte um die Keywords
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
                for pos in positions[:max_chunks]:
                    context_start = max(0, pos - 200)
                    context_end = min(len(full_text), pos + 200)
                    context = full_text[context_start:context_end]
                    
                    relevant_content.append({
                        "text": context,
                        "source": doc.get("filename", "Unknown"),
                        "score": 1.0,
                        "position": pos
                    })
            
            # Suche auch in Chunks wenn verfügbar
            chunks = doc.get("text_chunks", [])
            for chunk in chunks[:10]:  # Begrenze auf erste 10 Chunks
                chunk_text = chunk.get("text", "").lower()
                
                # Prüfe auf Keyword-Übereinstimmungen
                query_words = query_lower.split()
                matches = sum(1 for word in query_words if word in chunk_text)
                
                if matches > 0:
                    relevance_score = matches / len(query_words)
                    relevant_content.append({
                        "text": chunk.get("text", ""),
                        "source": doc.get("filename", "Unknown"),
                        "score": relevance_score,
                        "chunk_id": chunk.get("id")
                    })
        
        # Sortiere nach Relevanz und begrenze Ergebnisse
        relevant_content.sort(key=lambda x: x["score"], reverse=True)
        result_texts = [item["text"] for item in relevant_content[:max_chunks]]
        
        print(f"✅ Gefunden: {len(result_texts)} relevante Textpassagen")
        
        return result_texts
    
    def enhance_prompt_with_context(self, user_query: str, project_id: str, base_prompt: str) -> str:
        """
        Erweitert den Prompt um relevanten Dokumentenkontext
        
        Args:
            user_query: Benutzeranfrage
            project_id: Projekt-ID
            base_prompt: Basis System-Prompt
            
        Returns:
            Erweiterter Prompt mit Dokumentenkontext
        """
        relevant_content = self.find_relevant_content(user_query, project_id)
        
        if not relevant_content:
            print("ℹ️ Keine relevanten Dokumenteninhalte gefunden")
            return base_prompt
        
        # Füge Dokumentenkontext hinzu
        context_section = "\n\nRELEVANTE DOKUMENTENINHALTE:\n"
        context_section += "=" * 50 + "\n"
        
        for i, content in enumerate(relevant_content, 1):
            context_section += f"\n[Abschnitt {i}]\n"
            context_section += content[:800] + ("..." if len(content) > 800 else "")
            context_section += "\n" + "-" * 30 + "\n"
        
        context_section += "\nBitte beziehe dich in deiner Antwort auf diese Dokumenteninhalte, wenn sie relevant für die Frage sind.\n"
        
        print(f"✅ Prompt erweitert mit {len(relevant_content)} Dokumentenabschnitten")
        
        return base_prompt + context_section

# Utility-Funktionen
async def process_uploaded_document(file_path: str, file_type: str, document_id: str, documents_db: dict):
    """
    Verarbeitet ein hochgeladenes Dokument und aktualisiert die Datenbank
    
    Args:
        file_path: Pfad zur Datei
        file_type: Dateityp
        document_id: Dokument-ID in der DB
        documents_db: Dokumenten-Datenbank
    """
    print(f"🚀 Starte Dokumentenverarbeitung: {file_path}")
    
    processor = DocumentProcessor()
    
    # Dokument verarbeiten
    result = await processor.process_document(file_path, file_type)
    
    # Datenbank aktualisieren
    if document_id in documents_db:
        doc = documents_db[document_id]
        
        if result["success"]:
            doc["processing_status"] = "completed"
            doc["extracted_text"] = result["text"]
            doc["text_chunks"] = result["chunks"]
            doc["text_metadata"] = result["metadata"]
            doc["processing_error"] = None
            print(f"✅ Dokument {doc['filename']} erfolgreich verarbeitet")
        else:
            doc["processing_status"] = "failed"
            doc["processing_error"] = result["error"]
            doc["extracted_text"] = ""
            doc["text_chunks"] = []
            print(f"❌ Fehler bei {doc['filename']}: {result['error']}")
        
        doc["processed_at"] = datetime.utcnow().isoformat()
    else:
        print(f"⚠️ Dokument-ID {document_id} nicht in Datenbank gefunden")

if __name__ == "__main__":
    # Test der Klassen
    print("🧪 DocumentProcessor Test")
    processor = DocumentProcessor()
    print(f"Unterstützte Dateitypen: {list(processor.supported_types.keys())}")
    print(f"PDF verfügbar: {PDFPLUMBER_AVAILABLE or PYPDF2_AVAILABLE}")
    print(f"DOCX verfügbar: {DOCX_AVAILABLE}")