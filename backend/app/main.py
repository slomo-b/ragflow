#!/usr/bin/env python3
"""
Enhanced RagFlow Backend - Intelligente RAG-Integration mit proaktiver Dokumentenanalyse
Version 2.3.0 - Intelligente Chat-Features
"""

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import json
from pathlib import Path
from datetime import datetime
import uuid
import asyncio
from dotenv import load_dotenv
import re
from collections import Counter

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

# === Enhanced Intelligence Classes ===

class IntelligentDocumentAnalyzer:
    """Intelligente Dokumentenanalyse für bessere AI-Antworten"""
    
    def __init__(self, documents_db: dict):
        self.documents_db = documents_db
    
    def extract_keywords_from_text(self, text: str, max_keywords: int = 15) -> list:
        """Extrahiere wichtige Schlüsselwörter aus Text"""
        if not text:
            return []
        
        # Entferne Sonderzeichen und normalisiere
        words = re.findall(r'\b\w{3,}\b', text.lower())
        
        # Deutsche und englische Stopwörter
        stopwords = {
            'der', 'die', 'das', 'und', 'oder', 'aber', 'für', 'mit', 'auf', 'ist', 'sind',
            'eine', 'ein', 'von', 'zu', 'im', 'am', 'um', 'an', 'als', 'bei', 'nach', 'vor',
            'über', 'unter', 'durch', 'ohne', 'gegen', 'bis', 'seit', 'während', 'wegen',
            'the', 'and', 'or', 'but', 'for', 'with', 'is', 'are', 'a', 'an', 'of', 'to',
            'in', 'on', 'at', 'by', 'from', 'this', 'that', 'these', 'those', 'will', 'would'
        }
        
        filtered_words = [word for word in words if word not in stopwords and len(word) > 3]
        word_counts = Counter(filtered_words)
        
        return [word for word, count in word_counts.most_common(max_keywords)]
    
    def create_intelligent_document_context(self, project_id: str) -> str:
        """Erstelle intelligenten Dokumentenkontext für bessere AI-Antworten"""
        
        project_docs = [
            doc for doc in self.documents_db.values()
            if project_id in doc.get("project_ids", []) and 
               doc.get("processing_status") == "completed"
        ]
        
        if not project_docs:
            return "❌ Keine verarbeiteten Dokumente verfügbar für intelligente Analyse."
        
        context_parts = ["🧠 INTELLIGENTE DOKUMENTENANALYSE - VERFÜGBARE INHALTE:"]
        context_parts.append("=" * 80)
        
        for i, doc in enumerate(project_docs, 1):
            doc_name = doc.get("filename", "Unbekanntes Dokument")
            doc_type = doc.get("file_type", "").upper()
            
            # Extrahiere Text und Schlüsselwörter
            extracted_text = doc.get("extracted_text", "")
            keywords = self.extract_keywords_from_text(extracted_text)
            
            # Erstelle Inhaltsvorschau
            content_preview = extracted_text[:400] + "..." if len(extracted_text) > 400 else extracted_text
            
            # Zähle Wörter und Zeichen
            word_count = len(extracted_text.split()) if extracted_text else 0
            char_count = len(extracted_text)
            
            context_parts.append(f"""
📄 DOKUMENT {i}: {doc_name} ({doc_type})
📊 STATISTIKEN: {word_count} Wörter, {char_count} Zeichen
🔍 SCHLÜSSELWÖRTER: {', '.join(keywords[:8])}
📝 INHALT (Vorschau):
{content_preview}
{'-' * 60}""")
        
        context_parts.append(f"""
🎯 ANALYSEAUFTRAG:
Bei JEDER Benutzeranfrage analysiere diese {len(project_docs)} Dokumente:

1. 🔍 AUTOMATISCHE SUCHE: Durchsuche alle Inhalte nach relevanten Informationen
2. 📝 KONKRETE ANTWORTEN: Gib spezifische Antworten basierend auf tatsächlichen Inhalten  
3. 📄 QUELLENANGABEN: Zitiere immer den Dateinamen und relevante Textpassagen
4. 💡 PROAKTIVE HILFE: Bei unklaren Anfragen zeige verfügbare Optionen auf
5. 🚀 INTELLIGENTE SUCHE: Erkenne ähnliche Begriffe und Konzepte automatisch

❌ NIEMALS sagen: "Bitte geben Sie den vollständigen Dateinamen an"
✅ STATTDESSEN: Dokumente durchsuchen und relevante Treffer präsentieren
""")
        
        return "\n".join(context_parts)

class IntelligentChatEnhancer:
    """Verbessert Chat-Anfragen mit intelligenter Dokumentensuche"""
    
    def __init__(self, documents_db: dict):
        self.documents_db = documents_db
        self.analyzer = IntelligentDocumentAnalyzer(documents_db)
    
    def detect_user_intent(self, query: str, available_docs: list) -> dict:
        """Erkenne Benutzerintention"""
        
        intents = {
            "file_search": ["datei", "file", "txt", "pdf", "dokument", "steht", "inhalt"],
            "content_analysis": ["inhalt", "content", "analyze", "analysiere", "was"],
            "summary": ["zusammenfassung", "summary", "überblick", "fasse", "zusammen"],
            "comparison": ["vergleich", "compare", "unterschied", "vergleiche"],
            "specific_info": ["info", "information", "details", "suche", "finde", "zeige"],
            "list_files": ["dateien", "files", "liste", "übersicht", "verfügbar"]
        }
        
        query_lower = query.lower()
        detected_intent = "general"
        confidence = 0.0
        
        for intent, keywords in intents.items():
            matches = sum(1 for keyword in keywords if keyword in query_lower)
            intent_confidence = matches / len(keywords) if keywords else 0
            
            if intent_confidence > confidence:
                confidence = intent_confidence
                detected_intent = intent
        
        return {
            "intent": detected_intent,
            "confidence": confidence,
            "suggested_action": self._get_action_for_intent(detected_intent, available_docs)
        }
    
    def _get_action_for_intent(self, intent: str, docs: list) -> str:
        """Schlage konkrete Aktionen vor"""
        
        actions = {
            "file_search": f"🔍 Durchsuche {len(docs)} verfügbare Dokumente nach relevanten Inhalten",
            "content_analysis": "📝 Analysiere Inhalte aller verarbeiteten Dokumente",
            "summary": "📋 Erstelle Zusammenfassung aller Dokumente im Projekt",
            "comparison": "⚖️ Vergleiche Inhalte der verfügbaren Dokumente",
            "specific_info": "🎯 Suche spezifische Informationen in allen Dokumenten",
            "list_files": "📂 Zeige Übersicht aller verfügbaren Dokumente"
        }
        
        return actions.get(intent, "🤖 Allgemeine intelligente Dokumentenanalyse")
    
    def fuzzy_match(self, query: str, filename: str, threshold: float = 0.4) -> bool:
        """Intelligentes Fuzzy Matching für Dateinamen"""
        
        # Normalisiere beide Strings
        query_clean = re.sub(r'[^a-zA-Z0-9]', '', query.lower())
        filename_clean = re.sub(r'[^a-zA-Z0-9]', '', filename.lower())
        
        # Prüfe auf Teilstrings
        if query_clean in filename_clean or filename_clean in query_clean:
            return True
        
        # Jaccard Similarity mit Zeichen-N-Grammen
        def get_ngrams(s, n=2):
            return set(s[i:i+n] for i in range(len(s)-n+1))
        
        query_ngrams = get_ngrams(query_clean)
        filename_ngrams = get_ngrams(filename_clean)
        
        if not query_ngrams or not filename_ngrams:
            return False
        
        intersection = len(query_ngrams.intersection(filename_ngrams))
        union = len(query_ngrams.union(filename_ngrams))
        
        similarity = intersection / union if union > 0 else 0
        return similarity >= threshold
    
    def enhance_user_query(self, query: str, project_id: str) -> tuple[str, list, dict]:
        """Verbessere Benutzeranfrage und finde relevante Dokumente"""
        
        # Finde alle Dokumente des Projekts
        project_docs = [
            doc for doc in self.documents_db.values()
            if project_id in doc.get("project_ids", []) and 
               doc.get("processing_status") == "completed"
        ]
        
        if not project_docs:
            return query, [], {"intent": "no_docs", "confidence": 1.0}
        
        # Erkenne Intent
        intent_info = self.detect_user_intent(query, project_docs)
        
        # Suche nach relevanten Dokumenten
        query_lower = query.lower()
        filename_matches = []
        content_matches = []
        
        for doc in project_docs:
            doc_name = doc.get("filename", "")
            extracted_text = doc.get("extracted_text", "")
            
            # Fuzzy Matching für Dateinamen
            if self.fuzzy_match(query_lower, doc_name):
                filename_matches.append(doc)
            
            # Inhaltssuche
            query_words = [word for word in query_lower.split() if len(word) > 2]
            if query_words and any(word in extracted_text.lower() for word in query_words):
                content_matches.append(doc)
        
        # Kombiniere Ergebnisse
        relevant_docs = list(set(filename_matches + content_matches))
        
        # Erweitere Query mit Kontext
        enhanced_query = query
        
        if relevant_docs:
            enhanced_query += f"\n\n🎯 RELEVANTE DOKUMENTE GEFUNDEN ({len(relevant_docs)}):\n"
            
            for doc in relevant_docs[:3]:  # Zeige max 3 relevante Docs
                doc_name = doc.get("filename", "")
                content_preview = doc.get("extracted_text", "")[:200]
                enhanced_query += f"📄 {doc_name}: {content_preview}...\n"
        
        return enhanced_query, relevant_docs, intent_info

class EnhancedRAGChatEnhancer:
    """Erweiterte RAG-Integration mit intelligenter Suche"""
    
    def __init__(self, documents_db: dict):
        self.documents_db = documents_db
        self.chat_enhancer = IntelligentChatEnhancer(documents_db)
        self.analyzer = IntelligentDocumentAnalyzer(documents_db)
    
    def find_relevant_content(self, query: str, project_id: str, max_chunks: int = 5) -> List[Dict[str, Any]]:
        """Intelligente Suche nach relevanten Dokumenteninhalten"""
        
        print(f"🧠 Intelligente Suche für: '{query}' in Projekt: {project_id}")
        
        # Finde Dokumente des Projekts
        project_docs = [
            doc for doc in self.documents_db.values()
            if project_id in doc.get("project_ids", []) and 
               doc.get("processing_status") == "completed" and
               doc.get("extracted_text")
        ]
        
        if not project_docs:
            print("❌ Keine verarbeiteten Dokumente gefunden")
            return []
        
        relevant_content = []
        query_lower = query.lower()
        query_words = [word for word in query_lower.split() if len(word) > 2]
        
        for doc in project_docs:
            doc_name = doc.get("filename", "Unknown")
            full_text = doc.get("extracted_text", "")
            
            if not full_text:
                continue
            
            # Mehrschichtige Suche
            text_lower = full_text.lower()
            
            # 1. Exakte Phrase-Suche
            if query_lower in text_lower:
                positions = []
                start = 0
                while True:
                    pos = text_lower.find(query_lower, start)
                    if pos == -1:
                        break
                    positions.append(pos)
                    start = pos + 1
                
                for pos in positions[:2]:  # Max 2 exakte Treffer pro Dokument
                    context_start = max(0, pos - 250)
                    context_end = min(len(full_text), pos + 250)
                    context = full_text[context_start:context_end]
                    
                    relevant_content.append({
                        "text": context,
                        "source": doc_name,
                        "score": 1.0,
                        "match_type": "exact_phrase",
                        "position": pos
                    })
            
            # 2. Keyword-basierte Suche
            if query_words:
                sentences = full_text.split('.')
                for i, sentence in enumerate(sentences):
                    sentence_lower = sentence.lower()
                    matches = sum(1 for word in query_words if word in sentence_lower)
                    
                    if matches > 0:
                        relevance_score = matches / len(query_words)
                        if relevance_score >= 0.3:  # Mindestens 30% der Wörter
                            # Erweitere Kontext um benachbarte Sätze
                            start_sentence = max(0, i - 1)
                            end_sentence = min(len(sentences), i + 2)
                            extended_context = '. '.join(sentences[start_sentence:end_sentence])
                            
                            relevant_content.append({
                                "text": extended_context,
                                "source": doc_name,
                                "score": relevance_score,
                                "match_type": "keyword",
                                "matched_words": matches
                            })
            
            # 3. Chunk-basierte Suche wenn verfügbar
            chunks = doc.get("text_chunks", [])
            for chunk in chunks[:5]:  # Begrenze auf erste 5 Chunks
                chunk_text = chunk.get("text", "").lower()
                
                if query_words:
                    matches = sum(1 for word in query_words if word in chunk_text)
                    if matches > 0:
                        relevance_score = matches / len(query_words)
                        
                        relevant_content.append({
                            "text": chunk.get("text", ""),
                            "source": doc_name,
                            "score": relevance_score * 0.8,  # Chunk-Treffer etwas niedriger gewichten
                            "match_type": "chunk",
                            "chunk_id": chunk.get("id")
                        })
        
        # Sortiere nach Relevanz und entferne Duplikate
        relevant_content.sort(key=lambda x: x["score"], reverse=True)
        
        # Entferne sehr ähnliche Treffer
        filtered_content = []
        for item in relevant_content:
            is_duplicate = False
            for existing in filtered_content:
                if (existing["source"] == item["source"] and 
                    len(set(item["text"].lower().split()).intersection(set(existing["text"].lower().split()))) > 5):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_content.append(item)
                
            if len(filtered_content) >= max_chunks:
                break
        
        print(f"✅ Intelligente Suche: {len(filtered_content)} relevante Inhalte gefunden")
        return filtered_content
    
    def enhance_prompt_with_context(self, user_query: str, project_id: str, base_prompt: str) -> str:
        """Erweitert den Prompt um intelligenten Dokumentenkontext"""
        
        # Intelligente Query-Verbesserung
        enhanced_query, relevant_docs, intent_info = self.chat_enhancer.enhance_user_query(user_query, project_id)
        
        # Finde relevante Inhalte
        relevant_content = self.find_relevant_content(user_query, project_id)
        
        # Erstelle Dokumentenkontext
        documents_context = self.analyzer.create_intelligent_document_context(project_id)
        
        if not relevant_content and not relevant_docs:
            print("ℹ️ Keine relevanten Dokumenteninhalte gefunden - verwende allgemeinen Kontext")
            return base_prompt + "\n\n" + documents_context
        
        # Baue erweiterten Prompt auf
        enhanced_prompt = base_prompt + "\n\n" + documents_context
        
        if relevant_content:
            enhanced_prompt += f"\n\n🎯 SPEZIFISCH RELEVANTE INHALTE FÜR ANFRAGE '{user_query}':\n"
            enhanced_prompt += "=" * 80 + "\n"
            
            for i, content in enumerate(relevant_content, 1):
                match_info = f"({content['match_type']}, Score: {content['score']:.2f})"
                enhanced_prompt += f"\n📄 TREFFER {i} aus {content['source']} {match_info}:\n"
                enhanced_prompt += f"{content['text'][:600]}...\n"
                enhanced_prompt += "-" * 60 + "\n"
        
        # Füge Intent-Information hinzu
        enhanced_prompt += f"\n\n🧠 ERKANNTE BENUTZERINTENTION: {intent_info['intent']}\n"
        enhanced_prompt += f"🎯 EMPFOHLENE AKTION: {intent_info['suggested_action']}\n"
        
        enhanced_prompt += """
📋 ANTWORT-ANWEISUNGEN:
1. Beziehe dich direkt auf die gefundenen Inhalte
2. Zitiere konkrete Passagen aus den Dokumenten  
3. Nenne immer die Quelldateien
4. Gib spezifische, inhaltsbasierte Antworten
5. Bei unklaren Begriffen: Suche nach ähnlichen Konzepten in den Dokumenten
"""
        
        print(f"✅ Prompt erweitert mit {len(relevant_content)} relevanten Inhalten")
        return enhanced_prompt

# === Enhanced Helper Functions ===

def get_enhanced_system_prompt(project_context: str = "", documents_context: str = "") -> str:
    """Erstelle einen intelligenten System-Prompt der proaktiv agiert"""
    
    base_prompt = """🤖 Du bist ein hochintelligenter AI-Assistent für RagFlow - spezialisiert auf proaktive Dokumentenanalyse.

🎯 DEINE HAUPTAUFGABEN:
1. 🔍 AUTOMATISCH relevante Dokumente identifizieren und durchsuchen
2. 📄 Den Inhalt verfügbarer Dateien PROAKTIV analysieren  
3. 💡 Intelligente Antworten basierend auf Dokumenteninhalten geben
4. 🚀 Benutzer zu besseren Fragen führen
5. 🎯 Bei unklaren Anfragen: Verfügbare Optionen aufzeigen

🧠 INTELLIGENTES VERHALTEN:
- Analysiere ALLE verfügbaren Dokumente bei jeder Anfrage
- Suche nach ähnlichen Begriffen und Konzepten automatisch
- Biete konkrete Alternativen basierend auf verfügbaren Inhalten
- Zeige dem Benutzer immer WAS verfügbar ist

❌ NIEMALS sagen: "Bitte geben Sie den vollständigen Dateinamen an"
✅ STATTDESSEN: Dokumente durchsuchen und relevante Treffer anzeigen

📝 ANTWORT-STIL:
- Antworte präzise und hilfreich auf Deutsch
- Verwende Emojis zur besseren Strukturierung
- Zitiere immer Quelldateien bei Dokumenteninhalten
- Sei proaktiv und denke mit"""

    if project_context:
        base_prompt += f"\n\n📂 AKTUELLER PROJEKTKONTEXT:\n{project_context}"
    
    if documents_context:
        base_prompt += f"\n\n{documents_context}"
    
    return base_prompt

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
    
    # Sammle Dokument-Informationen
    project_docs = [
        doc for doc in documents_db.values() 
        if project_id in doc.get("project_ids", [])
    ]
    
    context_parts = [f"📂 AKTUELLES PROJEKT: '{project_name}'"]
    
    if project_description:
        context_parts.append(f"📋 BESCHREIBUNG: {project_description}")
    
    if project_docs:
        processed_docs = [doc for doc in project_docs if doc.get("processing_status") == "completed"]
        context_parts.append(f"📊 PROJEKT-STATISTIKEN: {len(project_docs)} Dokument(e), davon {len(processed_docs)} verarbeitet")
        
        context_parts.append("📄 VERFÜGBARE DOKUMENTE:")
        for i, doc in enumerate(project_docs[:5], 1):  # Zeige max. 5 Dokumente
            doc_name = doc.get("filename", "Unbekanntes Dokument")
            doc_type = doc.get("file_type", "").upper()
            status = doc.get("processing_status", "uploaded")
            status_emoji = {"completed": "✅", "processing": "🔄", "failed": "❌"}.get(status, "📄")
            
            context_parts.append(f"   {i}. {status_emoji} {doc_name} ({doc_type})")
        
        if len(project_docs) > 5:
            context_parts.append(f"   ... und {len(project_docs) - 5} weitere Dokumente")
    else:
        context_parts.append("📄 DOKUMENTE: Keine Dokumente hochgeladen")
    
    return "\n".join(context_parts)

# === FastAPI App ===
app = FastAPI(
    title="RagFlow Backend Enhanced",
    description="AI-powered document analysis mit intelligenter RAG-Integration",
    version="2.3.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === API Routes ===

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "🚀 RagFlow Backend Enhanced läuft!",
        "version": "2.3.0",
        "features": [
            "✅ Intelligente RAG-Integration",
            "✅ Proaktive Dokumentenanalyse", 
            "✅ Fuzzy Search & Intent Recognition",
            "✅ Automatische Inhaltssuche",
            "✅ Smarte Fallback-Antworten",
            "✅ Enhanced User Experience"
        ],
        "intelligence": {
            "document_analysis": "Automatische Analyse aller Dokumenteninhalte",
            "smart_search": "Intelligente Suche auch bei unklaren Anfragen",
            "intent_recognition": "Erkennung von Benutzerintentionen",
            "proactive_help": "Proaktive Hilfe basierend auf verfügbaren Inhalten"
        }
    }

@app.get("/api/health")
async def health_check():
    """Enhanced health check with intelligence features"""
    gemini = await get_gemini_service()
    
    processed_docs = sum(1 for doc in documents_db.values() if doc.get("processing_status") == "completed")
    total_content_length = sum(len(doc.get("extracted_text", "")) for doc in documents_db.values())
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.3.0",
        "services": {
            "gemini_ai": "connected" if gemini else "disconnected",
            "intelligent_rag": "operational",
            "document_analyzer": "operational",
            "intent_recognition": "operational"
        },
        "intelligence_stats": {
            "total_projects": len(projects_db),
            "total_documents": len(documents_db),
            "processed_documents": processed_docs,
            "total_content_chars": total_content_length,
            "total_chats": len(chats_db),
            "avg_content_per_doc": total_content_length // max(processed_docs, 1)
        }
    }

@app.post("/api/v1/chat")
async def enhanced_chat_endpoint(request: ChatRequest):
    """Enhanced Chat with intelligent RAG integration"""
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
        
        print(f"🧠 Enhanced Chat - Benutzeranfrage: '{user_message}'")
        
        # === INTELLIGENTE FEATURES ===
        
        # 1. Projektkontext aufbauen
        project_context = ""
        if request.project_id:
            project_context = get_project_context(request.project_id, projects_db, documents_db)
        
        # 2. Intelligente RAG-Enhancement
        rag_enhancer = EnhancedRAGChatEnhancer(documents_db)
        
        # 3. Erstelle intelligenten System-Prompt
        base_system_prompt = get_enhanced_system_prompt(project_context)
        
        # 4. Erweitere Prompt mit intelligentem Dokumentenkontext
        if request.project_id:
            enhanced_prompt = rag_enhancer.enhance_prompt_with_context(
                user_message,
                request.project_id,
                base_system_prompt
            )
        else:
            enhanced_prompt = base_system_prompt
        
        # 5. Sammle relevante Quellen für die Antwort
        relevant_sources = []
        if request.project_id:
            relevant_content = rag_enhancer.find_relevant_content(user_message, request.project_id)
            relevant_sources = [
                {
                    "type": "intelligent_search",
                    "filename": content["source"],
                    "excerpt": content["text"][:200] + "...",
                    "relevance_score": content["score"],
                    "match_type": content["match_type"]
                }
                for content in relevant_content[:3]
            ]
        
        # 6. Baue vollständige Conversation zusammen
        conversation_history = ""
        if len(request.messages) > 1:
            for msg in request.messages[:-1]:
                role_german = "Benutzer" if msg.role == "user" else "Assistent"
                conversation_history += f"{role_german}: {msg.content}\n"
        
        # 7. Erstelle finale Prompt
        full_prompt = f"""{enhanced_prompt}

{conversation_history}

🎯 AKTUELLE BENUTZERANFRAGE: {user_message}

Analysiere die verfügbaren Dokumente und gib eine intelligente, inhaltsbasierte Antwort."""
        
        # Debug: Logge die Prompt-Länge
        print(f"📝 Enhanced Prompt-Länge: {len(full_prompt)} Zeichen")
        
        # 8. Generiere intelligente Antwort
        response = gemini.generate_content(full_prompt)
        ai_response = response.text if response else "Entschuldigung, ich konnte keine Antwort generieren."
        
        # 9. Speichere Chat mit erweiterten Metadaten
        chat_id = str(uuid.uuid4())
        chats_db[chat_id] = {
            "id": chat_id,
            "project_id": request.project_id,
            "project_name": projects_db.get(request.project_id, {}).get("name", "") if request.project_id else "",
            "messages": [msg.dict() for msg in request.messages],
            "response": ai_response,
            "timestamp": datetime.utcnow().isoformat(),
            "model": "gemini-1.5-flash",
            "enhanced_features": {
                "intelligent_rag": True,
                "proactive_analysis": True,
                "context_enhanced": bool(project_context),
                "sources_found": len(relevant_sources),
                "prompt_enhanced": True
            }
        }
        
        # 10. Update Chat-Zähler des Projekts
        if request.project_id and request.project_id in projects_db:
            projects_db[request.project_id]["chat_count"] = projects_db[request.project_id].get("chat_count", 0) + 1
            projects_db[request.project_id]["updated_at"] = datetime.utcnow().isoformat()
        
        print(f"✅ Enhanced Chat erfolgreich - {len(relevant_sources)} Quellen verwendet")
        
        return {
            "response": ai_response,
            "chat_id": chat_id,
            "project_id": request.project_id,
            "timestamp": datetime.utcnow().isoformat(),
            "model_info": {
                "model": "gemini-1.5-flash",
                "version": "enhanced-2.3.0",
                "features_used": {
                    "intelligent_document_search": True,
                    "proactive_analysis": True,
                    "intent_recognition": True,
                    "fuzzy_matching": True,
                    "context_enhancement": True
                }
            },
            "sources": relevant_sources,
            "intelligence_metadata": {
                "sources_analyzed": len(relevant_sources),
                "context_enhanced": bool(project_context),
                "prompt_length": len(full_prompt)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Enhanced Chat Fehler: {str(e)}")
        return {
            "response": f"Entschuldigung, es gab einen Fehler bei der intelligenten Analyse: {str(e)}",
            "status": "error",
            "timestamp": datetime.utcnow().isoformat()
        }

@app.post("/api/v1/chat/test")
async def test_enhanced_gemini():
    """Test enhanced Gemini API connection"""
    gemini = await get_gemini_service()
    
    if not gemini:
        return {
            "status": "error",
            "message": "Google API Key nicht konfiguriert oder ungültig",
            "hint": "Setze GOOGLE_API_KEY in der .env Datei"
        }
    
    try:
        test_prompt = """Du bist der intelligente RagFlow AI-Assistent. 
        
Antworte nur mit: 'RagFlow Enhanced Backend funktioniert perfekt mit intelligenter RAG-Integration! 🚀🧠'"""
        
        response = gemini.generate_content(test_prompt)
        
        return {
            "status": "success",
            "message": "✅ Enhanced Gemini API funktioniert einwandfrei!",
            "gemini_response": response.text if response else "Keine Antwort",
            "timestamp": datetime.utcnow().isoformat(),
            "model": "gemini-1.5-flash",
            "features": ["intelligent_rag", "proactive_analysis", "enhanced_prompts"]
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Enhanced Gemini API Fehler: {str(e)}",
            "hint": "Überprüfe deinen API-Schlüssel unter https://ai.google.dev"
        }

# === Project Endpoints (wie vorher) ===

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
        print(f"📂 Neues Projekt erstellt: {project.name} ({project_id})")
        
        return new_project
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create project: {str(e)}")

@app.get("/api/v1/projects/")
async def get_projects(skip: int = 0, limit: int = 10, search: Optional[str] = None):
    """Get all projects"""
    projects = list(projects_db.values())
    
    if search:
        search_lower = search.lower()
        projects = [
            p for p in projects 
            if search_lower in p["name"].lower() or 
               (p.get("description") and search_lower in p["description"].lower())
        ]
    
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

# === Document Endpoints mit Enhanced Processing ===

@app.post("/api/v1/upload/documents")
async def upload_documents(
    files: List[UploadFile] = File(...),
    project_id: Optional[str] = Form(None),
    tags: Optional[str] = Form(None)
):
    """Upload documents with enhanced processing"""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
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
            
            # Create enhanced document metadata
            document = {
                "id": doc_id,
                "filename": file.filename,
                "safe_filename": safe_filename,
                "file_path": str(file_path),
                "file_type": file_extension[1:],
                "file_size": len(content),
                "uploaded_at": datetime.utcnow().isoformat(),
                "processing_status": "uploaded",
                "project_ids": [project_id] if project_id else [],
                "tags": tags.split(",") if tags else [],
                "extracted_text": "",
                "text_chunks": [],
                "text_metadata": {},
                "processing_error": None,
                "processed_at": None,
                "intelligence_features": {
                    "keywords_extracted": False,
                    "content_analyzed": False,
                    "search_ready": False
                }
            }
            
            documents_db[doc_id] = document
            uploaded_documents.append(document)
            
            # Start enhanced background processing
            print(f"🚀 Starte Enhanced Processing für {file.filename}...")
            asyncio.create_task(
                enhanced_process_document(
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
        "message": f"Successfully uploaded {len(uploaded_documents)} document(s). Enhanced processing started in background.",
        "documents": uploaded_documents,
        "features": ["intelligent_analysis", "keyword_extraction", "enhanced_search_preparation"]
    }

async def enhanced_process_document(file_path: str, file_type: str, document_id: str, documents_db: dict):
    """Enhanced document processing with intelligence features"""
    
    try:
        print(f"🧠 Enhanced Processing für Dokument {document_id}")
        
        # Import document processor
        try:
            from document_processor import DocumentProcessor
            processor = DocumentProcessor()
        except ImportError:
            print("⚠️ DocumentProcessor nicht verfügbar - verwende Fallback")
            # Einfacher Fallback für Textdateien
            if file_type in ['txt', 'md']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                # Simuliere Processing-Ergebnis
                result = {
                    "success": True,
                    "text": text,
                    "chunks": [{"id": str(uuid.uuid4()), "text": text, "start_pos": 0, "end_pos": len(text), "chunk_index": 0}],
                    "metadata": {
                        "word_count": len(text.split()),
                        "char_count": len(text),
                        "chunk_count": 1,
                        "extraction_method": file_type,
                        "processed_at": datetime.utcnow().isoformat()
                    }
                }
            else:
                result = {"success": False, "error": "Document processor not available"}
        
        if not result.get("success"):
            # Normale Verarbeitung
            result = await processor.process_document(file_path, file_type)
        
        # Update document in database
        if document_id in documents_db:
            doc = documents_db[document_id]
            
            if result["success"]:
                # Enhanced processing successful
                doc["processing_status"] = "completed"
                doc["extracted_text"] = result["text"]
                doc["text_chunks"] = result["chunks"]
                doc["text_metadata"] = result["metadata"]
                doc["processing_error"] = None
                
                # Add intelligence features
                analyzer = IntelligentDocumentAnalyzer(documents_db)
                keywords = analyzer.extract_keywords_from_text(result["text"])
                
                doc["intelligence_features"] = {
                    "keywords_extracted": True,
                    "content_analyzed": True,
                    "search_ready": True,
                    "keywords": keywords,
                    "content_preview": result["text"][:500] + "..." if len(result["text"]) > 500 else result["text"]
                }
                
                print(f"✅ Enhanced Processing erfolgreich für {doc['filename']}")
                print(f"   📊 {len(keywords)} Keywords extrahiert")
                print(f"   📝 {result['metadata'].get('word_count', 0)} Wörter verarbeitet")
                
            else:
                doc["processing_status"] = "failed"
                doc["processing_error"] = result["error"]
                doc["extracted_text"] = ""
                doc["text_chunks"] = []
                print(f"❌ Enhanced Processing fehlgeschlagen für {doc['filename']}: {result['error']}")
            
            doc["processed_at"] = datetime.utcnow().isoformat()
    
    except Exception as e:
        print(f"❌ Enhanced Processing Fehler: {e}")
        if document_id in documents_db:
            documents_db[document_id]["processing_status"] = "failed"
            documents_db[document_id]["processing_error"] = str(e)

@app.get("/api/v1/documents/")
async def get_documents(skip: int = 0, limit: int = 10, project_id: Optional[str] = None):
    """Get documents with enhanced metadata"""
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
        "limit": limit,
        "enhanced_features": ["keyword_extraction", "intelligent_search", "content_analysis"]
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
        if Path("data/projects.json").exists():
            with open("data/projects.json", "r", encoding="utf-8") as f:
                projects_db = json.load(f)
        
        if Path("data/documents.json").exists():
            with open("data/documents.json", "r", encoding="utf-8") as f:
                documents_db = json.load(f)
        
        if Path("data/chats.json").exists():
            with open("data/chats.json", "r", encoding="utf-8") as f:
                chats_db = json.load(f)
        
        processed_docs = sum(1 for doc in documents_db.values() if doc.get("processing_status") == "completed")
        print(f"📊 Enhanced Backend geladen: {len(projects_db)} Projekte, {len(documents_db)} Dokumente ({processed_docs} verarbeitet), {len(chats_db)} Chats")
        
        # Initialize intelligence features for existing documents
        analyzer = IntelligentDocumentAnalyzer(documents_db)
        enhanced_count = 0
        
        for doc_id, doc in documents_db.items():
            if doc.get("processing_status") == "completed" and not doc.get("intelligence_features", {}).get("keywords_extracted"):
                text = doc.get("extracted_text", "")
                if text:
                    keywords = analyzer.extract_keywords_from_text(text)
                    doc["intelligence_features"] = {
                        "keywords_extracted": True,
                        "content_analyzed": True,
                        "search_ready": True,
                        "keywords": keywords,
                        "content_preview": text[:500] + "..." if len(text) > 500 else text
                    }
                    enhanced_count += 1
        
        if enhanced_count > 0:
            print(f"🧠 {enhanced_count} Dokumente mit Intelligence Features erweitert")
            
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
        
        print("💾 Enhanced Backend Daten gespeichert")
    except Exception as e:
        print(f"⚠️ Fehler beim Speichern der Daten: {e}")

# === Main Entry Point ===

if __name__ == "__main__":
    print("🚀 RagFlow Enhanced Backend wird gestartet...")
    print("=" * 80)
    print(f"📍 API-URL: http://localhost:8000")
    print(f"🏥 Health Check: http://localhost:8000/api/health")
    print(f"🤖 Enhanced Chat Test: http://localhost:8000/api/v1/chat/test")
    print(f"📚 API Docs: http://localhost:8000/docs")
    print("=" * 80)
    print()
    print("🧠 ENHANCED INTELLIGENCE FEATURES:")
    print("   ✅ Proaktive Dokumentenanalyse")
    print("   ✅ Automatische Keyword-Extraktion")
    print("   ✅ Intelligente Fuzzy-Suche")
    print("   ✅ Intent-Erkennung")
    print("   ✅ Smarte Fallback-Antworten")
    print("   ✅ Erweiterte RAG-Integration")
    print()
    print("🎯 PROBLEM GELÖST:")
    print("   ❌ Vorher: 'Bitte geben Sie den vollständigen Dateinamen an'")
    print("   ✅ Jetzt: Automatische Suche und intelligente Antworten!")
    print()
    
    # Check environment
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key or api_key == "your_google_api_key_here":
        print("⚠️  WARNUNG: GOOGLE_API_KEY nicht konfiguriert!")
        print("   📝 Zur .env Datei hinzufügen: GOOGLE_API_KEY=dein_echter_schlüssel")
        print("   🔑 Schlüssel erhalten unter: https://ai.google.dev")
    else:
        print(f"✅ Google API Key konfiguriert (Länge: {len(api_key)})")
    
    print()
    print("🚀 Enhanced Server startet auf http://localhost:8000")
    print("   📄 Lade Dokumente hoch und stelle intelligente Fragen!")
    print("   🧠 Die AI analysiert automatisch alle verfügbaren Inhalte!")
    print()
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )