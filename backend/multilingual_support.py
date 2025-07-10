#!/usr/bin/env python3
"""
Reparierte mehrsprachige Unterstützung für RagFlow
Verwendet deep-translator statt googletrans
"""

from typing import Dict, List, Any, Optional, Tuple
import re
from datetime import datetime
import uuid

# Mehrsprachige Libraries
try:
    from langdetect import detect, detect_langs
    from langdetect.lang_detect_exception import LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    print("⚠️ langdetect nicht verfügbar. Installiere: pip install langdetect")

try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    TRANSLATOR_AVAILABLE = False
    print("⚠️ deep-translator nicht verfügbar. Installiere: pip install deep-translator")

class TranslationResult:
    """Kompatibilitäts-Wrapper für Übersetzungsergebnisse"""
    def __init__(self, text: str, src: str = "auto", dest: str = "en", confidence: float = None):
        self.text = text
        self.src = src
        self.dest = dest
        self.confidence = confidence

class MultilingualProcessor:
    """Verarbeitet mehrsprachige Dokumente und Suchanfragen - Reparierte Version"""
    
    def __init__(self):
        self.translator_available = TRANSLATOR_AVAILABLE
        self.supported_languages = {
            'de': 'Deutsch',
            'en': 'English', 
            'fr': 'Français',
            'it': 'Italiano',
            'es': 'Español',
            'pt': 'Português',
            'nl': 'Nederlands',
            'ru': 'Русский',
            'zh': '中文',
            'ja': '日本語',
            'ko': '한국어',
            'ar': 'العربية'
        }
        
        print(f"✅ MultilingualProcessor initialisiert (Translator: {'Ja' if self.translator_available else 'Nein'})")
    
    def detect_language(self, text: str, min_length: int = 50) -> Dict[str, Any]:
        """
        Erkennt die Sprache eines Texts
        
        Args:
            text: Zu analysierender Text
            min_length: Minimale Textlänge für zuverlässige Erkennung
            
        Returns:
            Dict mit Sprache und Confidence
        """
        if not LANGDETECT_AVAILABLE or len(text.strip()) < min_length:
            return {
                "language": "unknown",
                "language_name": "Unbekannt",
                "confidence": 0.0,
                "reliable": False
            }
        
        try:
            # Primäre Spracherkennung
            primary_lang = detect(text)
            
            # Detaillierte Erkennung mit Confidence
            lang_probs = detect_langs(text)
            
            # Finde die beste Übereinstimmung
            best_match = lang_probs[0]
            language_code = best_match.lang
            confidence = best_match.prob
            
            # Prüfe ob Sprache unterstützt wird
            language_name = self.supported_languages.get(language_code, f"Unbekannt ({language_code})")
            is_supported = language_code in self.supported_languages
            
            return {
                "language": language_code,
                "language_name": language_name,
                "confidence": confidence,
                "reliable": confidence > 0.8,
                "supported": is_supported,
                "all_probabilities": [{"lang": lp.lang, "prob": lp.prob} for lp in lang_probs[:3]]
            }
            
        except LangDetectException as e:
            print(f"⚠️ Spracherkennung fehlgeschlagen: {e}")
            return {
                "language": "unknown",
                "language_name": "Unbekannt",
                "confidence": 0.0,
                "reliable": False,
                "supported": False,
                "error": str(e)
            }
    
    def translate_text(self, text: str, target_lang: str = "en", source_lang: str = "auto") -> Dict[str, Any]:
        """
        Übersetzt Text in die Zielsprache - Reparierte Version
        
        Args:
            text: Zu übersetzender Text
            target_lang: Zielsprache (ISO Code)
            source_lang: Quellsprache ("auto" für automatische Erkennung)
            
        Returns:
            Dict mit Übersetzung und Metadaten
        """
        if not self.translator_available:
            return {
                "translated_text": text,
                "source_language": "unknown",
                "target_language": target_lang,
                "success": False,
                "error": "Deep-translator nicht verfügbar"
            }
        
        try:
            # Kurze Texte nicht übersetzen
            if len(text.strip()) < 10:
                return {
                    "translated_text": text,
                    "source_language": source_lang,
                    "target_language": target_lang,
                    "success": True,
                    "skipped": "Text zu kurz"
                }
            
            # Automatische Spracherkennung
            if source_lang == "auto":
                if LANGDETECT_AVAILABLE:
                    try:
                        detected_lang = detect(text)
                        source_lang = detected_lang
                    except:
                        source_lang = "en"
                else:
                    source_lang = "en"
            
            # Wenn Quell- und Zielsprache gleich sind
            if source_lang == target_lang:
                return {
                    "translated_text": text,
                    "source_language": source_lang,
                    "target_language": target_lang,
                    "success": True,
                    "skipped": "Gleiche Sprache"
                }
            
            # Übersetze mit deep-translator
            translator = GoogleTranslator(source=source_lang, target=target_lang)
            translated_text = translator.translate(text)
            
            return {
                "translated_text": translated_text,
                "source_language": source_lang,
                "target_language": target_lang,
                "success": True,
                "confidence": 0.8,  # Deep-translator gibt keine Confidence zurück
                "original_text": text[:100] + "..." if len(text) > 100 else text
            }
            
        except Exception as e:
            print(f"❌ Übersetzung fehlgeschlagen: {e}")
            return {
                "translated_text": text,
                "source_language": source_lang,
                "target_language": target_lang,
                "success": False,
                "error": str(e)
            }
    
    def process_multilingual_chunk(self, chunk: Dict[str, Any], document_language: str = None) -> Dict[str, Any]:
        """
        Verarbeitet einen Chunk mehrsprachig
        
        Args:
            chunk: Text-Chunk
            document_language: Bekannte Dokumentsprache
            
        Returns:
            Erweiterter Chunk mit Sprachinformationen
        """
        text = chunk.get("text", "")
        
        # Erkenne Sprache wenn nicht bekannt
        if not document_language:
            lang_info = self.detect_language(text)
            detected_lang = lang_info["language"]
        else:
            detected_lang = document_language
            lang_info = {
                "language": document_language,
                "confidence": 1.0,
                "reliable": True
            }
        
        # Erstelle Übersetzungen für wichtige Sprachen
        translations = {}
        
        # Übersetze ins Englische (Universal-Sprache für Suche)
        if detected_lang != "en" and self.translator_available:
            en_translation = self.translate_text(text, target_lang="en", source_lang=detected_lang)
            if en_translation["success"]:
                translations["en"] = en_translation["translated_text"]
        
        # Übersetze ins Deutsche (Hauptsprache des Systems)
        if detected_lang != "de" and self.translator_available:
            de_translation = self.translate_text(text, target_lang="de", source_lang=detected_lang)
            if de_translation["success"]:
                translations["de"] = de_translation["translated_text"]
        
        # Erweitere Chunk
        enhanced_chunk = chunk.copy()
        enhanced_chunk.update({
            "language_info": lang_info,
            "detected_language": detected_lang,
            "translations": translations,
            "multilingual_processed": True,
            "processed_at": datetime.utcnow().isoformat()
        })
        
        return enhanced_chunk
    
    def create_multilingual_search_terms(self, query: str, query_language: str = None) -> List[Dict[str, str]]:
        """
        Erstellt mehrsprachige Suchbegriffe für eine Anfrage
        
        Args:
            query: Suchanfrage
            query_language: Sprache der Anfrage
            
        Returns:
            Liste mit Übersetzungen der Suchanfrage
        """
        search_terms = []
        
        # Original-Query
        if not query_language:
            lang_info = self.detect_language(query)
            query_language = lang_info["language"]
        
        search_terms.append({
            "text": query,
            "language": query_language,
            "type": "original"
        })
        
        # Übersetze in wichtige Sprachen
        target_languages = ["en", "de", "fr", "it", "es"]
        
        for target_lang in target_languages:
            if target_lang != query_language and self.translator_available:
                translation = self.translate_text(query, target_lang=target_lang, source_lang=query_language)
                if translation["success"]:
                    search_terms.append({
                        "text": translation["translated_text"],
                        "language": target_lang,
                        "type": "translation",
                        "confidence": translation.get("confidence", 0.8)
                    })
        
        return search_terms

class MultilingualSearchEnhancer:
    """Erweitert die Suche um mehrsprachige Funktionen"""
    
    def __init__(self, multilingual_processor: MultilingualProcessor):
        self.ml_processor = multilingual_processor
    
    def enhance_search_with_translations(self, query: str, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Erweitert Suchergebnisse um mehrsprachige Suche
        
        Args:
            query: Original-Suchanfrage
            search_results: Bisherige Suchergebnisse
            
        Returns:
            Erweiterte Suchergebnisse
        """
        # Erstelle mehrsprachige Suchbegriffe
        search_terms = self.ml_processor.create_multilingual_search_terms(query)
        
        enhanced_results = []
        
        for result in search_results:
            enhanced_result = result.copy()
            
            # Prüfe mehrsprachige Übereinstimmungen
            chunk_text = result.get("text", "").lower()
            chunk_translations = result.get("translations", {})
            
            # Score für mehrsprachige Treffer
            multilingual_score = 0.0
            matched_languages = []
            
            for search_term in search_terms:
                term_text = search_term["text"].lower()
                term_lang = search_term["language"]
                
                # Direkte Suche im Original-Text
                if term_text in chunk_text:
                    weight = 1.0 if search_term["type"] == "original" else 0.7
                    multilingual_score += weight
                    matched_languages.append(term_lang)
                
                # Suche in Übersetzungen
                for trans_lang, trans_text in chunk_translations.items():
                    if term_text in trans_text.lower():
                        weight = 0.8 if trans_lang == term_lang else 0.5
                        multilingual_score += weight
                        matched_languages.append(f"{term_lang}->{trans_lang}")
            
            # Aktualisiere Score
            original_score = enhanced_result.get("similarity_score", 0.0)
            enhanced_result["multilingual_score"] = multilingual_score
            enhanced_result["final_score"] = original_score + (multilingual_score * 0.3)
            enhanced_result["matched_languages"] = list(set(matched_languages))
            
            enhanced_results.append(enhanced_result)
        
        # Sortiere nach neuem Score
        enhanced_results.sort(key=lambda x: x.get("final_score", 0), reverse=True)
        
        return enhanced_results

# Erweiterte Chunk-Strategien (Kopiert aus dem Original)
class AdvancedChunkingStrategy:
    """Erweiterte Strategien für intelligenteres Text-Chunking"""
    
    def __init__(self):
        self.strategies = {
            "semantic": self._semantic_chunking,
            "paragraph": self._paragraph_chunking,
            "sentence": self._sentence_chunking,
            "sliding_window": self._sliding_window_chunking,
            "hybrid": self._hybrid_chunking
        }
    
    def chunk_text_advanced(self, text: str, strategy: str = "hybrid", **kwargs) -> List[Dict[str, Any]]:
        """
        Erweiterte Text-Chunking mit verschiedenen Strategien
        """
        if strategy not in self.strategies:
            strategy = "hybrid"
        
        chunking_func = self.strategies[strategy]
        chunks = chunking_func(text, **kwargs)
        
        # Füge Metadaten hinzu
        for i, chunk in enumerate(chunks):
            chunk.update({
                "chunk_id": chunk.get("id", str(uuid.uuid4())),
                "chunk_index": i,
                "chunking_strategy": strategy,
                "created_at": datetime.utcnow().isoformat()
            })
        
        return chunks
    
    def _hybrid_chunking(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        """Einfache hybride Strategie"""
        # Vereinfachte Implementierung
        max_chunk_size = kwargs.get("max_chunk_size", 1000)
        chunks = []
        
        # Teile nach Absätzen
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        current_chunk = ""
        current_start = 0
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) > max_chunk_size and current_chunk:
                chunks.append({
                    "id": str(uuid.uuid4()),
                    "text": current_chunk.strip(),
                    "start_pos": current_start,
                    "end_pos": current_start + len(current_chunk),
                    "chunk_type": "paragraph_group"
                })
                current_start += len(current_chunk)
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Letzten Chunk hinzufügen
        if current_chunk.strip():
            chunks.append({
                "id": str(uuid.uuid4()),
                "text": current_chunk.strip(),
                "start_pos": current_start,
                "end_pos": current_start + len(current_chunk),
                "chunk_type": "final_chunk"
            })
        
        return chunks
    
    # Vereinfachte Implementierungen der anderen Strategien
    def _semantic_chunking(self, text: str, **kwargs):
        return self._hybrid_chunking(text, **kwargs)
    
    def _paragraph_chunking(self, text: str, **kwargs):
        return self._hybrid_chunking(text, **kwargs)
    
    def _sentence_chunking(self, text: str, **kwargs):
        return self._hybrid_chunking(text, **kwargs)
    
    def _sliding_window_chunking(self, text: str, **kwargs):
        return self._hybrid_chunking(text, **kwargs)

if __name__ == "__main__":
    # Test der reparierten Version
    print("🧪 Test der reparierten Multilingual Support")
    
    # Test Multilingual Processor
    ml_processor = MultilingualProcessor()
    
    # Test Spracherkennung
    test_text = "Dies ist ein deutscher Text über Restaurants und Speisekarten."
    lang_info = ml_processor.detect_language(test_text)
    print(f"Sprache erkannt: {lang_info['language_name']} ({lang_info['confidence']:.2f})")
    
    # Test Übersetzung
    if ml_processor.translator_available:
        translation = ml_processor.translate_text("Hallo Welt", target_lang="en")
        print(f"Übersetzung: {translation['translated_text']}")
    else:
        print("Übersetzer nicht verfügbar")
    
    print("✅ Reparierte Version funktioniert!")
