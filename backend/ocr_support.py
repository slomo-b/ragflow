#!/usr/bin/env python3
"""
OCR Support für gescannte PDFs und Bilder
Unterstützt Tesseract OCR und verschiedene Bildformate
"""

import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import asyncio
from datetime import datetime

# OCR und Bildverarbeitung
try:
    import pytesseract
    from PIL import Image, ImageEnhance, ImageFilter
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("⚠️ OCR nicht verfügbar. Installiere: pip install pytesseract pillow")

try:
    import pdf2image
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    print("⚠️ pdf2image nicht verfügbar. Installiere: pip install pdf2image")

try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("⚠️ OpenCV nicht verfügbar. Installiere: pip install opencv-python")

class OCRProcessor:
    """Verarbeitet gescannte Dokumente mit OCR"""
    
    def __init__(self, tesseract_path: str = None):
        """
        Initialisiert OCR Processor
        
        Args:
            tesseract_path: Pfad zur Tesseract Installation (falls nicht in PATH)
        """
        self.tesseract_available = TESSERACT_AVAILABLE
        self.pdf2image_available = PDF2IMAGE_AVAILABLE
        self.opencv_available = OPENCV_AVAILABLE
        
        # Tesseract konfigurieren
        if tesseract_path and TESSERACT_AVAILABLE:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # Unterstützte Sprachen für OCR
        self.ocr_languages = {
            'de': 'deu',      # Deutsch
            'en': 'eng',      # Englisch  
            'fr': 'fra',      # Französisch
            'it': 'ita',      # Italienisch
            'es': 'spa',      # Spanisch
            'pt': 'por',      # Portugiesisch
            'nl': 'nld',      # Niederländisch
            'ru': 'rus',      # Russisch
        }
        
        # Prüfe verfügbare Tesseract-Sprachen
        self.available_languages = self._check_available_languages()
    
    def _check_available_languages(self) -> List[str]:
        """Prüft welche Sprachen in Tesseract verfügbar sind"""
        if not self.tesseract_available:
            return []
        
        try:
            langs = pytesseract.get_languages(config='')
            print(f"✅ Verfügbare OCR-Sprachen: {', '.join(langs)}")
            return langs
        except Exception as e:
            print(f"⚠️ Fehler beim Prüfen der OCR-Sprachen: {e}")
            return ['eng']  # Fallback auf Englisch
    
    def is_scanned_pdf(self, pdf_path: str) -> bool:
        """
        Prüft ob ein PDF gescannt ist (wenig/kein extrahierbarer Text)
        
        Args:
            pdf_path: Pfad zur PDF-Datei
            
        Returns:
            True wenn PDF wahrscheinlich gescannt ist
        """
        try:
            # Versuche Text zu extrahieren
            if TESSERACT_AVAILABLE:
                import PyPDF2
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    total_text = ""
                    
                    # Prüfe erste 3 Seiten
                    for i, page in enumerate(reader.pages[:3]):
                        page_text = page.extract_text()
                        total_text += page_text
                    
                    # Wenn weniger als 50 Zeichen pro Seite -> wahrscheinlich gescannt
                    avg_chars_per_page = len(total_text) / min(len(reader.pages), 3)
                    is_scanned = avg_chars_per_page < 50
                    
                    print(f"📊 PDF-Analyse: {avg_chars_per_page:.1f} Zeichen/Seite -> {'Gescannt' if is_scanned else 'Text-PDF'}")
                    return is_scanned
            
        except Exception as e:
            print(f"⚠️ Fehler bei PDF-Analyse: {e}")
            return False  # Im Zweifel kein OCR
        
        return False
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Verbessert Bild für bessere OCR-Ergebnisse
        
        Args:
            image: PIL Image
            
        Returns:
            Verbessertes Image
        """
        try:
            # Konvertiere zu Graustufen
            if image.mode != 'L':
                image = image.convert('L')
            
            # Erhöhe Kontrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
            
            # Erhöhe Schärfe
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.5)
            
            # Rauschreduzierung (falls OpenCV verfügbar)
            if self.opencv_available:
                # Konvertiere zu OpenCV Format
                cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_GRAY2BGR)
                
                # Rauschreduzierung
                cv_image = cv2.fastNlMeansDenoisingColored(cv_image, None, 10, 10, 7, 21)
                
                # Zurück zu PIL
                image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY))
            
            # Schwellenwert-Filter für bessere Texterkennung
            # image = image.point(lambda x: 0 if x < 128 else 255, '1')
            
            return image
            
        except Exception as e:
            print(f"⚠️ Bildverbesserung fehlgeschlagen: {e}")
            return image
    
    async def extract_text_from_image(self, image_path: str, language: str = 'eng') -> Dict[str, Any]:
        """
        Extrahiert Text aus einem Bild mit OCR
        
        Args:
            image_path: Pfad zum Bild
            language: Sprache für OCR
            
        Returns:
            Dict mit extrahiertem Text und Metadaten
        """
        if not self.tesseract_available:
            return {
                "success": False,
                "text": "",
                "error": "Tesseract OCR nicht verfügbar",
                "confidence": 0.0
            }
        
        try:
            print(f"🔍 OCR für Bild: {Path(image_path).name}")
            
            # Lade und verbessere Bild
            image = Image.open(image_path)
            original_size = image.size
            
            # Bildverbesserung
            processed_image = self.preprocess_image(image)
            
            # OCR Konfiguration
            ocr_config = r'--oem 3 --psm 6'  # OCR Engine Mode 3, Page Segmentation Mode 6
            
            # Bestimme Sprache
            tesseract_lang = self.ocr_languages.get(language, 'eng')
            if tesseract_lang not in self.available_languages:
                tesseract_lang = 'eng'  # Fallback
            
            # Führe OCR durch
            extracted_text = pytesseract.image_to_string(
                processed_image, 
                lang=tesseract_lang, 
                config=ocr_config
            )
            
            # Hole Confidence-Daten
            try:
                data = pytesseract.image_to_data(
                    processed_image, 
                    lang=tesseract_lang, 
                    config=ocr_config,
                    output_type=pytesseract.Output.DICT
                )
                
                # Berechne durchschnittliche Confidence
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                
            except Exception:
                avg_confidence = 50  # Fallback-Wert
            
            # Bereinige Text
            cleaned_text = self._clean_ocr_text(extracted_text)
            
            result = {
                "success": True,
                "text": cleaned_text,
                "raw_text": extracted_text,
                "confidence": avg_confidence,
                "language": tesseract_lang,
                "image_size": original_size,
                "char_count": len(cleaned_text),
                "word_count": len(cleaned_text.split()),
                "processing_info": {
                    "preprocessed": True,
                    "ocr_config": ocr_config,
                    "tesseract_version": pytesseract.get_tesseract_version()
                }
            }
            
            print(f"✅ OCR erfolgreich: {len(cleaned_text)} Zeichen, {avg_confidence:.1f}% Confidence")
            return result
            
        except Exception as e:
            print(f"❌ OCR fehlgeschlagen: {e}")
            return {
                "success": False,
                "text": "",
                "error": str(e),
                "confidence": 0.0
            }
    
    async def extract_text_from_scanned_pdf(self, pdf_path: str, language: str = 'eng') -> Dict[str, Any]:
        """
        Extrahiert Text aus gescannter PDF mit OCR
        
        Args:
            pdf_path: Pfad zur PDF-Datei
            language: Sprache für OCR
            
        Returns:
            Dict mit extrahiertem Text und Metadaten
        """
        if not self.pdf2image_available or not self.tesseract_available:
            return {
                "success": False,
                "text": "",
                "error": "PDF2Image oder Tesseract nicht verfügbar",
                "pages_processed": 0
            }
        
        try:
            print(f"📄 OCR für gescannte PDF: {Path(pdf_path).name}")
            
            # Konvertiere PDF zu Bildern
            pages = pdf2image.convert_from_path(pdf_path, dpi=300)  # Hohe DPI für bessere Qualität
            print(f"📊 {len(pages)} Seiten zur OCR-Verarbeitung")
            
            all_text = []
            page_results = []
            total_confidence = 0
            
            for i, page_image in enumerate(pages):
                print(f"🔍 Verarbeite Seite {i+1}/{len(pages)}")
                
                # Speichere Seite als temporäres Bild
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                    page_image.save(temp_file.name, 'PNG')
                    temp_path = temp_file.name
                
                try:
                    # OCR für diese Seite
                    page_result = await self.extract_text_from_image(temp_path, language)
                    
                    if page_result["success"]:
                        page_text = page_result["text"]
                        page_confidence = page_result["confidence"]
                        
                        all_text.append(f"[Seite {i+1}]\n{page_text}")
                        total_confidence += page_confidence
                        
                        page_results.append({
                            "page_number": i+1,
                            "text": page_text,
                            "confidence": page_confidence,
                            "char_count": len(page_text),
                            "word_count": len(page_text.split())
                        })
                    else:
                        print(f"⚠️ OCR fehlgeschlagen für Seite {i+1}: {page_result.get('error', 'Unbekannter Fehler')}")
                        page_results.append({
                            "page_number": i+1,
                            "text": "",
                            "confidence": 0,
                            "error": page_result.get('error', 'OCR fehlgeschlagen')
                        })
                
                finally:
                    # Lösche temporäre Datei
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
            
            # Kombiniere alle Seiten
            full_text = "\n\n".join(all_text)
            avg_confidence = total_confidence / len(pages) if pages else 0
            
            result = {
                "success": True,
                "text": full_text,
                "pages_processed": len(pages),
                "successful_pages": len([p for p in page_results if p.get("confidence", 0) > 0]),
                "average_confidence": avg_confidence,
                "total_chars": len(full_text),
                "total_words": len(full_text.split()),
                "page_results": page_results,
                "language": language,
                "processing_info": {
                    "pdf_path": pdf_path,
                    "dpi": 300,
                    "ocr_method": "pdf2image + tesseract"
                }
            }
            
            print(f"✅ PDF OCR abgeschlossen: {len(pages)} Seiten, {avg_confidence:.1f}% Confidence")
            return result
            
        except Exception as e:
            print(f"❌ PDF OCR fehlgeschlagen: {e}")
            return {
                "success": False,
                "text": "",
                "error": str(e),
                "pages_processed": 0
            }
    
    def _clean_ocr_text(self, text: str) -> str:
        """Bereinigt OCR-Text von typischen Fehlern"""
        if not text:
            return ""
        
        # Entferne übermäßige Leerzeichen
        import re
        
        # Mehrfache Leerzeichen durch einzelne ersetzen
        text = re.sub(r' +', ' ', text)
        
        # Mehrfache Zeilenumbrüche reduzieren
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Entferne Leerzeichen am Zeilenanfang/-ende
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        # Typische OCR-Fehler korrigieren
        ocr_corrections = {
            r'\|': 'I',  # Pipe zu I
            r'0(?=[a-zA-Z])': 'O',  # 0 vor Buchstaben zu O
            r'(?<=[a-zA-Z])0': 'o',  # 0 nach Buchstaben zu o
            r'5(?=[a-zA-Z])': 'S',  # 5 vor Buchstaben zu S
            r'1(?=[a-zA-Z])': 'l',  # 1 vor Buchstaben zu l
        }
        
        for pattern, replacement in ocr_corrections.items():
            text = re.sub(pattern, replacement, text)
        
        return text.strip()

class EnhancedDocumentProcessor:
    """Erweiterte Dokumentenverarbeitung mit allen Features"""
    
    def __init__(self):
        self.ocr_processor = OCRProcessor()
        
        # Importiere andere Prozessoren
        try:
            from multilingual_support import MultilingualProcessor
            self.multilingual_processor = MultilingualProcessor()
        except ImportError:
            self.multilingual_processor = None
            print("⚠️ Multilingual Support nicht verfügbar")
        
        try:
            from vector_embeddings import VectorEmbeddingManager
            self.vector_manager = VectorEmbeddingManager()
        except ImportError:
            self.vector_manager = None
            print("⚠️ Vector Embeddings nicht verfügbar")
    
    async def process_document_complete(self, file_path: str, file_type: str, document_id: str, 
                                      document_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Vollständige Dokumentenverarbeitung mit allen Features
        
        Args:
            file_path: Pfad zur Datei
            file_type: Dateityp
            document_id: Dokument-ID
            document_metadata: Metadaten
            
        Returns:
            Vollständiges Verarbeitungsergebnis
        """
        print(f"🚀 Vollständige Verarbeitung: {document_metadata.get('filename')}")
        
        result = {
            "success": False,
            "text": "",
            "chunks": [],
            "metadata": {},
            "features_used": [],
            "processing_time": 0
        }
        
        start_time = datetime.utcnow()
        
        try:
            # 1. Grundlegende Textextraktion
            if file_type == "pdf":
                # Prüfe ob PDF gescannt ist
                if self.ocr_processor.is_scanned_pdf(file_path):
                    print("📄 Gescannte PDF erkannt - verwende OCR")
                    
                    # Erkenne Sprache wenn möglich (aus Metadaten oder Dateiname)
                    ocr_language = self._detect_document_language(document_metadata)
                    
                    # OCR-Verarbeitung
                    ocr_result = await self.ocr_processor.extract_text_from_scanned_pdf(file_path, ocr_language)
                    
                    if ocr_result["success"]:
                        text = ocr_result["text"]
                        result["ocr_info"] = ocr_result
                        result["features_used"].append("ocr")
                        print(f"✅ OCR erfolgreich: {len(text)} Zeichen")
                    else:
                        print(f"❌ OCR fehlgeschlagen: {ocr_result.get('error')}")
                        return result
                else:
                    # Normale PDF-Verarbeitung
                    from document_processor import DocumentProcessor
                    processor = DocumentProcessor()
                    basic_result = await processor.process_document(file_path, file_type)
                    
                    if basic_result["success"]:
                        text = basic_result["text"]
                        result["features_used"].append("pdf_extraction")
                    else:
                        return result
            else:
                # Andere Dateitypen (DOCX, TXT, MD)
                from document_processor import DocumentProcessor
                processor = DocumentProcessor()
                basic_result = await processor.process_document(file_path, file_type)
                
                if basic_result["success"]:
                    text = basic_result["text"]
                    result["features_used"].append(f"{file_type}_extraction")
                else:
                    return result
            
            # 2. Spracherkennung
            language_info = None
            if self.multilingual_processor and text:
                language_info = self.multilingual_processor.detect_language(text)
                result["language_info"] = language_info
                result["features_used"].append("language_detection")
                print(f"🌍 Sprache erkannt: {language_info['language_name']}")
            
            # 3. Erweiterte Chunk-Strategien
            from multilingual_support import AdvancedChunkingStrategy
            chunker = AdvancedChunkingStrategy()
            
            # Wähle Chunking-Strategie
            strategy = self._choose_chunking_strategy(file_type, text, language_info)
            chunks = chunker.chunk_text_advanced(text, strategy=strategy)
            
            result["chunking_strategy"] = strategy
            result["features_used"].append("advanced_chunking")
            print(f"📄 {len(chunks)} Chunks erstellt (Strategie: {strategy})")
            
            # 4. Mehrsprachige Verarbeitung
            if self.multilingual_processor and language_info and language_info.get("supported", False):
                enhanced_chunks = []
                for chunk in chunks:
                    enhanced_chunk = self.multilingual_processor.process_multilingual_chunk(
                        chunk, language_info["language"]
                    )
                    enhanced_chunks.append(enhanced_chunk)
                chunks = enhanced_chunks
                result["features_used"].append("multilingual_processing")
                print("🌍 Mehrsprachige Chunk-Verarbeitung abgeschlossen")
            
            # 5. Vector Embeddings
            if self.vector_manager and self.vector_manager.model:
                await self.vector_manager.process_document_chunks(document_id, chunks, document_metadata)
                result["features_used"].append("vector_embeddings")
                print("🔍 Vector Embeddings erstellt")
            
            # 6. Erweiterte Metadaten
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            enhanced_metadata = {
                **document_metadata,
                "text_length": len(text),
                "word_count": len(text.split()),
                "chunk_count": len(chunks),
                "language_info": language_info,
                "chunking_strategy": strategy,
                "features_used": result["features_used"],
                "processing_time_seconds": processing_time,
                "enhanced_processing": True,
                "processed_at": datetime.utcnow().isoformat()
            }
            
            # 7. Erfolgreiche Rückgabe
            result.update({
                "success": True,
                "text": text,
                "chunks": chunks,
                "metadata": enhanced_metadata,
                "processing_time": processing_time
            })
            
            print(f"✅ Vollständige Verarbeitung abgeschlossen ({processing_time:.2f}s)")
            return result
            
        except Exception as e:
            print(f"❌ Vollständige Verarbeitung fehlgeschlagen: {e}")
            result["error"] = str(e)
            return result
    
    def _detect_document_language(self, document_metadata: Dict[str, Any]) -> str:
        """Versucht Dokumentsprache zu erkennen"""
        # Aus Dateiname
        filename = document_metadata.get("filename", "").lower()
        
        language_indicators = {
            "de": ["deutsch", "german", "de_", "_de", "ger"],
            "en": ["english", "en_", "_en", "eng"],
            "fr": ["french", "francais", "fr_", "_fr", "fra"],
            "it": ["italian", "italiano", "it_", "_it", "ita"],
            "es": ["spanish", "espanol", "es_", "_es", "spa"]
        }
        
        for lang_code, indicators in language_indicators.items():
            if any(indicator in filename for indicator in indicators):
                return lang_code
        
        return "eng"  # Fallback auf Englisch
    
    def _choose_chunking_strategy(self, file_type: str, text: str, language_info: Dict = None) -> str:
        """Wählt optimale Chunking-Strategie"""
        text_length = len(text)
        
        # Für sehr kurze Texte
        if text_length < 1000:
            return "paragraph"
        
        # Für gescannte Dokumente (OCR-Text ist oft weniger strukturiert)
        if hasattr(self, 'ocr_info'):
            return "sliding_window"
        
        # Für strukturierte Texte
        if file_type in ["md", "txt"] or (text.count('\n\n') / max(text.count('\n'), 1) > 0.3):
            return "semantic"
        
        # Für PDFs
        if file_type == "pdf":
            return "hybrid"
        
        return "paragraph"

# Installation Helper
def check_and_install_dependencies():
    """Prüft und gibt Installationsanweisungen für fehlende Pakete"""
    
    missing_packages = []
    
    # Basis OCR
    if not TESSERACT_AVAILABLE:
        missing_packages.extend(["pytesseract", "pillow"])
    
    if not PDF2IMAGE_AVAILABLE:
        missing_packages.append("pdf2image")
    
    if not OPENCV_AVAILABLE:
        missing_packages.append("opencv-python")
    
    # Vector Embeddings
    try:
        import sentence_transformers
        import faiss
    except ImportError:
        missing_packages.extend(["sentence-transformers", "faiss-cpu"])
    
    # Mehrsprachig
    try:
        import langdetect
        import googletrans
    except ImportError:
        missing_packages.extend(["langdetect", "googletrans==4.0.0rc1"])
    
    if missing_packages:
        print("📦 Fehlende Pakete für erweiterte Features:")
        print(f"pip install {' '.join(set(missing_packages))}")
        print()
        print("🔧 Zusätzlich benötigt:")
        print("- Tesseract OCR: https://github.com/tesseract-ocr/tesseract")
        print("- Poppler (für pdf2image): https://poppler.freedesktop.org/")
        return False
    else:
        print("✅ Alle Pakete für erweiterte Features verfügbar!")
        return True

if __name__ == "__main__":
    # Test aller erweiterten Features
    async def test_enhanced_features():
        print("🧪 Test der erweiterten Features")
        print("=" * 50)
        
        # Prüfe Dependencies
        if not check_and_install_dependencies():
            print("❌ Nicht alle Pakete verfügbar - installiere fehlende Pakete")
            return
        
        # Teste OCR
        ocr_processor = OCRProcessor()
        print(f"OCR verfügbar: {ocr_processor.tesseract_available}")
        print(f"PDF2Image verfügbar: {ocr_processor.pdf2image_available}")
        
        # Teste Enhanced Processor
        enhanced_processor = EnhancedDocumentProcessor()
        
        # Statistiken
        print("\n📊 Feature-Verfügbarkeit:")
        print(f"   OCR: {'✅' if enhanced_processor.ocr_processor.tesseract_available else '❌'}")
        print(f"   Mehrsprachig: {'✅' if enhanced_processor.multilingual_processor else '❌'}")
        print(f"   Vector Embeddings: {'✅' if enhanced_processor.vector_manager and enhanced_processor.vector_manager.model else '❌'}")
        
        print("\n🎯 Erweiterte Features bereit für Integration!")
    
    asyncio.run(test_enhanced_features())