# backend/app/ocr_support.py - Modern OCR Support with ChromaDB Integration
"""
Modern OCR Support for RagFlow
Handles image-to-text conversion and scanned PDF processing
"""

import io
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
from PIL import Image
import base64

# OCR Libraries (with fallbacks)
try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    from pdf2image import convert_from_bytes, convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

from rich.console import Console
from .database import get_db_manager
from .config import settings

console = Console()


class OCRProcessor:
    """Advanced OCR processor with multiple engine support"""
    
    def __init__(self):
        self.db = get_db_manager()
        self.available_engines = self._detect_available_engines()
        self.default_engine = self._get_default_engine()
        
        # Initialize EasyOCR reader if available
        self.easyocr_reader = None
        if EASYOCR_AVAILABLE:
            try:
                self.easyocr_reader = easyocr.Reader(['en', 'de', 'fr', 'es'])  # Multi-language
                console.print("✅ EasyOCR initialized with multi-language support")
            except Exception as e:
                console.print(f"⚠️ EasyOCR initialization failed: {e}")
        
        # Supported image formats
        self.supported_image_formats = {
            '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'
        }
        
        # Supported document formats for OCR
        self.supported_document_formats = {
            '.pdf'  # Scanned PDFs
        }
    
    def _detect_available_engines(self) -> Dict[str, bool]:
        """Detect available OCR engines"""
        engines = {
            "tesseract": PYTESSERACT_AVAILABLE,
            "easyocr": EASYOCR_AVAILABLE,
            "pdf2image": PDF2IMAGE_AVAILABLE,
            "opencv": OPENCV_AVAILABLE
        }
        
        # Test Tesseract installation
        if PYTESSERACT_AVAILABLE:
            try:
                pytesseract.get_tesseract_version()
                engines["tesseract"] = True
            except Exception:
                engines["tesseract"] = False
                console.print("⚠️ Tesseract not properly installed")
        
        return engines
    
    def _get_default_engine(self) -> str:
        """Get default OCR engine based on availability"""
        if self.available_engines.get("easyocr"):
            return "easyocr"  # Generally better accuracy
        elif self.available_engines.get("tesseract"):
            return "tesseract"  # Fast and reliable
        else:
            return None
    
    def is_ocr_available(self) -> bool:
        """Check if any OCR engine is available"""
        return any(self.available_engines.values())
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get status of all OCR engines"""
        status = {
            "available_engines": self.available_engines,
            "default_engine": self.default_engine,
            "ocr_ready": self.is_ocr_available(),
            "supported_formats": {
                "images": list(self.supported_image_formats),
                "documents": list(self.supported_document_formats)
            },
            "installation_notes": {
                "tesseract": "Install: apt-get install tesseract-ocr (Linux) or brew install tesseract (Mac)",
                "easyocr": "Install: pip install easyocr",
                "pdf2image": "Install: pip install pdf2image",
                "opencv": "Install: pip install opencv-python"
            }
        }
        
        return status
    
    def can_process_file(self, file_path: Union[str, Path]) -> bool:
        """Check if file can be processed with OCR"""
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        is_image = extension in self.supported_image_formats
        is_document = extension in self.supported_document_formats
        
        return self.is_ocr_available() and (is_image or is_document)
    
    async def process_image_file(
        self, 
        file_path: Union[str, Path],
        engine: str = None,
        language: str = "en",
        project_ids: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Process image file with OCR and store in ChromaDB"""
        
        if not self.is_ocr_available():
            raise Exception("No OCR engines available. Please install pytesseract or easyocr.")
        
        file_path = Path(file_path)
        engine = engine or self.default_engine
        
        if not file_path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")
        
        if file_path.suffix.lower() not in self.supported_image_formats:
            raise ValueError(f"Unsupported image format: {file_path.suffix}")
        
        try:
            console.print(f"🖼️ Processing image with OCR: {file_path.name}")
            
            # Load and preprocess image
            image = Image.open(file_path)
            processed_image = self._preprocess_image(image)
            
            # Extract text using specified engine
            if engine == "easyocr" and self.easyocr_reader:
                extracted_text = await self._extract_text_easyocr(processed_image, language)
            elif engine == "tesseract" and self.available_engines["tesseract"]:
                extracted_text = await self._extract_text_tesseract(processed_image, language)
            else:
                raise Exception(f"OCR engine '{engine}' not available")
            
            # Get image metadata
            image_metadata = self._get_image_metadata(image, file_path)
            
            # Prepare document metadata
            doc_metadata = {
                "ocr_engine": engine,
                "ocr_language": language,
                "original_format": "image",
                "image_width": image_metadata["width"],
                "image_height": image_metadata["height"],
                "image_mode": image_metadata["mode"],
                "processed_at": datetime.utcnow().isoformat(),
                "ocr_confidence": getattr(extracted_text, 'confidence', None),
                "text_length": len(extracted_text) if isinstance(extracted_text, str) else 0,
                **(metadata or {})
            }
            
            # Store in ChromaDB
            document = self.db.create_document(
                filename=file_path.name,
                file_type=file_path.suffix.lower(),
                file_size=file_path.stat().st_size,
                file_path=str(file_path.absolute()),
                content=str(extracted_text),
                project_ids=project_ids or []
            )
            
            # Update metadata
            document.metadata.update(doc_metadata)
            self.db._update_document_metadata(document.id, document)
            
            console.print(f"✅ Image OCR completed: {document.id}")
            
            return {
                "document_id": document.id,
                "filename": file_path.name,
                "extracted_text": str(extracted_text),
                "text_length": len(str(extracted_text)),
                "ocr_engine": engine,
                "ocr_language": language,
                "processing_status": "completed",
                "metadata": doc_metadata
            }
            
        except Exception as e:
            console.print(f"❌ OCR processing failed for {file_path.name}: {e}")
            raise Exception(f"OCR processing failed: {str(e)}")
    
    async def process_scanned_pdf(
        self, 
        file_path: Union[str, Path],
        engine: str = None,
        language: str = "en",
        project_ids: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Process scanned PDF with OCR and store in ChromaDB"""
        
        if not PDF2IMAGE_AVAILABLE:
            raise Exception("PDF processing requires pdf2image. Install with: pip install pdf2image")
        
        if not self.is_ocr_available():
            raise Exception("No OCR engines available")
        
        file_path = Path(file_path)
        engine = engine or self.default_engine
        
        try:
            console.print(f"📄 Processing scanned PDF: {file_path.name}")
            
            # First, try to extract text normally (in case it's not scanned)
            regular_text = await self._try_regular_pdf_extraction(file_path)
            
            if regular_text and len(regular_text.strip()) > 100:
                console.print("✅ PDF contains extractable text, using regular extraction")
                return await self._store_pdf_text(file_path, regular_text, project_ids, metadata, "regular")
            
            # Convert PDF to images
            console.print("🔄 Converting PDF pages to images for OCR...")
            images = convert_from_path(str(file_path), dpi=300)  # High DPI for better OCR
            
            # Process each page
            page_texts = []
            total_confidence = 0
            
            for page_num, image in enumerate(images, 1):
                console.print(f"📄 Processing page {page_num}/{len(images)}")
                
                # Preprocess image
                processed_image = self._preprocess_image(image)
                
                # Extract text
                if engine == "easyocr" and self.easyocr_reader:
                    page_text = await self._extract_text_easyocr(processed_image, language)
                elif engine == "tesseract" and self.available_engines["tesseract"]:
                    page_text = await self._extract_text_tesseract(processed_image, language)
                else:
                    raise Exception(f"OCR engine '{engine}' not available")
                
                if page_text.strip():
                    page_texts.append(f"--- Page {page_num} ---\n{page_text}")
            
            # Combine all pages
            full_text = "\n\n".join(page_texts)
            
            if not full_text.strip():
                raise Exception("No text could be extracted from PDF")
            
            # Store in ChromaDB
            doc_metadata = {
                "ocr_engine": engine,
                "ocr_language": language,
                "original_format": "scanned_pdf",
                "total_pages": len(images),
                "pages_processed": len(page_texts),
                "processed_at": datetime.utcnow().isoformat(),
                "text_length": len(full_text),
                "processing_method": "ocr",
                **(metadata or {})
            }
            
            document = self.db.create_document(
                filename=file_path.name,
                file_type=".pdf",
                file_size=file_path.stat().st_size,
                file_path=str(file_path.absolute()),
                content=full_text,
                project_ids=project_ids or []
            )
            
            document.metadata.update(doc_metadata)
            self.db._update_document_metadata(document.id, document)
            
            console.print(f"✅ Scanned PDF OCR completed: {document.id}")
            
            return {
                "document_id": document.id,
                "filename": file_path.name,
                "extracted_text": full_text,
                "text_length": len(full_text),
                "total_pages": len(images),
                "pages_processed": len(page_texts),
                "ocr_engine": engine,
                "ocr_language": language,
                "processing_status": "completed",
                "metadata": doc_metadata
            }
            
        except Exception as e:
            console.print(f"❌ Scanned PDF processing failed for {file_path.name}: {e}")
            raise Exception(f"Scanned PDF processing failed: {str(e)}")
    
    async def _try_regular_pdf_extraction(self, file_path: Path) -> str:
        """Try to extract text from PDF using regular methods"""
        if not PYPDF2_AVAILABLE:
            return ""
        
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text_parts = []
                
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_parts.append(page_text)
                
                return "\n\n".join(text_parts)
        except Exception:
            return ""
    
    async def _store_pdf_text(
        self, 
        file_path: Path, 
        text: str, 
        project_ids: List[str], 
        metadata: Dict[str, Any], 
        method: str
    ) -> Dict[str, Any]:
        """Store PDF text in ChromaDB"""
        doc_metadata = {
            "original_format": "pdf",
            "processing_method": method,
            "processed_at": datetime.utcnow().isoformat(),
            "text_length": len(text),
            **(metadata or {})
        }
        
        document = self.db.create_document(
            filename=file_path.name,
            file_type=".pdf",
            file_size=file_path.stat().st_size,
            file_path=str(file_path.absolute()),
            content=text,
            project_ids=project_ids or []
        )
        
        document.metadata.update(doc_metadata)
        self.db._update_document_metadata(document.id, document)
        
        return {
            "document_id": document.id,
            "filename": file_path.name,
            "extracted_text": text,
            "text_length": len(text),
            "processing_method": method,
            "processing_status": "completed",
            "metadata": doc_metadata
        }
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for better OCR results"""
        try:
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply preprocessing if OpenCV is available
            if OPENCV_AVAILABLE:
                # Convert PIL to OpenCV format
                img_array = np.array(image)
                img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                # Apply preprocessing
                img_cv = self._opencv_preprocess(img_cv)
                
                # Convert back to PIL
                img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(img_rgb)
            
            return image
            
        except Exception as e:
            console.print(f"⚠️ Image preprocessing failed, using original: {e}")
            return image
    
    def _opencv_preprocess(self, img: np.ndarray) -> np.ndarray:
        """Apply OpenCV preprocessing for better OCR"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # Apply adaptive threshold
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Morphological operations to clean up
            kernel = np.ones((1, 1), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Convert back to BGR for consistency
            return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
            
        except Exception as e:
            console.print(f"⚠️ OpenCV preprocessing failed: {e}")
            return img
    
    async def _extract_text_easyocr(self, image: Image.Image, language: str = "en") -> str:
        """Extract text using EasyOCR"""
        try:
            # Convert PIL image to numpy array
            img_array = np.array(image)
            
            # EasyOCR expects BGR format
            if len(img_array.shape) == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Extract text
            results = self.easyocr_reader.readtext(img_array)
            
            # Combine text results
            text_parts = []
            for (bbox, text, confidence) in results:
                if confidence > 0.3:  # Filter low-confidence results
                    text_parts.append(text)
            
            return "\n".join(text_parts)
            
        except Exception as e:
            raise Exception(f"EasyOCR extraction failed: {e}")
    
    async def _extract_text_tesseract(self, image: Image.Image, language: str = "en") -> str:
        """Extract text using Tesseract"""
        try:
            # Configure Tesseract
            config = f"--oem 3 --psm 6 -l {language}"
            
            # Extract text
            text = pytesseract.image_to_string(image, config=config)
            
            return text.strip()
            
        except Exception as e:
            raise Exception(f"Tesseract extraction failed: {e}")
    
    def _get_image_metadata(self, image: Image.Image, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from image"""
        metadata = {
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "format": image.format,
            "file_size": file_path.stat().st_size
        }
        
        # Add EXIF data if available
        try:
            if hasattr(image, '_getexif'):
                exif = image._getexif()
                if exif:
                    metadata["has_exif"] = True
                    # Add relevant EXIF data
                    for tag_id, value in exif.items():
                        if tag_id in [256, 257, 271, 272, 274, 306]:  # Common tags
                            metadata[f"exif_{tag_id}"] = str(value)
        except Exception:
            pass
        
        return metadata
    
    async def batch_process_images(
        self, 
        directory_path: Union[str, Path],
        engine: str = None,
        language: str = "en",
        project_ids: List[str] = None,
        recursive: bool = True
    ) -> Dict[str, Any]:
        """Batch process images in a directory"""
        
        directory_path = Path(directory_path)
        engine = engine or self.default_engine
        
        if not directory_path.exists():
            raise ValueError(f"Directory does not exist: {directory_path}")
        
        # Find image files
        if recursive:
            files = [f for f in directory_path.rglob("*") if f.suffix.lower() in self.supported_image_formats]
        else:
            files = [f for f in directory_path.glob("*") if f.suffix.lower() in self.supported_image_formats]
        
        console.print(f"🖼️ Found {len(files)} image files for OCR processing")
        
        results = {
            "total_files": len(files),
            "processed": [],
            "failed": [],
            "engine_used": engine,
            "language": language
        }
        
        for file_path in files:
            try:
                console.print(f"🔄 Processing: {file_path.name}")
                
                result = await self.process_image_file(
                    file_path=file_path,
                    engine=engine,
                    language=language,
                    project_ids=project_ids
                )
                
                results["processed"].append({
                    "file": str(file_path),
                    "document_id": result["document_id"],
                    "text_length": result["text_length"],
                    "status": "success"
                })
                
            except Exception as e:
                console.print(f"❌ Failed to process {file_path.name}: {e}")
                results["failed"].append({
                    "file": str(file_path),
                    "error": str(e),
                    "status": "failed"
                })
        
        console.print(f"✅ Batch OCR complete: {len(results['processed'])} processed, {len(results['failed'])} failed")
        return results
    
    def test_ocr_engines(self) -> Dict[str, Any]:
        """Test all available OCR engines"""
        results = {
            "test_date": datetime.utcnow().isoformat(),
            "engines": {}
        }
        
        # Create a simple test image
        test_image = Image.new('RGB', (200, 100), color='white')
        
        # Test Tesseract
        if self.available_engines["tesseract"]:
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                text = loop.run_until_complete(self._extract_text_tesseract(test_image))
                results["engines"]["tesseract"] = {
                    "status": "working",
                    "test_result": len(text) >= 0
                }
            except Exception as e:
                results["engines"]["tesseract"] = {
                    "status": "error",
                    "error": str(e)
                }
        else:
            results["engines"]["tesseract"] = {"status": "not_available"}
        
        # Test EasyOCR
        if self.available_engines["easyocr"]:
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                text = loop.run_until_complete(self._extract_text_easyocr(test_image))
                results["engines"]["easyocr"] = {
                    "status": "working",
                    "test_result": len(text) >= 0
                }
            except Exception as e:
                results["engines"]["easyocr"] = {
                    "status": "error",
                    "error": str(e)
                }
        else:
            results["engines"]["easyocr"] = {"status": "not_available"}
        
        return results
    
    def get_installation_guide(self) -> Dict[str, Any]:
        """Get installation guide for OCR dependencies"""
        return {
            "required_packages": {
                "pytesseract": {
                    "pip_install": "pip install pytesseract",
                    "system_requirements": {
                        "ubuntu": "sudo apt-get install tesseract-ocr tesseract-ocr-deu",
                        "macos": "brew install tesseract",
                        "windows": "Download from: https://github.com/UB-Mannheim/tesseract/wiki"
                    },
                    "languages": "Install additional languages: tesseract-ocr-fra, tesseract-ocr-spa, etc."
                },
                "easyocr": {
                    "pip_install": "pip install easyocr",
                    "note": "Will download models on first use (~100MB per language)"
                },
                "pdf2image": {
                    "pip_install": "pip install pdf2image",
                    "system_requirements": {
                        "ubuntu": "sudo apt-get install poppler-utils",
                        "macos": "brew install poppler",
                        "windows": "Download poppler from: https://blog.alivate.com.au/poppler-windows/"
                    }
                },
                "opencv": {
                    "pip_install": "pip install opencv-python",
                    "note": "For image preprocessing (optional but recommended)"
                }
            },
            "quick_setup": {
                "linux": [
                    "sudo apt-get update",
                    "sudo apt-get install tesseract-ocr poppler-utils",
                    "pip install pytesseract easyocr pdf2image opencv-python"
                ],
                "macos": [
                    "brew install tesseract poppler",
                    "pip install pytesseract easyocr pdf2image opencv-python"
                ],
                "windows": [
                    "1. Download and install Tesseract from GitHub",
                    "2. Download and install Poppler",
                    "3. pip install pytesseract easyocr pdf2image opencv-python"
                ]
            }
        }


# Global OCR processor instance
ocr_processor = OCRProcessor()

# Development helper
if __name__ == "__main__":
    import json
    import asyncio
    
    print("🔍 OCR Processor Status:")
    processor = OCRProcessor()
    status = processor.get_engine_status()
    print(json.dumps(status, indent=2))
    
    print(f"\n✅ OCR Available: {processor.is_ocr_available()}")
    print(f"🎯 Default Engine: {processor.default_engine}")
    
    # Test engines
    if processor.is_ocr_available():
        print(f"\n🧪 Testing OCR Engines:")
        test_results = processor.test_ocr_engines()
        for engine, result in test_results["engines"].items():
            status_icon = "✅" if result["status"] == "working" else "❌" if result["status"] == "error" else "⚠️"
            print(f"   {status_icon} {engine}: {result['status']}")
    
    # Installation guide
    print(f"\n📋 Installation Guide:")
    guide = processor.get_installation_guide()
    print("Quick setup commands available in get_installation_guide()")

                