#!/usr/bin/env python3
"""
RagFlow Backend Startup Script
Startet das optimierte Backend mit allen Checks
"""

import sys
import os
import asyncio
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

def check_dependencies():
    """Check if all required dependencies are installed"""
    missing_deps = []
    
    try:
        import fastapi
        import uvicorn
        import numpy
        import sklearn
    except ImportError as e:
        missing_deps.append(str(e))
    
    # Optional dependencies
    optional_deps = {}
    
    try:
        import google.generativeai
        optional_deps['Google AI'] = True
    except ImportError:
        optional_deps['Google AI'] = False
    
    try:
        import sentence_transformers
        optional_deps['Sentence Transformers'] = True
    except ImportError:
        optional_deps['Sentence Transformers'] = False
    
    try:
        import PyPDF2
        optional_deps['PDF Processing (PyPDF2)'] = True
    except ImportError:
        try:
            import pdfplumber
            optional_deps['PDF Processing (pdfplumber)'] = True
        except ImportError:
            optional_deps['PDF Processing'] = False
    
    try:
        from docx import Document
        optional_deps['DOCX Processing'] = True
    except ImportError:
        optional_deps['DOCX Processing'] = False
    
    return missing_deps, optional_deps

def setup_environment():
    """Setup environment and directories"""
    # Create necessary directories
    dirs_to_create = ['data', 'uploads', 'logs']
    
    for dir_name in dirs_to_create:
        dir_path = Path(dir_name)
        dir_path.mkdir(exist_ok=True)
        print(f"✓ Directory created/verified: {dir_path}")
    
    # Check .env file
    env_file = Path('.env')
    env_example = Path('.env.example')
    
    if not env_file.exists() and env_example.exists():
        print("⚠️  .env file not found. Please copy .env.example to .env and configure it.")
        return False
    
    return True

def print_startup_banner():
    """Print startup banner"""
    print("\n" + "="*60)
    print("🚀 RagFlow Backend - Optimized Version")
    print("="*60)
    
    # Check dependencies
    missing_deps, optional_deps = check_dependencies()
    
    if missing_deps:
        print("\n❌ Missing required dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\nPlease install missing dependencies with:")
        print("  pip install -r requirements.txt")
        return False
    
    print("\n✅ All required dependencies available")
    
    print("\n📦 Optional dependencies:")
    for dep, available in optional_deps.items():
        status = "✓" if available else "✗"
        print(f"  {status} {dep}")
    
    # Setup environment
    if not setup_environment():
        return False
    
    print("\n🔧 Configuration:")
    
    # Check environment variables
    google_api_key = os.getenv('GOOGLE_API_KEY')
    print(f"  Google AI API Key: {'✓ Set' if google_api_key else '✗ Not set'}")
    
    if not google_api_key:
        print("  ⚠️  Set GOOGLE_API_KEY in .env for AI features")
    
    print(f"  Upload Directory: {os.getenv('UPLOAD_DIR', './uploads')}")
    print(f"  Data Directory: {os.getenv('DATA_DIR', './data')}")
    print(f"  Debug Mode: {os.getenv('DEBUG', 'false')}")
    
    print("\n🌐 Server will start on: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/api/health")
    
    print("="*60)
    
    return True

async def main():
    """Main startup function"""
    if not print_startup_banner():
        sys.exit(1)
    
    print("\n🔄 Starting RagFlow Backend...")
    
    try:
        # Import after banner to show any import errors clearly
        from app.main import app
        from app.config import settings, print_config_summary
        
        # Print detailed config in debug mode
        if settings.DEBUG:
            print_config_summary()
        
        # Start server
        import uvicorn
        
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=8000,
            reload=settings.RELOAD,
            log_level=settings.LOG_LEVEL.lower(),
            access_log=True
        )
        
        server = uvicorn.Server(config)
        await server.serve()
        
    except KeyboardInterrupt:
        print("\n👋 RagFlow Backend shutdown complete")
    except Exception as e:
        print(f"\n❌ Startup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)