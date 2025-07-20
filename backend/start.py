#!/usr/bin/env python3
"""
RagFlow Backend Startup - Modern Version 2025 (Fixed)
Mit neuesten FastAPI, Pydantic V2, Python 3.13
"""

import sys
import os
import asyncio
from pathlib import Path

# Rich für schöne Ausgaben
try:
    from rich import print as rprint
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    def rprint(*args, **kwargs):
        print(*args, **kwargs)
    console = None

def setup_environment():
    """Setup environment and directories"""
    # Create necessary directories
    dirs_to_create = ['data', 'uploads', 'logs']
    
    for dir_name in dirs_to_create:
        dir_path = Path(dir_name)
        dir_path.mkdir(exist_ok=True)
    
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
        return True
    except ImportError:
        rprint("⚠️ python-dotenv not available - using system environment")
        return True

def check_dependencies():
    """Check dependencies with modern versions"""
    deps_status = {}
    
    # Core dependencies (required)
    core_deps = {
        'fastapi': 'FastAPI',
        'uvicorn': 'Uvicorn', 
        'pydantic': 'Pydantic V2',
        'pydantic_settings': 'Pydantic Settings'
    }
    
    # Optional dependencies
    optional_deps = {
        'google.generativeai': 'Google AI',
        'numpy': 'NumPy',
        'sklearn': 'Scikit-learn',
        'sentence_transformers': 'Sentence Transformers',
        'pypdf': 'PyPDF (modern)',
        'docx': 'Python-DOCX',
        'chromadb': 'ChromaDB',
        'rich': 'Rich (terminal formatting)'
    }
    
    missing_core = []
    
    # Check core dependencies
    for module, name in core_deps.items():
        try:
            __import__(module)
            deps_status[name] = "✅"
        except ImportError:
            deps_status[name] = "❌ MISSING"
            missing_core.append(name)
    
    # Check optional dependencies
    for module, name in optional_deps.items():
        try:
            __import__(module)
            deps_status[name] = "✅"
        except ImportError:
            deps_status[name] = "⚠️ Optional"
    
    return deps_status, missing_core

def print_startup_banner():
    """Print modern startup banner"""
    if RICH_AVAILABLE:
        return print_rich_banner()
    else:
        return print_simple_banner()

def print_rich_banner():
    """Rich formatted banner"""
    console.print()
    
    # Main banner
    banner = Panel.fit(
        "[bold blue]🚀 RagFlow Backend[/bold blue]\n"
        "[green]Modern Version 3.0.0[/green]\n"
        "[dim]Latest FastAPI + Pydantic V2 + Python 3.13[/dim]",
        border_style="blue"
    )
    console.print(banner)
    
    # Dependencies check
    deps_status, missing_core = check_dependencies()
    
    # Create dependency table
    table = Table(title="📦 Dependencies Status", show_header=True, header_style="bold magenta")
    table.add_column("Component", style="cyan", no_wrap=True)
    table.add_column("Status", style="green")
    
    for name, status in deps_status.items():
        table.add_row(name, status)
    
    console.print()
    console.print(table)
    
    # Check if any core dependencies are missing
    if missing_core:
        console.print(f"\n[bold red]❌ Missing required dependencies:[/bold red]")
        for dep in missing_core:
            console.print(f"  - {dep}")
        console.print("\n[yellow]Install with:[/yellow]")
        console.print("  [dim]pip install -r requirements.txt[/dim]")
        return False
    
    # Environment info
    google_key = os.getenv('GOOGLE_API_KEY')
    console.print()
    console.print("[bold yellow]🔧 Configuration:[/bold yellow]")
    console.print(f"  Google AI API Key: {'✅ Configured' if google_key else '⚠️ Not set'}")
    console.print(f"  Upload Directory: [cyan]{os.getenv('UPLOAD_DIR', './uploads')}[/cyan]")
    console.print(f"  Data Directory: [cyan]{os.getenv('DATA_DIR', './data')}[/cyan]")
    
    # Server info
    console.print()
    console.print("[bold green]🌐 Server URLs:[/bold green]")
    console.print("  Main: [link=http://localhost:8000]http://localhost:8000[/link]")
    console.print("  API Docs: [link=http://localhost:8000/docs]http://localhost:8000/docs[/link]")
    console.print("  Health: [link=http://localhost:8000/api/health]http://localhost:8000/api/health[/link]")
    
    console.print()
    return True  # All good!

def print_simple_banner():
    """Simple banner for systems without Rich"""
    print("\n" + "="*80)
    print("🚀 RagFlow Backend - Modern Version 3.0.0")
    print("Latest FastAPI + Pydantic V2 + Python 3.13")
    print("="*80)
    
    # Check dependencies
    deps_status, missing_core = check_dependencies()
    
    if missing_core:
        print("\n❌ Missing required dependencies:")
        for dep in missing_core:
            print(f"  - {dep}")
        print("\nInstall with: pip install -r requirements.txt")
        return False
    
    print("\n📦 Dependencies:")
    for name, status in deps_status.items():
        print(f"  {status} {name}")
    
    # Environment
    google_key = os.getenv('GOOGLE_API_KEY')
    print(f"\n🔧 Configuration:")
    print(f"  Google AI API Key: {'✅ Configured' if google_key else '⚠️ Not set'}")
    print(f"  Upload Directory: {os.getenv('UPLOAD_DIR', './uploads')}")
    print(f"  Data Directory: {os.getenv('DATA_DIR', './data')}")
    
    print(f"\n🌐 Server URLs:")
    print(f"  Main: http://localhost:8000")
    print(f"  API Docs: http://localhost:8000/docs")
    print(f"  Health: http://localhost:8000/api/health")
    
    print("="*80)
    return True

async def main():
    """Modern async main function"""
    # Setup environment
    setup_environment()
    
    # Show banner and check dependencies
    if not print_startup_banner():
        rprint("\n❌ Setup failed - please install missing dependencies")
        sys.exit(1)
    
    if RICH_AVAILABLE:
        console.print("\n[bold green]✅ Environment setup complete[/bold green]")
        console.print("[dim]Starting server...[/dim]")
    else:
        print("\n✅ Environment setup complete")
        print("Starting server...")
    
    try:
        # Import after checks
        from app.main import app, settings
        import uvicorn
        
        # Configure uvicorn
        config = uvicorn.Config(
            app,
            host=settings.host,
            port=settings.port,
            reload=settings.reload,
            log_level="info",
            access_log=False,  # Cleaner output
        )
        
        server = uvicorn.Server(config)
        
        if RICH_AVAILABLE:
            console.print(f"\n[bold green]🎉 Server starting on {settings.host}:{settings.port}[/bold green]")
        else:
            print(f"\n🎉 Server starting on {settings.host}:{settings.port}")
        
        # Start server
        await server.serve()
        
    except ImportError as e:
        rprint(f"\n❌ Import error: {e}")
        rprint("Make sure all dependencies are installed:")
        rprint("  pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        rprint(f"\n❌ Server error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def sync_main():
    """Synchronous wrapper for main"""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        if RICH_AVAILABLE:
            console.print("\n[yellow]👋 Shutdown requested by user[/yellow]")
        else:
            print("\n👋 Shutdown requested by user")
        sys.exit(0)
    except Exception as e:
        rprint(f"\n❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    sync_main()