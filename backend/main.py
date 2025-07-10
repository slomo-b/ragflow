#!/usr/bin/env python3
"""
RagFlow Backend - Main Application Entry Point
FastAPI application for AI-powered document analysis
"""

import uvicorn
from app.main import app

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )