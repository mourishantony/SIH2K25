"""FastAPI Backend for Patient Contact Tracing System."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Add src directory to path for imports
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
BACKEND_DIR = ROOT_DIR / "backend"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(BACKEND_DIR))

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables
load_dotenv(ROOT_DIR / ".env", override=True)

# Import routers
from routers import auth, persons, mdr, alerts, dashboard, face_registration, unknown_persons, monitoring, pathogens

# Import database initialization
from database import initialize_database


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    print("[API] Starting Patient Contact Tracing API...")
    initialize_database()
    print("[API] API started successfully!")
    yield
    # Shutdown
    from database import close_connection
    close_connection()
    print("[API] API shutdown complete.")


# Create FastAPI app
app = FastAPI(
    title="Patient Contact Tracing API",
    description="API for Patient Contact Tracing System - SIH 2025",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",  # Vite default
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["Dashboard"])
app.include_router(persons.router, prefix="/api/persons", tags=["Persons"])
app.include_router(face_registration.router, prefix="/api/face", tags=["Face Registration"])
app.include_router(mdr.router, prefix="/api/mdr", tags=["MDR Management"])
app.include_router(pathogens.router, prefix="/api/pathogens", tags=["Pathogen Management"])
app.include_router(alerts.router, prefix="/api/alerts", tags=["Alerts"])
app.include_router(unknown_persons.router, prefix="/api/unknown", tags=["Unknown Persons"])
app.include_router(monitoring.router, prefix="/api/monitoring", tags=["AI Monitoring"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Patient Contact Tracing API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("BACKEND_HOST", "0.0.0.0")
    port = int(os.getenv("BACKEND_PORT", "8000"))
    debug = os.getenv("BACKEND_DEBUG", "true").lower() == "true"
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug
    )
