"""
FastAPI main application entry point.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.routers import health, predictions, music_generation

app = FastAPI(
    title="Compose Pro Backend",
    description="FastAPI backend for ML model serving and music generation",
    version="1.0.0",
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(predictions.router, prefix="/api", tags=["predictions"])
app.include_router(music_generation.router, prefix="/api", tags=["music-generation"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to Compose Pro Backend API",
        "docs": "/docs",
        "health": "/api/health"
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
