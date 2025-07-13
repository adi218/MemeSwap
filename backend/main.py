import logging
import sys

# Configure logging FIRST, before any other imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log', mode='a')
    ],
    force=True  # Force reconfiguration
)

# Now import other modules
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.api import api_router

logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="API for GIF face swapping application",
    version=settings.VERSION
)

# Add CORS middleware to allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/")
def read_root():
    logger.info("Root endpoint accessed")
    return {"message": "Welcome to the MemeSwap backend! 🚀"}

@app.get("/health")
def health_check():
    """Health check endpoint"""
    logger.info("Health check endpoint accessed")
    return {"status": "healthy", "message": "MemeSwap API is running"}

@app.get("/debug-env")
def debug_env():
    """Debug endpoint to check environment variables"""
    logger.info("Debug env endpoint accessed")
    return {
        "api_key_exists": settings.TENOR_API_KEY is not None,
        "api_key_length": len(settings.TENOR_API_KEY) if settings.TENOR_API_KEY else 0,
        "api_key_preview": settings.TENOR_API_KEY[:10] + "..." if settings.TENOR_API_KEY and len(settings.TENOR_API_KEY) > 10 else settings.TENOR_API_KEY,
        "version": settings.VERSION
    } 