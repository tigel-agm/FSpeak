import os
import logging
import subprocess
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from models import (
    FFmpegRequest, 
    FFmpegResponse, 
    ModelConfig, 
    HealthResponse, 
    ErrorResponse,
    ModelProvider
)
from ai_service import AIService


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Global variables
ai_service: Optional[AIService] = None
model_config: Optional[ModelConfig] = None


def load_config() -> ModelConfig:
    """Load configuration from environment variables"""
    return ModelConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.2"),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
        complexity_threshold=float(os.getenv("COMPLEXITY_THRESHOLD", "0.7")),
        prefer_local=os.getenv("PREFER_LOCAL", "false").lower() == "true"
    )


def check_ffmpeg() -> bool:
    """Check if FFmpeg is available"""
    try:
        result = subprocess.run(
            ["C:\\ffmpeg\\bin\\ffmpeg.exe", "-version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global ai_service, model_config
    
    # Startup
    logger.info("Starting FSPEAK - FFmpeg Natural Language Interface")
    
    # Load configuration
    model_config = load_config()
    logger.info("Configuration loaded")
    
    # Initialize AI service
    ai_service = AIService(model_config)
    logger.info("AI service initialized")
    
    # Check FFmpeg availability
    if check_ffmpeg():
        logger.info("FFmpeg found and accessible")
    else:
        logger.warning("FFmpeg not found at C:\\ffmpeg\\bin\\ffmpeg.exe")
    
    # Check model availability
    model_status = await ai_service.get_model_status()
    available_models = [k for k, v in model_status.items() if v]
    if available_models:
        logger.info(f"Available AI models: {', '.join(available_models)}")
    else:
        logger.warning("No AI models are currently available")
    
    yield
    
    # Shutdown
    logger.info("Shutting down FSPEAK")


# Create FastAPI app
app = FastAPI(
    title="FSPEAK - FFmpeg Natural Language Interface",
    description="Convert natural language descriptions to FFmpeg commands using AI",
    version="1.0.0",
    lifespan=lifespan
)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency to get AI service
async def get_ai_service() -> AIService:
    if ai_service is None:
        raise HTTPException(
            status_code=503,
            detail="AI service not initialized"
        )
    return ai_service


# Exception handlers
@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc: ValidationError):
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error="Validation Error",
            error_code="VALIDATION_ERROR",
            details={"validation_errors": exc.errors()}
        ).model_dump()
    )


@app.exception_handler(ValueError)
async def value_error_handler(request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error=str(exc),
            error_code="VALUE_ERROR"
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            error_code="INTERNAL_ERROR",
            details={"message": str(exc)}
        ).model_dump()
    )


# API Routes
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with basic information"""
    return {
        "service": "FSPEAK - FFmpeg Natural Language Interface",
        "version": "1.0.0",
        "description": "Convert natural language to FFmpeg commands",
        "endpoints": {
            "generate": "/generate - Generate FFmpeg command from natural language",
            "health": "/health - Service health check",
            "models": "/models - Available AI models status"
        }
    }


@app.post("/generate", response_model=FFmpegResponse)
async def generate_ffmpeg_command(
    request: FFmpegRequest,
    service: AIService = Depends(get_ai_service)
):
    """Generate FFmpeg command from natural language description"""
    try:
        logger.info(f"Generating command for query: {request.query[:100]}...")
        
        response = await service.generate_ffmpeg_command(request)
        
        logger.info(f"Command generated using {response.model_used} with confidence {response.confidence}")
        return response
        
    except ValueError as e:
        logger.error(f"Value error in command generation: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating command: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate FFmpeg command")


@app.get("/health", response_model=HealthResponse)
async def health_check(service: AIService = Depends(get_ai_service)):
    """Health check endpoint"""
    try:
        # Check AI models
        models_status = await service.get_model_status()
        
        # Check FFmpeg
        ffmpeg_available = check_ffmpeg()
        
        # Determine overall health
        any_model_available = any(models_status.values())
        
        if ffmpeg_available and any_model_available:
            status = "healthy"
        elif ffmpeg_available or any_model_available:
            status = "degraded"
        else:
            status = "unhealthy"
        
        return HealthResponse(
            status=status,
            models_available=models_status,
            ffmpeg_available=ffmpeg_available
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            models_available={},
            ffmpeg_available=False
        )


@app.get("/models", response_model=Dict[str, Any])
async def get_models_status(service: AIService = Depends(get_ai_service)):
    """Get detailed status of all AI model providers"""
    try:
        models_status = await service.get_model_status()
        
        return {
            "models": models_status,
            "config": {
                "openai_model": model_config.openai_model if model_config else None,
                "ollama_model": model_config.ollama_model if model_config else None,
                "gemini_model": model_config.gemini_model if model_config else None,
                "ollama_base_url": model_config.ollama_base_url if model_config else None,
                "gemini_model": model_config.gemini_model if model_config else None,
                "complexity_threshold": model_config.complexity_threshold if model_config else None,
                "prefer_local": model_config.prefer_local if model_config else None
            },
            "available_providers": [k for k, v in models_status.items() if v]
        }
        
    except Exception as e:
        logger.error(f"Error getting models status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get models status")


@app.post("/validate", response_model=Dict[str, Any])
async def validate_query(request: FFmpegRequest):
    """Validate a query without generating the command"""
    try:
        # Basic validation
        if not request.query.strip():
            raise ValueError("Query cannot be empty")
        
        # Analyze complexity
        from ai_service import AIService
        if model_config is None:
            raise ValueError("Model configuration is not initialized")
        temp_service = AIService(model_config)
        complexity = temp_service._analyze_complexity(request.query)
        
        # Suggest model based on complexity
        if complexity > model_config.complexity_threshold:
            suggested_models = [ModelProvider.OPENAI, ModelProvider.GEMINI]
        else:
            suggested_models = [ModelProvider.OLLAMA, ModelProvider.OPENAI]
        
        return {
            "valid": True,
            "complexity_score": complexity,
            "suggested_models": [m.value for m in suggested_models],
            "query_length": len(request.query),
            "word_count": len(request.query.split())
        }
        
    except Exception as e:
        return {
            "valid": False,
            "error": str(e)
        }


if __name__ == "__main__":
    import uvicorn
    
    # Load environment variables from .env file if it exists
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )