from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Literal, Dict, Any, List
from enum import Enum


class ModelProvider(str, Enum):
    """Available AI model providers"""
    OPENAI = "openai"
    OLLAMA = "ollama"
    GEMINI = "gemini"
    AUTO = "auto"


class FFmpegRequest(BaseModel):
    """Request model for FFmpeg command generation"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid"
    )
    
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Natural language description of the desired FFmpeg operation",
        examples=[
            "convert video.mp4 to audio.mp3",
            "crop the first 10 seconds of video.avi",
            "add subtitles.srt to movie.mp4 and compress to 720p"
        ]
    )
    
    model_provider: ModelProvider = Field(
        default=ModelProvider.AUTO,
        description="AI model provider to use for command generation"
    )
    
    input_file: Optional[str] = Field(
        default=None,
        description="Input file path (optional, can be extracted from query)"
    )
    
    output_file: Optional[str] = Field(
        default=None,
        description="Output file path (optional, can be extracted from query)"
    )
    
    additional_context: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Additional context or constraints for the operation"
    )


class FFmpegResponse(BaseModel):
    """Response model for FFmpeg command generation"""
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid"
    )
    
    command: str = Field(
        ...,
        description="Generated FFmpeg command"
    )
    
    explanation: str = Field(
        ...,
        description="Human-readable explanation of what the command does"
    )
    
    model_used: ModelProvider = Field(
        ...,
        description="AI model provider that generated this command"
    )
    
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for the generated command (0.0 to 1.0)"
    )
    
    warnings: List[str] = Field(
        default_factory=list,
        description="Any warnings or considerations for the command"
    )
    
    estimated_processing_time: Optional[str] = Field(
        default=None,
        description="Estimated processing time (if determinable)"
    )


class ModelConfig(BaseModel):
    """Configuration for AI model providers"""
    model_config = ConfigDict(
        validate_assignment=True,
        extra="allow"  # Allow extra fields for provider-specific configs
    )
    
    openai_api_key: Optional[str] = Field(default=None)
    openai_model: str = Field(default="gpt-4o-mini")
    
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama3.2")
    
    gemini_api_key: Optional[str] = Field(default=None)
    gemini_model: str = Field(default="gemini-1.5-flash")
    
    # Routing logic weights
    complexity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    prefer_local: bool = Field(default=False)


class HealthResponse(BaseModel):
    """Health check response"""
    status: Literal["healthy", "degraded", "unhealthy"] = "healthy"
    models_available: Dict[str, bool] = Field(default_factory=dict)
    ffmpeg_available: bool = True
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")