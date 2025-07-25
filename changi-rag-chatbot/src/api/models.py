# src/api/models.py
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any
from datetime import datetime
import uuid

class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str = Field(..., min_length=1, max_length=1000, description="User message")
    session_id: Optional[str] = Field(default=None, description="Session ID for conversation continuity")
    context: Optional[Dict[str, Any]] = Field(default={}, description="Additional context")
    
    @validator('session_id', pre=True, always=True)
    def generate_session_id(cls, v):
        return v or str(uuid.uuid4())
    
    @validator('message')
    def validate_message(cls, v):
        if not v.strip():
            raise ValueError('Message cannot be empty')
        return v.strip()

class Source(BaseModel):
    """Source information for responses"""
    title: str = Field(..., description="Title of the source")
    url: str = Field(..., description="URL of the source")
    snippet: str = Field(..., description="Relevant snippet from source")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")

class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    response: str = Field(..., description="Generated response")
    sources: List[Source] = Field(default=[], description="Sources used for response")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence score")
    session_id: str = Field(..., description="Session ID")
    suggestions: List[str] = Field(default=[], description="Follow-up question suggestions")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Overall health status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    services: Dict[str, str] = Field(..., description="Status of individual services")

class ConversationHistory(BaseModel):
    """Conversation history model"""
    session_id: str = Field(..., description="Session ID")
    messages: List[Dict[str, str]] = Field(..., description="Conversation messages")
    created_at: datetime = Field(..., description="Conversation start time")
    updated_at: datetime = Field(..., description="Last update time")

class AdminStatsResponse(BaseModel):
    """Admin statistics response"""
    total_conversations: int = Field(..., description="Total number of conversations")
    total_messages: int = Field(..., description="Total number of messages")
    average_confidence: float = Field(..., description="Average confidence score")
    popular_topics: List[str] = Field(..., description="Most popular topics")
    system_metrics: Dict[str, Any] = Field(..., description="System performance metrics")

class RefreshDataRequest(BaseModel):
    """Request to refresh data"""
    force: bool = Field(default=False, description="Force refresh even if recent")
    sources: Optional[List[str]] = Field(default=None, description="Specific sources to refresh")

class RefreshDataResponse(BaseModel):
    """Response for data refresh"""
    status: str = Field(..., description="Refresh status")
    message: str = Field(..., description="Refresh message")
    processed_documents: int = Field(..., description="Number of documents processed")
    timestamp: datetime = Field(..., description="Refresh timestamp")

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")

# Configuration models
class RAGConfig(BaseModel):
    """RAG configuration model"""
    max_sources: int = Field(default=5, ge=1, le=10, description="Maximum number of sources to retrieve")
    confidence_threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="Minimum confidence threshold")
    max_tokens: int = Field(default=1000, ge=100, le=4000, description="Maximum response tokens")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Response creativity")

class UpdateConfigRequest(BaseModel):
    """Request to update configuration"""
    rag_config: Optional[RAGConfig] = Field(default=None, description="RAG configuration updates")
    
class ConfigResponse(BaseModel):
    """Configuration response"""
    rag_config: RAGConfig = Field(..., description="Current RAG configuration")
    updated_at: datetime = Field(..., description="Last update time")