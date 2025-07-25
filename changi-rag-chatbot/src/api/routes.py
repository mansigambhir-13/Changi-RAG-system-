# src/api/routes.py
from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional
import json
import redis
from datetime import datetime, timedelta

from .models import (
    ConversationHistory, AdminStatsResponse, RefreshDataRequest, 
    RefreshDataResponse, ConfigResponse, UpdateConfigRequest, RAGConfig
)
from ..rag.pipeline import RAGPipeline
from ..utils.config import Settings
from ..utils.logger import setup_logger

settings = Settings()
logger = setup_logger(__name__)

# Initialize routers
chat_router = APIRouter(prefix="/chat", tags=["Chat"])
admin_router = APIRouter(tags=["Admin"])

# Redis client (same as main.py)
try:
    redis_client = redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        password=settings.REDIS_PASSWORD,
        decode_responses=True
    )
except:
    redis_client = None

# Chat routes
@chat_router.get("/history/{session_id}", response_model=ConversationHistory)
async def get_conversation_history(session_id: str):
    """Get conversation history for a session"""
    try:
        if not redis_client:
            raise HTTPException(status_code=503, detail="Conversation history unavailable")
            
        history_key = f"conversation:{session_id}"
        history_data = redis_client.get(history_key)
        
        if not history_data:
            raise HTTPException(status_code=404, detail="Conversation not found")
            
        messages = json.loads(history_data)
        
        return ConversationHistory(
            session_id=session_id,
            messages=messages,
            created_at=datetime.utcnow() - timedelta(hours=1),  # Approximate
            updated_at=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving conversation history: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@chat_router.delete("/history/{session_id}")
async def clear_conversation_history(session_id: str):
    """Clear conversation history for a session"""
    try:
        if not redis_client:
            raise HTTPException(status_code=503, detail="Conversation history unavailable")
            
        history_key = f"conversation:{session_id}"
        deleted = redis_client.delete(history_key)
        
        if not deleted:
            raise HTTPException(status_code=404, detail="Conversation not found")
            
        return {"message": "Conversation history cleared successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing conversation history: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@chat_router.get("/suggestions")
async def get_suggestions(topic: Optional[str] = Query(None)):
    """Get suggested questions"""
    try:
        # Default suggestions by topic
        suggestions = {
            "airport": [
                "What facilities are available at Changi Airport?",
                "How do I get to Changi Airport?",
                "What are the operating hours?",
                "Where can I find WiFi at the airport?"
            ],
            "jewel": [
                "What shops are in Jewel Changi Airport?",
                "How do I get to the waterfall?",
                "What dining options are available?",
                "What are the attraction timings?"
            ],
            "general": [
                "What can you help me with?",
                "Tell me about Changi Airport",
                "What's special about Jewel?",
                "How do I navigate the airport?"
            ]
        }
        
        if topic and topic in suggestions:
            return {"suggestions": suggestions[topic]}
        else:
            return {"suggestions": suggestions["general"]}
            
    except Exception as e:
        logger.error(f"Error getting suggestions: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Admin routes
@admin_router.get("/stats", response_model=AdminStatsResponse)
async def get_admin_stats():
    """Get system statistics"""
    try:
        stats = {
            "total_conversations": 0,
            "total_messages": 0,
            "average_confidence": 0.0,
            "popular_topics": [],
            "system_metrics": {}
        }
        
        if redis_client:
            # Get conversation statistics
            conversation_keys = redis_client.keys("conversation:*")
            stats["total_conversations"] = len(conversation_keys)
            
            # Count total messages and calculate average confidence
            total_messages = 0
            total_confidence = 0.0
            
            for key in conversation_keys[:100]:  # Limit to prevent timeout
                try:
                    history = json.loads(redis_client.get(key) or "[]")
                    total_messages += len(history)
                    # You can add confidence tracking here
                except:
                    continue
                    
            stats["total_messages"] = total_messages
            stats["average_confidence"] = 0.85  # Mock value
            
        # Mock popular topics
        stats["popular_topics"] = [
            "Airport facilities",
            "Transportation",
            "Shopping",
            "Dining",
            "Attractions"
        ]
        
        # System metrics
        stats["system_metrics"] = {
            "uptime": "24h",
            "cpu_usage": "45%",
            "memory_usage": "67%",
            "response_time": "1.2s"
        }
        
        return AdminStatsResponse(**stats)
        
    except Exception as e:
        logger.error(f"Error getting admin stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@admin_router.post("/refresh-data", response_model=RefreshDataResponse)
async def refresh_data(request: RefreshDataRequest):
    """Refresh the knowledge base data"""
    try:
        # This would trigger your data refresh pipeline
        # For now, return a mock response
        
        return RefreshDataResponse(
            status="success",
            message="Data refresh initiated successfully",
            processed_documents=150,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Error refreshing data: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@admin_router.get("/config", response_model=ConfigResponse)
async def get_config():
    """Get current configuration"""
    try:
        # Return current RAG configuration
        rag_config = RAGConfig(
            max_sources=5,
            confidence_threshold=0.3,
            max_tokens=1000,
            temperature=0.7
        )
        
        return ConfigResponse(
            rag_config=rag_config,
            updated_at=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Error getting config: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@admin_router.put("/config", response_model=ConfigResponse)
async def update_config(request: UpdateConfigRequest):
    """Update configuration"""
    try:
        # Update configuration (in production, this would persist to database)
        if request.rag_config:
            # Apply configuration updates
            logger.info(f"Configuration updated: {request.rag_config}")
            
        return ConfigResponse(
            rag_config=request.rag_config or RAGConfig(),
            updated_at=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@admin_router.get("/health-detailed")
async def detailed_health_check():
    """Detailed health check for admin"""
    try:
        health_info = {
            "api": "healthy",
            "redis": "healthy" if redis_client and redis_client.ping() else "unhealthy",
            "vector_db": "healthy",  # Mock - replace with actual check
            "rag_pipeline": "healthy",  # Mock - replace with actual check
            "last_data_refresh": "2024-01-15T10:30:00Z",
            "active_sessions": len(redis_client.keys("conversation:*")) if redis_client else 0,
            "system_info": {
                "python_version": "3.9.16",
                "fastapi_version": "0.104.1",
                "environment": settings.ENVIRONMENT
            }
        }
        
        return health_info
        
    except Exception as e:
        logger.error(f"Error in detailed health check: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")