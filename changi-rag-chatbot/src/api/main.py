# src/api/main.py
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import uvicorn
import redis
import json
import os
from datetime import datetime
from typing import Optional
import logging

from .models import ChatRequest, ChatResponse, HealthResponse
from .routes import chat_router, admin_router
from ..rag.pipeline import RAGPipeline
from ..utils.config import Settings
from ..utils.logger import setup_logger

# Initialize settings and logger
settings = Settings()
logger = setup_logger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Initialize Redis for conversation memory
try:
    redis_client = redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        password=settings.REDIS_PASSWORD,
        decode_responses=True
    )
    redis_client.ping()
    logger.info("Connected to Redis successfully")
except Exception as e:
    logger.warning(f"Redis connection failed: {e}. Using in-memory storage.")
    redis_client = None

# Initialize RAG pipeline
rag_pipeline = RAGPipeline()

# Initialize FastAPI app
app = FastAPI(
    title="Changi Airport RAG Chatbot API",
    description="Intelligent chatbot for Changi Airport and Jewel Changi Airport information",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Security
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API token"""
    if credentials.credentials != settings.API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    return credentials.credentials

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=settings.ALLOWED_HOSTS
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Routes
app.include_router(chat_router, prefix="/api/v1", dependencies=[Depends(verify_token)])
app.include_router(admin_router, prefix="/api/v1/admin", dependencies=[Depends(verify_token)])

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Test RAG pipeline
        test_result = rag_pipeline.test_connection()
        
        # Test Redis
        redis_status = "healthy" if redis_client and redis_client.ping() else "unhealthy"
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.utcnow(),
            services={
                "rag_pipeline": "healthy" if test_result else "unhealthy",
                "redis": redis_status,
                "vector_db": "healthy" if test_result else "unhealthy"
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.post("/api/v1/chat", response_model=ChatResponse)
@limiter.limit("30/minute")
async def chat_endpoint(
    request: ChatRequest,
    remote_addr: str = Depends(get_remote_address),
    token: str = Depends(verify_token)
):
    """Main chat endpoint"""
    try:
        # Get conversation history
        conversation_history = []
        if redis_client and request.session_id:
            history_key = f"conversation:{request.session_id}"
            history_data = redis_client.get(history_key)
            if history_data:
                conversation_history = json.loads(history_data)

        # Generate response using RAG pipeline
        response_data = await rag_pipeline.generate_response(
            query=request.message,
            conversation_history=conversation_history,
            session_id=request.session_id
        )

        # Update conversation history
        if redis_client and request.session_id:
            conversation_history.append({
                "user": request.message,
                "assistant": response_data["response"],
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Keep only last 10 exchanges
            conversation_history = conversation_history[-10:]
            
            redis_client.setex(
                f"conversation:{request.session_id}",
                3600,  # 1 hour expiry
                json.dumps(conversation_history)
            )

        return ChatResponse(
            response=response_data["response"],
            sources=response_data.get("sources", []),
            confidence=response_data.get("confidence", 0.0),
            session_id=request.session_id,
            suggestions=response_data.get("suggestions", [])
        )

    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Internal server error processing your request"
        )

@app.on_event("startup")
async def startup_event():
    """Application startup"""
    logger.info("Starting Changi Airport RAG Chatbot API")
    
    # Initialize RAG pipeline
    try:
        await rag_pipeline.initialize()
        logger.info("RAG pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown"""
    logger.info("Shutting down Changi Airport RAG Chatbot API")
    
    # Cleanup
    if redis_client:
        redis_client.close()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_config=None
    )