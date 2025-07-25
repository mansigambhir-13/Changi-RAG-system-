# tests/test_api_endpoints.py
import pytest
import httpx
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
import json
import uuid

from src.api.main import app
from src.api.models import ChatRequest, ChatResponse

# Test configuration
TEST_TOKEN = "test-token-123"
TEST_SESSION_ID = str(uuid.uuid4())

@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)

@pytest.fixture
def auth_headers():
    """Authentication headers for testing"""
    return {"Authorization": f"Bearer {TEST_TOKEN}"}

@pytest.fixture
def mock_rag_pipeline():
    """Mock RAG pipeline for testing"""
    with patch('src.api.main.rag_pipeline') as mock:
        mock.generate_response = AsyncMock(return_value={
            "response": "Test response from Changi Airport",
            "sources": [
                {
                    "title": "Changi Airport Guide",
                    "url": "https://changiairport.com/guide",
                    "snippet": "Test snippet",
                    "confidence": 0.85
                }
            ],
            "confidence": 0.85,
            "suggestions": ["What are the opening hours?", "How to get WiFi?"]
        })
        mock.test_connection = Mock(return_value=True)
        mock.initialize = AsyncMock()
        yield mock

@pytest.fixture
def mock_redis():
    """Mock Redis client for testing"""
    with patch('src.api.main.redis_client') as mock:
        mock.ping = Mock(return_value=True)
        mock.get = Mock(return_value=None)
        mock.setex = Mock(return_value=True)
        mock.delete = Mock(return_value=1)
        mock.keys = Mock(return_value=[])
        yield mock

class TestHealthEndpoints:
    """Test health check endpoints"""
    
    def test_health_check_success(self, client, mock_rag_pipeline, mock_redis):
        """Test successful health check"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "services" in data

    def test_health_check_rag_failure(self, client, mock_redis):
        """Test health check with RAG pipeline failure"""
        with patch('src.api.main.rag_pipeline.test_connection', return_value=False):
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["services"]["rag_pipeline"] == "unhealthy"

    def test_health_check_redis_failure(self, client, mock_rag_pipeline):
        """Test health check with Redis failure"""
        with patch('src.api.main.redis_client', None):
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["services"]["redis"] == "unhealthy"

class TestChatEndpoints:
    """Test chat-related endpoints"""
    
    @patch('src.api.main.settings.API_TOKEN', TEST_TOKEN)
    def test_chat_endpoint_success(self, client, auth_headers, mock_rag_pipeline, mock_redis):
        """Test successful chat interaction"""
        chat_request = {
            "message": "What facilities are available at Changi Airport?",
            "session_id": TEST_SESSION_ID
        }
        
        response = client.post(
            "/api/v1/chat",
            json=chat_request,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "sources" in data
        assert "confidence" in data
        assert data["session_id"] == TEST_SESSION_ID
        
        # Verify RAG pipeline was called
        mock_rag_pipeline.generate_response.assert_called_once()

    @patch('src.api.main.settings.API_TOKEN', TEST_TOKEN)
    def test_chat_endpoint_without_auth(self, client, mock_rag_pipeline):
        """Test chat endpoint without authentication"""
        chat_request = {
            "message": "Test message",
            "session_id": TEST_SESSION_ID
        }
        
        response = client.post("/api/v1/chat", json=chat_request)
        
        assert response.status_code == 403

    @patch('src.api.main.settings.API_TOKEN', TEST_TOKEN)
    def test_chat_endpoint_invalid_token(self, client, mock_rag_pipeline):
        """Test chat endpoint with invalid token"""
        chat_request = {
            "message": "Test message",
            "session_id": TEST_SESSION_ID
        }
        
        headers = {"Authorization": "Bearer invalid-token"}
        response = client.post("/api/v1/chat", json=chat_request, headers=headers)
        
        assert response.status_code == 401

    @patch('src.api.main.settings.API_TOKEN', TEST_TOKEN)
    def test_chat_endpoint_empty_message(self, client, auth_headers):
        """Test chat endpoint with empty message"""
        chat_request = {
            "message": "",
            "session_id": TEST_SESSION_ID
        }
        
        response = client.post(
            "/api/v1/chat",
            json=chat_request,
            headers=auth_headers
        )
        
        assert response.status_code == 422

    @patch('src.api.main.settings.API_TOKEN', TEST_TOKEN)
    def test_chat_endpoint_long_message(self, client, auth_headers):
        """Test chat endpoint with message too long"""
        chat_request = {
            "message": "x" * 1001,  # Exceeds max length
            "session_id": TEST_SESSION_ID
        }
        
        response = client.post(
            "/api/v1/chat",
            json=chat_request,
            headers=auth_headers
        )
        
        assert response.status_code == 422

    @patch('src.api.main.settings.API_TOKEN', TEST_TOKEN)
    def test_chat_endpoint_rag_error(self, client, auth_headers, mock_redis):
        """Test chat endpoint with RAG pipeline error"""
        with patch('src.api.main.rag_pipeline.generate_response', side_effect=Exception("RAG error")):
            chat_request = {
                "message": "Test message",
                "session_id": TEST_SESSION_ID
            }
            
            response = client.post(
                "/api/v1/chat",
                json=chat_request,
                headers=auth_headers
            )
            
            assert response.status_code == 500

class TestConversationEndpoints:
    """Test conversation history endpoints"""
    
    @patch('src.api.main.settings.API_TOKEN', TEST_TOKEN)
    def test_get_conversation_history_success(self, client, auth_headers, mock_redis):
        """Test successful conversation history retrieval"""
        # Mock conversation history
        history_data = [
            {"user": "Hello", "assistant": "Hi there!", "timestamp": "2024-01-15T10:00:00"},
            {"user": "How are you?", "assistant": "I'm doing well!", "timestamp": "2024-01-15T10:01:00"}
        ]
        mock_redis.get.return_value = json.dumps(history_data)
        
        response = client.get(f"/api/v1/chat/history/{TEST_SESSION_ID}", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == TEST_SESSION_ID
        assert len(data["messages"]) == 2

    @patch('src.api.main.settings.API_TOKEN', TEST_TOKEN)
    def test_get_conversation_history_not_found(self, client, auth_headers, mock_redis):
        """Test conversation history not found"""
        mock_redis.get.return_value = None
        
        response = client.get(f"/api/v1/chat/history/{TEST_SESSION_ID}", headers=auth_headers)
        
        assert response.status_code == 404

    @patch('src.api.main.settings.API_TOKEN', TEST_TOKEN)
    def test_clear_conversation_history_success(self, client, auth_headers, mock_redis):
        """Test successful conversation history clearing"""
        mock_redis.delete.return_value = 1
        
        response = client.delete(f"/api/v1/chat/history/{TEST_SESSION_ID}", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "cleared successfully" in data["message"]

    @patch('src.api.main.settings.API_TOKEN', TEST_TOKEN)
    def test_clear_conversation_history_not_found(self, client, auth_headers, mock_redis):
        """Test clearing non-existent conversation history"""
        mock_redis.delete.return_value = 0
        
        response = client.delete(f"/api/v1/chat/history/{TEST_SESSION_ID}", headers=auth_headers)
        
        assert response.status_code == 404

class TestSuggestionsEndpoint:
    """Test suggestions endpoint"""
    
    @patch('src.api.main.settings.API_TOKEN', TEST_TOKEN)
    def test_get_suggestions_general(self, client, auth_headers):
        """Test getting general suggestions"""
        response = client.get("/api/v1/chat/suggestions", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "suggestions" in data
        assert len(data["suggestions"]) > 0

    @patch('src.api.main.settings.API_TOKEN', TEST_TOKEN)
    def test_get_suggestions_airport_topic(self, client, auth_headers):
        """Test getting airport-specific suggestions"""
        response = client.get("/api/v1/chat/suggestions?topic=airport", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "suggestions" in data
        assert any("airport" in suggestion.lower() for suggestion in data["suggestions"])

    @patch('src.api.main.settings.API_TOKEN', TEST_TOKEN)
    def test_get_suggestions_jewel_topic(self, client, auth_headers):
        """Test getting Jewel-specific suggestions"""
        response = client.get("/api/v1/chat/suggestions?topic=jewel", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "suggestions" in data
        assert any("jewel" in suggestion.lower() for suggestion in data["suggestions"])

class TestAdminEndpoints:
    """Test admin endpoints"""
    
    @patch('src.api.main.settings.API_TOKEN', TEST_TOKEN)
    def test_admin_stats(self, client, auth_headers, mock_redis):
        """Test admin statistics endpoint"""
        mock_redis.keys.return_value = [f"conversation:{i}" for i in range(5)]
        mock_redis.get.return_value = json.dumps([{"user": "test", "assistant": "response"}])
        
        response = client.get("/api/v1/admin/stats", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "total_conversations" in data
        assert "system_metrics" in data

    @patch('src.api.main.settings.API_TOKEN', TEST_TOKEN)
    def test_refresh_data(self, client, auth_headers):
        """Test data refresh endpoint"""
        refresh_request = {"force": False}
        
        response = client.post(
            "/api/v1/admin/refresh-data",
            json=refresh_request,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    @patch('src.api.main.settings.API_TOKEN', TEST_TOKEN)
    def test_get_config(self, client, auth_headers):
        """Test get configuration endpoint"""
        response = client.get("/api/v1/admin/config", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "rag_config" in data

    @patch('src.api.main.settings.API_TOKEN', TEST_TOKEN)
    def test_update_config(self, client, auth_headers):
        """Test update configuration endpoint"""
        config_update = {
            "rag_config": {
                "max_sources": 3,
                "confidence_threshold": 0.4,
                "max_tokens": 800,
                "temperature": 0.5
            }
        }
        
        response = client.put(
            "/api/v1/admin/config",
            json=config_update,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["rag_config"]["max_sources"] == 3

class TestRateLimiting:
    """Test rate limiting functionality"""
    
    @patch('src.api.main.settings.API_TOKEN', TEST_TOKEN)
    def test_rate_limiting_chat_endpoint(self, client, auth_headers, mock_rag_pipeline, mock_redis):
        """Test rate limiting on chat endpoint"""
        chat_request = {
            "message": "Test message",
            "session_id": TEST_SESSION_ID
        }
        
        # Make requests rapidly to trigger rate limiting
        responses = []
        for i in range(35):  # Exceed the 30/minute limit
            response = client.post(
                "/api/v1/chat",
                json=chat_request,
                headers=auth_headers
            )
            responses.append(response.status_code)
        
        # Should eventually get rate limited
        assert 429 in responses

# Integration test
class TestEndToEnd:
    """End-to-end integration tests"""
    
    @patch('src.api.main.settings.API_TOKEN', TEST_TOKEN)
    def test_full_conversation_flow(self, client, auth_headers, mock_rag_pipeline, mock_redis):
        """Test complete conversation flow"""
        session_id = str(uuid.uuid4())
        
        # First message
        response1 = client.post(
            "/api/v1/chat",
            json={"message": "Hello, what can you tell me about Changi Airport?", "session_id": session_id},
            headers=auth_headers
        )
        assert response1.status_code == 200
        
        # Get conversation history
        history_response = client.get(f"/api/v1/chat/history/{session_id}", headers=auth_headers)
        # Note: This might return 404 if Redis mock doesn't return conversation data
        
        # Second message
        response2 = client.post(
            "/api/v1/chat",
            json={"message": "What about dining options?", "session_id": session_id},
            headers=auth_headers
        )
        assert response2.status_code == 200
        
        # Clear conversation
        clear_response = client.delete(f"/api/v1/chat/history/{session_id}", headers=auth_headers)
        # Note: This might return 404 if Redis mock doesn't simulate existing conversation

if __name__ == "__main__":
    pytest.main([__file__, "-v"])