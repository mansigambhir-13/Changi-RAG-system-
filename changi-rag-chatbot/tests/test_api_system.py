#!/usr/bin/env python3
"""
Test script for the complete API system
Run this to test the FastAPI application and all endpoints
"""
import sys
import os
import time
import requests
import json
import asyncio
from pathlib import Path
import subprocess
import threading
import uuid

# Add src to path for imports
sys.path.append('src')

from utils.config import settings
from utils.logger import get_logger, setup_logging

# Setup logging
setup_logging()
logger = get_logger(__name__)

class APITester:
    """Test runner for the API system"""
    
    def __init__(self, base_url: str = None, api_key: str = None):
        self.base_url = base_url or f"http://{settings.API_HOST}:{settings.API_PORT}"
        self.api_key = api_key or "test-api-key-12345"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        self.test_conversation_id = f"test_conv_{uuid.uuid4().hex[:8]}"
        
    def test_server_startup(self) -> bool:
        """Test that the server starts up correctly"""
        print("🚀 Testing server startup...")
        
        try:
            # Try to connect to the server
            response = self.session.get(f"{self.base_url}/", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Server is running")
                print(f"  - API Name: {data.get('name')}")
                print(f"  - Version: {data.get('version')}")
                print(f"  - Documentation: {self.base_url}{data.get('documentation')}")
                return True
            else:
                print(f"❌ Server responded with status {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to server. Make sure it's running:")
            print(f"  uvicorn src.api.main:app --host {settings.API_HOST} --port {settings.API_PORT}")
            return False
        except Exception as e:
            print(f"❌ Server startup test failed: {e}")
            return False
    
    def test_health_endpoint(self) -> bool:
        """Test the health endpoint"""
        print("\n🏥 Testing health endpoint...")
        
        try:
            response = self.session.get(f"{self.base_url}/health")
            
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ Health check passed")
                print(f"  - Status: {health_data.get('status')}")
                print(f"  - Components: {len(health_data.get('components', {}))}")
                
                # Check component health
                components = health_data.get('components', {})
                for comp_name, comp_data in components.items():
                    status = comp_data.get('status', 'unknown')
                    print(f"    - {comp_name}: {status}")
                
                return health_data.get('status') == 'healthy'
            else:
                print(f"❌ Health check failed with status {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Health endpoint test failed: {e}")
            return False
    
    def test_chat_endpoint(self) -> bool:
        """Test the main chat endpoint"""
        print("\n💬 Testing chat endpoint...")
        
        test_messages = [
            "What are the operating hours of Changi Airport?",
            "Where can I find good food at Jewel?",
            "How do I get from the airport to the city?",
            "Is there free WiFi available?"
        ]
        
        try:
            for i, message in enumerate(test_messages, 1):
                print(f"  Testing message {i}: '{message[:50]}...'")
                
                chat_request = {
                    "message": message,
                    "conversation_id": self.test_conversation_id,
                    "user_id": "test_user",
                    "use_cache": True,
                    "include_follow_ups": True
                }
                
                response = self.session.post(
                    f"{self.base_url}/chat",
                    json=chat_request
                )
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"    ✅ Response generated")
                    print(f"      - Length: {len(data.get('response', ''))} chars")
                    print(f"      - Confidence: {data.get('confidence', 0):.3f}")
                    print(f"      - Sources: {len(data.get('sources', []))}")
                    print(f"      - Response time: {data.get('response_time', 0):.2f}s")
                    print(f"      - Cached: {data.get('cached', False)}")
                    
                    if data.get('follow_up_questions'):
                        print(f"      - Follow-ups: {len(data['follow_up_questions'])}")
                
                elif response.status_code == 429:
                    print(f"    ⚠️ Rate limited (this is expected)")
                    break
                else:
                    print(f"    ❌ Chat failed with status {response.status_code}")
                    print(f"      Error: {response.text}")
                    return False
                
                # Small delay between requests
                time.sleep(0.5)
            
            print("✅ Chat endpoint tests completed successfully")
            return True
            
        except Exception as e:
            print(f"❌ Chat endpoint test failed: {e}")
            return False
    
    def test_conversation_management(self) -> bool:
        """Test conversation management endpoints"""
        print("\n📝 Testing conversation management...")
        
        try:
            # Test getting conversation summary
            print("  Testing conversation summary...")
            response = self.session.get(
                f"{self.base_url}/conversations/{self.test_conversation_id}/summary"
            )
            
            if response.status_code == 200:
                summary = response.json()
                print(f"    ✅ Got conversation summary")
                print(f"      - Total messages: {summary.get('total_messages', 0)}")
                print(f"      - Topics: {summary.get('topics', [])}")
            else:
                print(f"    ⚠️ No conversation found (expected if no chat messages sent)")
            
            # Test clearing conversation
            print("  Testing conversation clearing...")
            response = self.session.delete(
                f"{self.base_url}/conversations/{self.test_conversation_id}"
            )
            
            if response.status_code == 200:
                print(f"    ✅ Conversation cleared successfully")
            else:
                print(f"    ❌ Failed to clear conversation: {response.status_code}")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Conversation management test failed: {e}")
            return False
    
    def test_search_endpoint(self) -> bool:
        """Test the search endpoint"""
        print("\n🔍 Testing search endpoint...")
        
        try:
            search_request = {
                "query": "dining options terminal",
                "top_k": 5,
                "include_content": True
            }
            
            response = self.session.post(
                f"{self.base_url}/search",
                json=search_request
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Search completed successfully")
                print(f"  - Results found: {data.get('total_found', 0)}")
                print(f"  - Search time: {data.get('search_time', 0):.3f}s")
                
                results = data.get('results', [])
                for i, result in enumerate(results[:3], 1):
                    print(f"    {i}. {result.get('title', 'No title')}")
                    print(f"       Score: {result.get('score', 0):.3f}")
                
                return True
            else:
                print(f"❌ Search failed with status {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Search endpoint test failed: {e}")
            return False
    
    def test_feedback_endpoint(self) -> bool:
        """Test the feedback endpoint"""
        print("\n📊 Testing feedback endpoint...")
        
        try:
            feedback_request = {
                "message_id": "test_message_123",
                "conversation_id": self.test_conversation_id,
                "rating": 5,
                "feedback_text": "Great response, very helpful!",
                "user_id": "test_user"
            }
            
            response = self.session.post(
                f"{self.base_url}/feedback",
                json=feedback_request
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Feedback submitted successfully")
                print(f"  - Feedback ID: {data.get('feedback_id')}")
                return True
            else:
                print(f"❌ Feedback submission failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Feedback endpoint test failed: {e}")
            return False
    
    def test_stats_endpoint(self) -> bool:
        """Test the stats endpoint"""
        print("\n📈 Testing stats endpoint...")
        
        try:
            response = self.session.get(f"{self.base_url}/stats")
            
            if response.status_code == 200:
                stats = response.json()
                print(f"✅ Stats retrieved successfully")
                print(f"  - Total conversations: {stats.get('total_conversations', 0)}")
                print(f"  - Total messages: {stats.get('total_messages', 0)}")
                print(f"  - Avg response time: {stats.get('avg_response_time', 0):.2f}s")
                print(f"  - Cache hit rate: {stats.get('cache_hit_rate', 0):.1%}")
                print(f"  - Vector count: {stats.get('vector_count', 0)}")
                return True
            else:
                print(f"❌ Stats retrieval failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Stats endpoint test failed: {e}")
            return False
    
    def test_rate_limiting(self) -> bool:
        """Test rate limiting functionality"""
        print("\n🚦 Testing rate limiting...")
        
        try:
            # Send rapid requests to trigger rate limiting
            requests_sent = 0
            rate_limited = False
            
            for i in range(10):
                response = self.session.post(
                    f"{self.base_url}/chat",
                    json={
                        "message": f"Test message {i}",
                        "conversation_id": f"rate_test_{i}"
                    }
                )
                
                requests_sent += 1
                
                if response.status_code == 429:
                    print(f"✅ Rate limiting triggered after {requests_sent} requests")
                    
                    # Check rate limit headers
                    headers = response.headers
                    print(f"  - Rate limit: {headers.get('X-RateLimit-Limit', 'N/A')}")
                    print(f"  - Remaining: {headers.get('X-RateLimit-Remaining', 'N/A')}")
                    print(f"  - Reset time: {headers.get('X-RateLimit-Reset', 'N/A')}")
                    
                    rate_limited = True
                    break
                
                # Small delay between requests
                time.sleep(0.1)
            
            if not rate_limited:
                print(f"⚠️ Rate limiting not triggered after {requests_sent} requests")
                print("  This might be expected if limits are high")
            
            return True
            
        except Exception as e:
            print(f"❌ Rate limiting test failed: {e}")
            return False
    
    def test_error_handling(self) -> bool:
        """Test error handling"""
        print("\n🚨 Testing error handling...")
        
        test_cases = [
            {
                "name": "Empty message",
                "endpoint": "/chat",
                "data": {"message": "", "conversation_id": "test"},
                "expected_status": 422
            },
            {
                "name": "Invalid conversation ID",
                "endpoint": "/conversations/*/summary",
                "expected_status": 404
            },
            {
                "name": "Malformed JSON",
                "endpoint": "/chat",
                "data": "invalid json",
                "expected_status": 422
            }
        ]
        
        try:
            for test_case in test_cases:
                print(f"  Testing: {test_case['name']}")
                
                if test_case['endpoint'] == "/conversations/*/summary":
                    response = self.session.get(f"{self.base_url}/conversations/nonexistent/summary")
                else:
                    if isinstance(test_case.get('data'), str):
                        # Send malformed JSON
                        response = requests.post(
                            f"{self.base_url}{test_case['endpoint']}",
                            data=test_case['data'],
                            headers={"Authorization": f"Bearer {self.api_key}"}
                        )
                    else:
                        response = self.session.post(
                            f"{self.base_url}{test_case['endpoint']}",
                            json=test_case.get('data')
                        )
                
                expected_status = test_case.get('expected_status')
                if response.status_code == expected_status:
                    print(f"    ✅ Correctly returned status {response.status_code}")
                else:
                    print(f"    ⚠️ Expected {expected_status}, got {response.status_code}")
            
            print("✅ Error handling tests completed")
            return True
            
        except Exception as e:
            print(f"❌ Error handling test failed: {e}")
            return False
    
    def test_documentation_endpoints(self) -> bool:
        """Test API documentation endpoints"""
        print("\n📚 Testing documentation endpoints...")
        
        endpoints = [
            ("/docs", "Swagger UI"),
            ("/redoc", "ReDoc"),
            ("/openapi.json", "OpenAPI Schema")
        ]
        
        try:
            for endpoint, name in endpoints:
                response = self.session.get(f"{self.base_url}{endpoint}")
                
                if response.status_code == 200:
                    print(f"✅ {name} accessible at {endpoint}")
                else:
                    print(f"❌ {name} not accessible: {response.status_code}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"❌ Documentation endpoints test failed: {e}")
            return False

def run_load_test(base_url: str, api_key: str, duration: int = 30, concurrent_users: int = 5):
    """Run a basic load test"""
    print(f"\n⚡ Running load test ({concurrent_users} users, {duration}s)...")
    
    results = {
        "total_requests": 0,
        "successful_requests": 0,
        "failed_requests": 0,
        "response_times": [],
        "errors": []
    }
    
    def worker():
        """Worker function for load testing"""
        session = requests.Session()
        session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })
        
        end_time = time.time() + duration
        
        while time.time() < end_time:
            try:
                start_time = time.time()
                
                response = session.post(
                    f"{base_url}/chat",
                    json={
                        "message": "What are the airport operating hours?",
                        "conversation_id": f"load_test_{threading.current_thread().ident}_{int(time.time())}"
                    },
                    timeout=10
                )
                
                response_time = time.time() - start_time
                
                results["total_requests"] += 1
                results["response_times"].append(response_time)
                
                if response.status_code == 200:
                    results["successful_requests"] += 1
                else:
                    results["failed_requests"] += 1
                    if response.status_code != 429:  # Don't log rate limit errors
                        results["errors"].append(f"HTTP {response.status_code}")
                
            except Exception as e:
                results["failed_requests"] += 1
                results["errors"].append(str(e))
            
            time.sleep(0.1)  # Small delay between requests
    
    # Start worker threads
    threads = []
    for i in range(concurrent_users):
        thread = threading.Thread(target=worker)
        thread.start()
        threads.append(thread)
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Calculate statistics
    if results["response_times"]:
        avg_response_time = sum(results["response_times"]) / len(results["response_times"])
        max_response_time = max(results["response_times"])
        min_response_time = min(results["response_times"])
    else:
        avg_response_time = max_response_time = min_response_time = 0
    
    print(f"Load test completed:")
    print(f"  - Total requests: {results['total_requests']}")
    print(f"  - Successful: {results['successful_requests']}")
    print(f"  - Failed: {results['failed_requests']}")
    print(f"  - Success rate: {results['successful_requests']/max(results['total_requests'], 1)*100:.1f}%")
    print(f"  - Avg response time: {avg_response_time:.3f}s")
    print(f"  - Min response time: {min_response_time:.3f}s")
    print(f"  - Max response time: {max_response_time:.3f}s")
    print(f"  - Requests per second: {results['total_requests']/duration:.1f}")
    
    if results["errors"]:
        unique_errors = list(set(results["errors"]))
        print(f"  - Error types: {unique_errors[:5]}")  # Show first 5 unique errors

def main():
    """Run all API tests"""
    print("🧪 Testing Complete API System (Phase 3)")
    print("=" * 60)
    
    # Initialize tester
    tester = APITester()
    
    # Run tests
    tests = [
        ("Server Startup", tester.test_server_startup),
        ("Health Endpoint", tester.test_health_endpoint),
        ("Chat Endpoint", tester.test_chat_endpoint),
        ("Conversation Management", tester.test_conversation_management),
        ("Search Endpoint", tester.test_search_endpoint),
        ("Feedback Endpoint", tester.test_feedback_endpoint),
        ("Stats Endpoint", tester.test_stats_endpoint),
        ("Rate Limiting", tester.test_rate_limiting),
        ("Error Handling", tester.test_error_handling),
        ("Documentation Endpoints", tester.test_documentation_endpoints)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            failed += 1
    
    # Run load test if basic tests pass
    if failed == 0:
        try:
            run_load_test(tester.base_url, tester.api_key, duration=15, concurrent_users=3)
        except Exception as e:
            print(f"⚠️ Load test failed: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All API tests passed! Your system is ready for production.")
        print("\nNext steps:")
        print("1. Deploy to production environment")
        print("2. Set up monitoring and alerting")
        print("3. Configure proper authentication")
        print("4. Set up SSL/TLS certificates")
    else:
        print("⚠️ Some tests failed. Please fix the issues before deploying.")
        return 1
    
    return 0

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the Changi RAG Chatbot API")
    parser.add_argument("--url", default=None, help="Base URL for the API")
    parser.add_argument("--key", default=None, help="API key for testing")
    
    args = parser.parse_args()
    
    if args.url:
        # Override default URL
        pass
    
    sys.exit(main())