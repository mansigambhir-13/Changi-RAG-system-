# setup_integrated_system.py
"""
Complete setup script for the integrated Changi Airport RAG system
This will set up all components to work together seamlessly
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
import json

def create_directory_structure():
    """Create the complete directory structure"""
    directories = [
        "src/scraper",
        "src/embeddings", 
        "src/rag",
        "src/api",
        "src/utils",
        "data/raw",
        "data/processed", 
        "data/embeddings",
        "data/qdrant_snapshots",
        "docker",
        "deployment",
        "tests",
        "docs",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py files for Python packages
        if directory.startswith("src/"):
            init_file = Path(directory) / "__init__.py"
            if not init_file.exists():
                init_file.touch()
    
    print("✅ Directory structure created")

def create_requirements_file():
    """Create comprehensive requirements.txt"""
    requirements = """# Core Dependencies
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0

# Qdrant Vector Database
qdrant-client>=1.7.0

# AI and Language Models
google-generativeai>=0.3.2
sentence-transformers>=2.2.2

# Optional: OpenAI (fallback)
openai==1.3.0

# Data Processing
numpy>=1.24.3
pandas>=2.0.3
scikit-learn>=1.3.0
beautifulsoup4>=4.12.2
scrapy>=2.11.0

# API and Web
httpx>=0.25.0
requests>=2.31.0

# Database and Caching
redis>=5.0.1

# Utilities
python-dotenv>=1.0.0
python-multipart>=0.0.6

# Development and Testing
pytest>=7.4.3
pytest-asyncio>=0.21.1
black>=23.11.0

# Additional utilities
tqdm>=4.66.1
click>=8.1.7
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements)
    
    print("✅ Requirements file created")

def create_env_template():
    """Create comprehensive .env template"""
    template = """# Changi Airport RAG System - Integrated Configuration

# Google Gemini AI (Primary LLM) - REQUIRED
GEMINI_API_KEY=your-gemini-api-key-here
GOOGLE_API_KEY=your-gemini-api-key-here
GEMINI_MODEL=gemini-1.5-flash
GEMINI_TEMPERATURE=0.1
GEMINI_MAX_TOKENS=1000

# Qdrant Vector Database - REQUIRED
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=changi_rag_collection
QDRANT_DIMENSION=384

# Embedding Model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

# Content Processing
CHUNK_SIZE=800
MAX_CHUNK_SIZE=1500
MIN_CHUNK_SIZE=200
TARGET_CHUNK_SIZE=800

# Retrieval Settings
RETRIEVAL_TOP_K=5
MIN_SCORE_THRESHOLD=0.3

# API Configuration
API_TOKEN=your-secure-api-token-here

# Optional: Redis for conversation memory
REDIS_URL=redis://localhost:6379/0

# Optional: OpenAI (fallback)
OPENAI_API_KEY=your-openai-api-key-here

# Data Directories
RAW_DATA_DIR=data/raw
PROCESSED_DATA_DIR=data/processed
EMBEDDINGS_DATA_DIR=data/embeddings

# System Settings
LOG_LEVEL=INFO
ENABLE_RESPONSE_CACHE=true
CACHE_MAX_SIZE=100
MAX_CONVERSATION_HISTORY=10
"""
    
    with open(".env.example", "w") as f:
        f.write(template)
    
    print("✅ Environment template created (.env.example)")

def create_docker_compose():
    """Create docker-compose.yml for Qdrant and Redis"""
    docker_compose = """version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:latest
    container_name: changi_qdrant
    ports:
      - "6333:6333"  # REST API
      - "6334:6334"  # GRPC API
    volumes:
      - qdrant_storage:/qdrant/storage
      - ./data/qdrant_snapshots:/qdrant/snapshots
    environment:
      - QDRANT__SERVICE__HTTP_PORT=6333
      - QDRANT__SERVICE__GRPC_PORT=6334
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    container_name: changi_redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    command: redis-server --appendonly yes

volumes:
  qdrant_storage:
  redis_data:
"""
    
    with open("docker-compose.yml", "w") as f:
        f.write(docker_compose)
    
    print("✅ Docker Compose file created")

def create_startup_scripts():
    """Create startup scripts"""
    
    # Start services script
    start_services = '''#!/bin/bash
# start_services.sh - Start Qdrant and Redis services

echo "🚀 Starting Changi RAG Services..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Start services
echo "📦 Starting Qdrant and Redis..."
docker-compose up -d

# Wait for services
echo "⏳ Waiting for services to start..."
sleep 10

# Check Qdrant health
echo "🔍 Checking Qdrant health..."
if curl -f http://localhost:6333/health > /dev/null 2>&1; then
    echo "✅ Qdrant is running and healthy"
else
    echo "❌ Qdrant health check failed"
    exit 1
fi

# Check Redis health
echo "🔍 Checking Redis health..."
if redis-cli ping | grep -q PONG 2>/dev/null; then
    echo "✅ Redis is running and healthy"
else
    echo "⚠️  Redis check failed (optional service)"
fi

echo "🎉 Services started successfully!"
echo "📝 Qdrant Dashboard: http://localhost:6333/dashboard"
echo "📝 You can now run your RAG system"
'''
    
    with open("start_services.sh", "w") as f:
        f.write(start_services)
    os.chmod("start_services.sh", 0o755)
    
    # Run API script
    run_api = '''#!/bin/bash
# run_api.sh - Start the RAG API server

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ .env file not found. Please copy .env.example to .env and configure it."
    exit 1
fi

# Check if services are running
if ! curl -f http://localhost:6333/health > /dev/null 2>&1; then
    echo "❌ Qdrant is not running. Start services first: ./start_services.sh"
    exit 1
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the API server
echo "🚀 Starting Changi Airport RAG API..."
echo "📝 API Docs: http://localhost:8000/docs"
echo "📝 Qdrant Dashboard: http://localhost:6333/dashboard"
echo ""
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
'''
    
    with open("run_api.sh", "w") as f:
        f.write(run_api)
    os.chmod("run_api.sh", 0o755)
    
    print("✅ Startup scripts created")

def create_test_script():
    """Create comprehensive test script"""
    test_script = '''#!/usr/bin/env python3
# test_system.py - Test the complete integrated system

import sys
import os
import time
import requests

def test_environment():
    """Test environment setup"""
    print("🧪 Testing Environment Setup...")
    
    # Test Python imports
    try:
        from src.utils.config import get_settings, validate_environment
        from src.utils.logger import get_logger
        print("✅ Configuration and logging imports successful")
    except ImportError as e:
        print(f"❌ Configuration import failed: {e}")
        return False
    
    # Test environment validation
    try:
        validate_environment()
        print("✅ Environment validation completed")
    except Exception as e:
        print(f"❌ Environment validation failed: {e}")
        return False
    
    return True

def test_services():
    """Test external services"""
    print("\\n🔍 Testing External Services...")
    
    # Test Qdrant
    try:
        response = requests.get("http://localhost:6333/health", timeout=5)
        if response.status_code == 200:
            print("✅ Qdrant is healthy")
        else:
            print("❌ Qdrant is not healthy")
            return False
    except requests.exceptions.RequestException:
        print("❌ Cannot connect to Qdrant")
        print("💡 Start services with: ./start_services.sh")
        return False
    
    # Test Redis (optional)
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping()
        print("✅ Redis is healthy")
    except Exception:
        print("⚠️  Redis not available (optional)")
    
    return True

def test_components():
    """Test individual components"""
    print("\\n🔧 Testing Components...")
    
    # Test retriever
    try:
        from src.rag.retriever import RAGRetriever
        retriever = RAGRetriever()
        print("✅ Retriever initialized successfully")
    except Exception as e:
        print(f"❌ Retriever failed: {e}")
        return False
    
    # Test generator
    try:
        from src.rag.generator import GeminiRAGGenerator
        generator = GeminiRAGGenerator()
        print("✅ Generator initialized successfully")
    except Exception as e:
        print(f"❌ Generator failed: {e}")
        return False
    
    # Test pipeline
    try:
        from src.rag.pipeline import ChangiRAGPipeline
        pipeline = ChangiRAGPipeline()
        print("✅ Pipeline initialized successfully")
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        return False
    
    return True

def test_end_to_end():
    """Test complete end-to-end functionality"""
    print("\\n🎯 Testing End-to-End Functionality...")
    
    try:
        from src.rag.pipeline import ChangiRAGPipeline
        
        pipeline = ChangiRAGPipeline()
        
        # Test query
        response = pipeline.chat(
            message="What are the operating hours of Changi Airport?",
            conversation_id="test_001",
            user_id="test_user"
        )
        
        print(f"✅ End-to-end test successful!")
        print(f"   Response: {response.response[:100]}...")
        print(f"   Confidence: {response.confidence:.3f}")
        print(f"   Sources: {len(response.sources)}")
        
        return True
        
    except Exception as e:
        print(f"❌ End-to-end test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Changi RAG System Integration Test")
    print("=" * 50)
    
    tests = [
        ("Environment", test_environment),
        ("External Services", test_services),
        ("Components", test_components),
        ("End-to-End", test_end_to_end)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\\n🔍 Running {test_name} test...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} test failed")
    
    print(f"\\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your system is ready!")
    else:
        print("⚠️  Some tests failed. Please check the setup.")
        
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
'''
    
    with open("test_system.py", "w") as f:
        f.write(test_script)
    os.chmod("test_system.py", 0o755)
    
    print("✅ Test script created")

def install_dependencies():
    """Install Python dependencies"""
    try:
        print("📦 Installing dependencies...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def create_component_templates():
    """Create templates for missing components"""
    
    # Create empty __init__.py files
    init_files = [
        "src/__init__.py",
        "src/scraper/__init__.py",
        "src/embeddings/__init__.py",
        "src/rag/__init__.py",
        "src/api/__init__.py",
        "src/utils/__init__.py"
    ]
    
    for init_file in init_files:
        Path(init_file).touch()
    
    print("✅ Component templates created")

def main():
    """Main setup function"""
    print("🔧 Setting up Integrated Changi Airport RAG System")
    print("=" * 60)
    print()
    
    # Create directory structure
    print("1. Creating directory structure...")
    create_directory_structure()
    
    # Create component templates
    print("\n2. Creating component templates...")
    create_component_templates()
    
    # Create requirements file
    print("\n3. Creating requirements file...")
    create_requirements_file()
    
    # Create environment template
    print("\n4. Creating environment template...")
    create_env_template()
    
    # Create Docker setup
    print("\n5. Creating Docker configuration...")
    create_docker_compose()
    
    # Create startup scripts
    print("\n6. Creating startup scripts...")
    create_startup_scripts()
    
    # Create test script
    print("\n7. Creating test script...")
    create_test_script()
    
    # Install dependencies
    print("\n8. Installing dependencies...")
    if not install_dependencies():
        print("❌ Setup failed during dependency installation")
        return
    
    print("\n" + "=" * 60)
    print("🎉 Integrated System Setup Completed Successfully!")
    print("=" * 60)
    print()
    print("📝 Next Steps:")
    print("1. Configure your environment:")
    print("   cp .env.example .env")
    print("   # Edit .env with your API keys")
    print()
    print("2. Start external services:")
    print("   ./start_services.sh")
    print()
    print("3. Copy your integrated components:")
    print("   # Copy the integrated config.py to src/utils/config.py")
    print("   # Copy the integrated logger.py to src/utils/logger.py")
    print("   # Copy the integrated retriever.py to src/rag/retriever.py")
    print("   # Copy the integrated generator.py to src/rag/generator.py")
    print("   # Copy the integrated pipeline.py to src/rag/pipeline.py")
    print()
    print("4. Test the system:")
    print("   python test_system.py")
    print()
    print("5. Run the API:")
    print("   ./run_api.sh")
    print()
    print("🌐 Once running, access:")
    print("   - API Docs: http://localhost:8000/docs")
    print("   - Qdrant Dashboard: http://localhost:6333/dashboard")

if __name__ == "__main__":
    main()