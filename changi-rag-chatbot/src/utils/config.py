# src/utils/config.py
import os
from typing import List, Optional
from functools import lru_cache

# Load environment variables from .env file
def load_env_file():
    """Load environment variables from .env file"""
    env_path = ".env"
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

# Load .env file on import
load_env_file()

class Settings:
    """Simple settings class compatible with existing structure"""
    
    # API Configuration
    API_TOKEN = os.getenv('API_TOKEN', 'your-secret-token-here')
    ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', '["*"]')
    
    # Google Gemini Configuration (Primary LLM)
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')
    GEMINI_TEMPERATURE = float(os.getenv('GEMINI_TEMPERATURE', '0.1'))
    GEMINI_MAX_TOKENS = int(os.getenv('GEMINI_MAX_TOKENS', '1000'))
    
    # OpenAI Configuration (Fallback)
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
    MAX_RESPONSE_TOKENS = int(os.getenv('MAX_RESPONSE_TOKENS', '500'))
    RESPONSE_TEMPERATURE = float(os.getenv('RESPONSE_TEMPERATURE', '0.7'))
    
    # Qdrant Vector Database Configuration
    QDRANT_URL = os.getenv('QDRANT_URL', 'http://localhost:6333')
    QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
    QDRANT_COLLECTION_NAME = os.getenv('QDRANT_COLLECTION_NAME', 'changi_rag_collection')
    QDRANT_GRPC_PORT = int(os.getenv('QDRANT_GRPC_PORT', '6334'))
    QDRANT_PREFER_GRPC = os.getenv('QDRANT_PREFER_GRPC', 'false').lower() == 'true'
    QDRANT_DIMENSION = int(os.getenv('QDRANT_DIMENSION', '384'))
    
    # Embedding Configuration
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    EMBEDDING_DIMENSION = int(os.getenv('EMBEDDING_DIMENSION', '384'))
    EMBEDDING_BATCH_SIZE = int(os.getenv('EMBEDDING_BATCH_SIZE', '32'))
    
    # Chunking Configuration
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '800'))
    MAX_CHUNK_SIZE = int(os.getenv('MAX_CHUNK_SIZE', '1500'))
    MIN_CHUNK_SIZE = int(os.getenv('MIN_CHUNK_SIZE', '200'))
    CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '50'))
    TARGET_CHUNK_SIZE = int(os.getenv('TARGET_CHUNK_SIZE', '800'))
    
    # Retrieval Configuration
    RETRIEVAL_TOP_K = int(os.getenv('RETRIEVAL_TOP_K', '5'))
    VECTOR_SEARCH_LIMIT = int(os.getenv('VECTOR_SEARCH_LIMIT', '5'))
    VECTOR_SEARCH_SCORE_THRESHOLD = float(os.getenv('VECTOR_SEARCH_SCORE_THRESHOLD', '0.3'))
    MIN_SCORE_THRESHOLD = float(os.getenv('MIN_SCORE_THRESHOLD', '0.3'))
    
    # Redis Configuration
    REDIS_URL = os.getenv('REDIS_URL')
    REDIS_TTL = int(os.getenv('REDIS_TTL', '3600'))
    
    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = os.getenv('LOG_FORMAT', 'json')
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS = int(os.getenv('RATE_LIMIT_REQUESTS', '100'))
    RATE_LIMIT_WINDOW = int(os.getenv('RATE_LIMIT_WINDOW', '3600'))
    
    # Data Processing Paths
    RAW_DATA_DIR = os.getenv('RAW_DATA_DIR', 'data/raw')
    PROCESSED_DATA_DIR = os.getenv('PROCESSED_DATA_DIR', 'data/processed')
    EMBEDDINGS_DATA_DIR = os.getenv('EMBEDDINGS_DATA_DIR', 'data/embeddings')
    QDRANT_SNAPSHOTS_DIR = os.getenv('QDRANT_SNAPSHOTS_DIR', 'data/qdrant_snapshots')
    
    # File Processing
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', '10000'))
    MIN_CONTENT_LENGTH = int(os.getenv('MIN_CONTENT_LENGTH', '50'))
    
    # Conversation Settings
    MAX_CONVERSATION_HISTORY = int(os.getenv('MAX_CONVERSATION_HISTORY', '10'))
    CONVERSATION_TIMEOUT = int(os.getenv('CONVERSATION_TIMEOUT', '86400'))
    
    # Cache Settings
    ENABLE_RESPONSE_CACHE = os.getenv('ENABLE_RESPONSE_CACHE', 'true').lower() == 'true'
    CACHE_MAX_SIZE = int(os.getenv('CACHE_MAX_SIZE', '100'))
    CACHE_TTL = int(os.getenv('CACHE_TTL', '3600'))
    
    @property
    def effective_gemini_api_key(self):
        """Get the effective Gemini API key (prefer GEMINI_API_KEY over GOOGLE_API_KEY)"""
        return self.GEMINI_API_KEY or self.GOOGLE_API_KEY

# Create global settings instance
settings = Settings()

@lru_cache()
def get_settings():
    """Get settings instance for compatibility"""
    return settings

def validate_environment():
    """Validate the environment setup"""
    issues = []
    
    # Check Gemini API key
    if not settings.effective_gemini_api_key:
        issues.append("❌ GEMINI_API_KEY (or GOOGLE_API_KEY) not set")
    
    # Check Qdrant connection
    try:
        import requests
        response = requests.get(f"{settings.QDRANT_URL}/health", timeout=5)
        if response.status_code != 200:
            issues.append(f"❌ Qdrant not healthy at {settings.QDRANT_URL}")
    except Exception as e:
        issues.append(f"❌ Cannot connect to Qdrant: {e}")
    
    # Check required directories
    import pathlib
    for dir_path in [settings.RAW_DATA_DIR, settings.PROCESSED_DATA_DIR]:
        if not pathlib.Path(dir_path).exists():
            issues.append(f"⚠️  Directory missing: {dir_path}")
    
    if issues:
        print("🔧 Environment Issues Found:")
        for issue in issues:
            print(f"   {issue}")
        return False
    else:
        print("✅ Environment validation passed!")
        return True

def create_env_template():
    """Create comprehensive .env template file"""
    template = """# Changi Airport RAG System - Complete Configuration

# API Security
API_TOKEN=your-secure-api-token-here
ALLOWED_ORIGINS=["http://localhost:3000", "http://localhost:8080"]

# Google Gemini AI (Primary LLM)
GEMINI_API_KEY=your-gemini-api-key-here
GEMINI_MODEL=gemini-1.5-flash
GEMINI_TEMPERATURE=0.1
GEMINI_MAX_TOKENS=1000

# Google API (Alternative naming - for backward compatibility)
GOOGLE_API_KEY=your-gemini-api-key-here

# OpenAI (Optional fallback)
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_MODEL=gpt-3.5-turbo
MAX_RESPONSE_TOKENS=500
RESPONSE_TEMPERATURE=0.7

# Qdrant Vector Database
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION_NAME=changi_rag_collection
QDRANT_GRPC_PORT=6334
QDRANT_PREFER_GRPC=false
QDRANT_DIMENSION=384

# Embedding Model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
EMBEDDING_BATCH_SIZE=32

# Content Chunking
CHUNK_SIZE=800
MAX_CHUNK_SIZE=1500
MIN_CHUNK_SIZE=200
CHUNK_OVERLAP=50
TARGET_CHUNK_SIZE=800

# Vector Search & Retrieval
RETRIEVAL_TOP_K=5
VECTOR_SEARCH_LIMIT=5
VECTOR_SEARCH_SCORE_THRESHOLD=0.3
MIN_SCORE_THRESHOLD=0.3

# Redis (Optional - for conversation memory)
REDIS_URL=redis://localhost:6379/0
REDIS_TTL=3600

# Data Directories
RAW_DATA_DIR=data/raw
PROCESSED_DATA_DIR=data/processed
EMBEDDINGS_DATA_DIR=data/embeddings
QDRANT_SNAPSHOTS_DIR=data/qdrant_snapshots

# Processing Limits
MAX_CONTENT_LENGTH=10000
MIN_CONTENT_LENGTH=50

# Conversation & Caching
MAX_CONVERSATION_HISTORY=10
CONVERSATION_TIMEOUT=86400
ENABLE_RESPONSE_CACHE=true
CACHE_MAX_SIZE=100
CACHE_TTL=3600

# Logging & Monitoring
LOG_LEVEL=INFO
LOG_FORMAT=json

# Rate Limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=3600
"""
    
    with open(".env.template", "w") as f:
        f.write(template)
    
    print("✅ Environment template created: .env.template")
    print("📝 Copy to .env and configure your API keys")

if __name__ == "__main__":
    print("🔧 Changi RAG Configuration")
    print("=" * 40)
    
    print(f"📝 Current configuration:")
    print(f"   🤖 Gemini API Key: {'✅ Set' if settings.effective_gemini_api_key else '❌ Missing'}")
    print(f"   🗄️  Qdrant URL: {settings.QDRANT_URL}")
    print(f"   📊 Collection: {settings.QDRANT_COLLECTION_NAME}")
    print(f"   🧠 Embedding Model: {settings.EMBEDDING_MODEL}")
    print(f"   📏 Vector Dimension: {settings.EMBEDDING_DIMENSION}")
    
    print("\n🧪 Validating environment...")
    validate_environment()
    
    print("\n💡 To create environment template:")
    print("   python src/utils/config.py create_template")
    
    if len(__import__('sys').argv) > 1 and __import__('sys').argv[1] == 'create_template':
        create_env_template()