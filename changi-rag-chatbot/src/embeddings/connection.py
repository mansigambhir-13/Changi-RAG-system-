"""
Quick test to verify your Qdrant Cloud connection works
"""
import os

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

# Load .env file
load_env_file()

try:
    from qdrant_client import QdrantClient
    print("✅ Qdrant client imported successfully")
except ImportError:
    print("❌ Qdrant client not installed!")
    print("📦 Install with: pip install qdrant-client")
    exit(1)

def test_qdrant_connection():
    """Test your Qdrant Cloud connection"""
    print("🧪 Testing Qdrant Cloud Connection")
    print("=" * 50)
    
    # Get credentials from environment
    qdrant_url = os.getenv('QDRANT_URL')
    qdrant_api_key = os.getenv('QDRANT_API_KEY')
    
    print(f"🌐 URL: {qdrant_url}")
    if qdrant_api_key:
        print(f"🔑 API Key: {qdrant_api_key[:20]}...")
    else:
        print("❌ No API key found")
        return False
    
    try:
        # Initialize client
        print("\n🔧 Connecting to Qdrant Cloud...")
        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key
        )
        
        # Test connection by getting collections
        print("📋 Getting collections...")
        collections = client.get_collections()
        
        print(f"✅ Connection successful!")
        print(f"📊 Found {len(collections.collections)} collections:")
        for collection in collections.collections:
            print(f"   📁 {collection.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

if __name__ == "__main__":
    success = test_qdrant_connection()
    
    if success:
        print("\n🎉 Your Qdrant Cloud setup is working!")
        print("🎯 Next steps:")
        print("   1. python src/embeddings/embedding_generator.py")
        print("   2. python src/embeddings/vector_store.py")
    else:
        print("\n❌ Please check your Qdrant credentials in .env file")
        print("🔧 Make sure you have:")
        print("   QDRANT_URL=https://your-cluster.qdrant.tech:6333")
        print("   QDRANT_API_KEY=your-api-key")