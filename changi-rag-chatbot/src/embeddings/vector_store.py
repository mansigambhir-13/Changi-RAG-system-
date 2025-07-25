"""
FIXED Vector store implementation using Qdrant Cloud for RAG system
Fixed to use UUID IDs instead of string IDs
"""
import os
import sys
import time
import json
import uuid
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass

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
    from qdrant_client.models import Distance, VectorParams, PointStruct
    from qdrant_client.http import models
except ImportError:
    print("❌ Qdrant client not installed!")
    print("📦 Install with: pip install qdrant-client")
    sys.exit(1)

# Simple performance logging decorator
def log_performance(operation_name: str):
    """Simple performance logging decorator"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                print(f"✅ {operation_name} completed in {duration:.2f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                print(f"❌ {operation_name} failed after {duration:.2f}s: {e}")
                raise
        return wrapper
    return decorator

@dataclass
class VectorSearchResult:
    """Data class for vector search results"""
    id: str
    score: float
    metadata: Dict[str, Any]
    values: Optional[List[float]] = None

class QdrantVectorStore:
    """Vector store implementation using Qdrant Cloud"""
    
    def __init__(self, collection_name: str = None, dimension: int = None):
        # Configuration from environment variables or defaults
        self.collection_name = collection_name or os.getenv('QDRANT_COLLECTION_NAME', 'changi-rag-collection')
        self.dimension = dimension or int(os.getenv('QDRANT_DIMENSION', '384'))
        
        # Qdrant configuration
        qdrant_url = os.getenv('QDRANT_URL', 'memory')
        qdrant_api_key = os.getenv('QDRANT_API_KEY', None)
        
        print(f"🚀 Initializing Qdrant Cloud Vector Store")
        print(f"📁 Collection: {self.collection_name}")
        print(f"📐 Vector dimension: {self.dimension}")
        print(f"🌐 Qdrant URL: {qdrant_url}")
        if qdrant_api_key:
            print(f"🔑 API Key: {qdrant_api_key[:20]}...")
        
        # Initialize Qdrant client
        try:
            if qdrant_url == 'memory':
                # In-memory mode (for testing)
                print("💾 Using in-memory Qdrant")
                self.client = QdrantClient(":memory:")
            elif qdrant_url.startswith('http'):
                # Cloud or Docker connection
                print(f"☁️  Connecting to Qdrant Cloud...")
                self.client = QdrantClient(
                    url=qdrant_url,
                    api_key=qdrant_api_key
                )
                
                # Test connection
                collections = self.client.get_collections()
                print(f"✅ Successfully connected to Qdrant Cloud!")
                print(f"📊 Found {len(collections.collections)} existing collections")
                
            else:
                # File-based persistence
                qdrant_path = qdrant_url or "./qdrant_data"
                print(f"💾 Using persistent Qdrant storage at {qdrant_path}")
                self.client = QdrantClient(path=qdrant_path)
        
        except Exception as e:
            print(f"❌ Failed to connect to Qdrant: {e}")
            print("🔧 Check your QDRANT_URL and QDRANT_API_KEY in .env file")
            raise
        
        # Initialize collection
        self._initialize_collection()
    
    def _initialize_collection(self):
        """Initialize or connect to Qdrant collection"""
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            existing_collections = [col.name for col in collections]
            
            if self.collection_name not in existing_collections:
                print(f"🆕 Creating new collection: {self.collection_name}")
                
                # Create collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.dimension,
                        distance=Distance.COSINE
                    )
                )
                print("✅ Collection created successfully")
            else:
                print(f"🔗 Collection '{self.collection_name}' already exists")
            
            # Get collection info
            info = self.client.get_collection(self.collection_name)
            print(f"📊 Collection info:")
            print(f"   📄 Vectors count: {info.points_count}")
            print(f"   📐 Vector size: {info.config.params.vectors.size}")
            print(f"   📏 Distance metric: {info.config.params.vectors.distance}")
            print(f"   📊 Status: {info.status}")
            
        except Exception as e:
            print(f"❌ Failed to initialize collection: {e}")
            raise
    
    def _generate_uuid_from_string(self, string_id: str) -> str:
        """Generate a consistent UUID from a string ID"""
        # Use UUID5 to generate consistent UUIDs from string IDs
        namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')  # Standard namespace
        return str(uuid.uuid5(namespace, string_id))
    
    @log_performance("upsert_vectors")
    def upsert_vectors(
        self,
        vectors: List[Tuple[str, List[float], Dict[str, Any]]],
        batch_size: int = 100
    ) -> bool:
        """
        Upsert vectors to Qdrant collection
        
        Args:
            vectors: List of (original_id, vector, metadata) tuples
            batch_size: Number of vectors to upsert in each batch
        
        Returns:
            bool: Success status
        """
        try:
            print(f"📤 Upserting {len(vectors)} vectors to Qdrant Cloud...")
            print("🔄 Converting string IDs to UUIDs...")
            
            # Process in batches
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                
                # Prepare batch data
                points = []
                for original_id, vector, metadata in batch:
                    # Generate UUID from original string ID
                    uuid_id = self._generate_uuid_from_string(original_id)
                    
                    # Store original ID in metadata for searching
                    metadata['original_id'] = original_id
                    metadata['uuid_id'] = uuid_id
                    
                    point = PointStruct(
                        id=uuid_id,  # Use UUID instead of string
                        vector=vector,
                        payload=metadata
                    )
                    points.append(point)
                
                # Upsert batch
                result = self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                
                batch_num = i//batch_size + 1
                total_batches = (len(vectors)-1)//batch_size + 1
                print(f"   ✅ Batch {batch_num}/{total_batches} - Status: {result.status}")
            
            print("🎉 Vector upsert completed successfully")
            
            # Verify upload
            info = self.client.get_collection(self.collection_name)
            print(f"📊 Collection now has {info.points_count} vectors")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to upsert vectors: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    @log_performance("similarity_search")
    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
        include_values: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search in the vector store
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filter_dict: Metadata filters (Qdrant filter format)
            include_metadata: Whether to include metadata
            include_values: Whether to include vector values
        
        Returns:
            List of search results
        """
        try:
            print(f"🔍 Performing similarity search (top_k={top_k})")
            
            # Convert filter_dict to Qdrant filter if provided
            search_filter = None
            if filter_dict:
                # Simple implementation - can be expanded for complex filters
                must_conditions = []
                for key, value in filter_dict.items():
                    must_conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value)
                        )
                    )
                if must_conditions:
                    search_filter = models.Filter(must=must_conditions)
            
            # Perform search
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=search_filter,
                limit=top_k,
                with_payload=include_metadata,
                with_vectors=include_values
            )
            
            # Convert to standard format
            results = []
            for hit in search_results:
                result = {
                    'id': hit.payload.get('original_id', str(hit.id)) if hit.payload else str(hit.id),  # Return original ID
                    'score': float(hit.score),
                    'metadata': hit.payload if include_metadata and hit.payload else {}
                }
                
                if include_values and hit.vector:
                    result['values'] = hit.vector
                
                results.append(result)
            
            print(f"📋 Found {len(results)} results")
            if results:
                print(f"   🏆 Best score: {results[0]['score']:.3f}")
                print(f"   📉 Worst score: {results[-1]['score']:.3f}")
            
            return results
            
        except Exception as e:
            print(f"❌ Similarity search failed: {e}")
            return []
    
    def delete_vectors(self, original_ids: List[str]) -> bool:
        """Delete vectors by their original string IDs"""
        try:
            # Convert original IDs to UUIDs
            uuid_ids = [self._generate_uuid_from_string(original_id) for original_id in original_ids]
            
            result = self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=uuid_ids
                )
            )
            print(f"🗑️  Deleted {len(original_ids)} vectors - Status: {result.status}")
            return True
        except Exception as e:
            print(f"❌ Failed to delete vectors: {e}")
            return False
    
    def clear_collection(self) -> bool:
        """Clear all vectors from the collection"""
        try:
            # Delete all points
            result = self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter()
                )
            )
            print(f"🧹 Collection cleared - Status: {result.status}")
            return True
        except Exception as e:
            print(f"❌ Failed to clear collection: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                'total_vector_count': info.points_count,
                'dimension': info.config.params.vectors.size,
                'distance_metric': str(info.config.params.vectors.distance),
                'status': str(info.status),
                'indexed_vectors_count': info.indexed_vectors_count if hasattr(info, 'indexed_vectors_count') else info.points_count
            }
        except Exception as e:
            print(f"❌ Failed to get collection stats: {e}")
            return {}
    
    def get_vector(self, original_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific vector by its original string ID"""
        try:
            # Convert original ID to UUID
            uuid_id = self._generate_uuid_from_string(original_id)
            
            result = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[uuid_id],
                with_payload=True,
                with_vectors=True
            )
            
            if result:
                point = result[0]
                return {
                    'id': point.payload.get('original_id', original_id),
                    'values': point.vector,
                    'metadata': point.payload
                }
            return None
        except Exception as e:
            print(f"❌ Failed to get vector {original_id}: {e}")
            return None
    
    def test_connection(self) -> bool:
        """Test the Qdrant connection"""
        try:
            collections = self.client.get_collections()
            print(f"✅ Connection test successful!")
            print(f"📊 Available collections: {[c.name for c in collections.collections]}")
            return True
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False

def upload_embeddings_from_file():
    """Upload embeddings from file to Qdrant Cloud"""
    print("🚀 Starting embeddings upload to Qdrant Cloud...")
    print("=" * 70)
    
    try:
        # Initialize vector store
        print("🔧 Initializing Qdrant Cloud vector store...")
        vector_store = QdrantVectorStore()
        
        # Test connection first
        if not vector_store.test_connection():
            print("❌ Connection failed. Check your credentials.")
            return
        
        # Load embeddings from file
        embeddings_path = "data/embeddings/document_embeddings.json"
        
        if not os.path.exists(embeddings_path):
            print(f"❌ Embeddings file not found: {embeddings_path}")
            print("\n📝 To generate embeddings, run:")
            print("   python src/embeddings/embedding_generator.py")
            return
        
        print(f"📖 Loading embeddings from {embeddings_path}")
        with open(embeddings_path, 'r', encoding='utf-8') as f:
            embeddings_data = json.load(f)
        
        print(f"📄 Loaded {len(embeddings_data)} embeddings")
        
        # Prepare vectors for upload
        vectors = []
        for item in embeddings_data:
            original_id = item['chunk_id']  # Keep original string ID
            embedding = item['embedding']
            metadata = {
                'content_preview': item['content'][:500],  # Store first 500 chars for preview
                'title': item.get('title', ''),
                'url': item.get('url', ''),
                'chunk_index': item.get('chunk_index', 0),
                'content_hash': item.get('content_hash', ''),
                'chunk_id': original_id,  # Store original ID in metadata
                'full_content': item['content'],  # Store full content
                'content_length': len(item['content']),
                'source': 'changi_airport'
            }
            
            vectors.append((original_id, embedding, metadata))
        
        # Upload vectors
        success = vector_store.upsert_vectors(vectors)
        
        if success:
            print("\n🎉 SUCCESS! Embeddings uploaded to Qdrant Cloud")
            
            # Get and display stats
            stats = vector_store.get_collection_stats()
            print(f"📊 Collection Statistics:")
            print(f"   📄 Total vectors: {stats.get('total_vector_count', 0)}")
            print(f"   📐 Vector dimension: {stats.get('dimension', 'N/A')}")
            print(f"   📏 Distance metric: {stats.get('distance_metric', 'N/A')}")
            print(f"   ✅ Status: {stats.get('status', 'N/A')}")
            
            # Test search with actual data
            print("\n🧪 Testing search functionality...")
            if embeddings_data:
                # Use the first embedding for testing
                test_embedding = embeddings_data[0]['embedding']
                results = vector_store.similarity_search(test_embedding, top_k=3)
                
                print(f"✅ Search test returned {len(results)} results")
                
                if results:
                    print("\n📋 Sample search results:")
                    for i, result in enumerate(results[:2]):
                        print(f"   {i+1}. ID: {result['id']}")
                        print(f"      📝 Title: {result['metadata'].get('title', 'N/A')}")
                        print(f"      📊 Score: {result['score']:.3f}")
                        print(f"      📄 Content: {result['metadata'].get('content_preview', '')[:80]}...")
                        print(f"      🔗 URL: {result['metadata'].get('url', 'N/A')}")
            
            print("\n🎯 Next steps:")
            print("   ✅ Embeddings are uploaded to Qdrant Cloud")
            print("   🔍 Test retriever: python src/rag/retriever.py")
            print("   🚀 Start API server: python src/api/main.py")
            
        else:
            print("\n❌ Failed to upload embeddings")
            
    except Exception as e:
        print(f"\n❌ Error uploading embeddings: {e}")
        import traceback
        traceback.print_exc()

def test_vector_store():
    """Test vector store functionality with Qdrant Cloud"""
    print("🧪 Testing Qdrant Cloud Vector Store")
    print("=" * 60)
    
    try:
        # Initialize
        print("🔧 Initializing vector store...")
        vector_store = QdrantVectorStore()
        
        # Test connection
        if not vector_store.test_connection():
            return
        
        # Test data
        test_vectors = [
            (
                "test_001",  # Original string ID 
                [0.1] * vector_store.dimension,
                {"title": "Test Document 1", "content": "This is a test document about airports", "test": True}
            ),
            (
                "test_002",  # Original string ID
                [0.2] * vector_store.dimension,
                {"title": "Test Document 2", "content": "This is another test document about travel", "test": True}
            )
        ]
        
        # Test upsert
        print("\n📤 Testing vector upsert...")
        success = vector_store.upsert_vectors(test_vectors)
        print(f"   Upsert success: {success}")
        
        # Test search
        print("\n🔍 Testing similarity search...")
        query_vector = [0.15] * vector_store.dimension
        results = vector_store.similarity_search(query_vector, top_k=2)
        print(f"   Found {len(results)} results")
        
        for i, result in enumerate(results):
            print(f"   {i+1}. ID: {result['id']}, Score: {result['score']:.3f}")
            print(f"       Title: {result['metadata'].get('title', 'N/A')}")
        
        # Test filtered search
        print("\n🔍 Testing filtered search...")
        filter_results = vector_store.similarity_search(
            query_vector, 
            top_k=2, 
            filter_dict={"test": True}
        )
        print(f"   Found {len(filter_results)} filtered results")
        
        # Test individual vector retrieval
        print("\n🔍 Testing individual vector retrieval...")
        vector = vector_store.get_vector("test_001")
        if vector:
            print(f"   Retrieved vector: {vector['id']}")
            print(f"   Metadata: {vector['metadata']['title']}")
        
        # Test stats
        print("\n📊 Collection statistics:")
        stats = vector_store.get_collection_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        # Clean up test vectors
        print("\n🧹 Cleaning up test vectors...")
        vector_store.delete_vectors(["test_001", "test_002"])
        
        print("\n✅ Vector store test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Vector store test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function to handle command line arguments"""
    import sys
    
    print("🤖 Changi RAG Chatbot - Qdrant Cloud Vector Store (Fixed)")
    print("=" * 70)
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        if command == "test":
            test_vector_store()
        elif command == "upload":
            upload_embeddings_from_file()
        elif command == "stats":
            try:
                vector_store = QdrantVectorStore()
                stats = vector_store.get_collection_stats()
                print("📊 Current Collection Statistics:")
                for key, value in stats.items():
                    print(f"   {key}: {value}")
            except Exception as e:
                print(f"❌ Failed to get stats: {e}")
        elif command == "connection":
            try:
                vector_store = QdrantVectorStore()
                vector_store.test_connection()
            except Exception as e:
                print(f"❌ Connection test failed: {e}")
        elif command == "clear":
            try:
                vector_store = QdrantVectorStore()
                vector_store.clear_collection()
                print("✅ Collection cleared successfully")
            except Exception as e:
                print(f"❌ Failed to clear collection: {e}")
        else:
            print(f"❌ Unknown command: {command}")
            print("\nAvailable commands:")
            print("   python vector_store.py upload      - Upload embeddings to Qdrant")
            print("   python vector_store.py test        - Test vector store functionality")
            print("   python vector_store.py stats       - Show collection statistics")
            print("   python vector_store.py connection  - Test Qdrant connection")
            print("   python vector_store.py clear       - Clear all vectors")
    else:
        # Default action: upload embeddings
        upload_embeddings_from_file()

if __name__ == "__main__":
    main()