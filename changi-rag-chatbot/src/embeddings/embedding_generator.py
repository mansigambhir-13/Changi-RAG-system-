"""
STANDALONE Embedding generator using Sentence Transformers
This file doesn't depend on any other project files - it's completely self-contained
"""
import os
import sys
import json
import numpy as np
from typing import List, Union, Dict, Any
import hashlib
import time

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
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("❌ sentence-transformers not installed!")
    print("📦 Install with: pip install sentence-transformers")
    sys.exit(1)

# Simple performance logging
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

class EmbeddingGenerator:
    """Generate embeddings using Sentence Transformers"""
    
    def __init__(self, model_name: str = None):
        # Get model from environment or use default
        self.model_name = model_name or os.getenv('SENTENCE_TRANSFORMER_MODEL', 'all-MiniLM-L6-v2')
        
        # Load model
        print(f"🤖 Loading embedding model: {self.model_name}")
        print("   This may take a few minutes on first run...")
        
        try:
            self.model = SentenceTransformer(self.model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            print(f"✅ Model loaded successfully!")
            print(f"📐 Embedding dimension: {self.dimension}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            print("🔧 Try installing with: pip install sentence-transformers torch")
            raise
    
    @log_performance("generate_embedding")
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        try:
            if not text or not text.strip():
                print("⚠️  Empty text provided for embedding")
                return np.zeros(self.dimension)
            
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
            
        except Exception as e:
            print(f"❌ Failed to generate embedding: {e}")
            return np.zeros(self.dimension)
    
    @log_performance("generate_embeddings_batch")
    def generate_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts"""
        try:
            if not texts:
                return []
            
            # Filter out empty texts
            valid_texts = [text for text in texts if text and text.strip()]
            
            if not valid_texts:
                print("⚠️  No valid texts for embedding generation")
                return [np.zeros(self.dimension) for _ in texts]
            
            print(f"🔄 Generating embeddings for {len(valid_texts)} texts...")
            embeddings = self.model.encode(valid_texts, convert_to_numpy=True, show_progress_bar=True)
            
            return [emb for emb in embeddings]
            
        except Exception as e:
            print(f"❌ Failed to generate batch embeddings: {e}")
            return [np.zeros(self.dimension) for _ in texts]

def create_sample_documents():
    """Create sample documents for testing"""
    sample_docs = [
        {
            "chunk_id": "changi_001",
            "content": "Changi Airport operates 24 hours a day, 7 days a week. The airport provides world-class facilities and services to passengers traveling through Singapore.",
            "title": "Changi Airport Operating Hours",
            "url": "https://www.changiairport.com/corporate/about-us/airport-operations",
            "chunk_index": 0,
            "content_hash": "abc123"
        },
        {
            "chunk_id": "jewel_001", 
            "content": "Jewel Changi Airport features the Rain Vortex, the world's tallest indoor waterfall at 40 meters high. The attraction also houses numerous dining and shopping options across multiple levels.",
            "title": "Jewel Changi Airport - Rain Vortex",
            "url": "https://www.jewelchangiairport.com/attractions/rain-vortex",
            "chunk_index": 1,
            "content_hash": "def456"
        },
        {
            "chunk_id": "wifi_001",
            "content": "Free WiFi is available throughout Changi Airport terminals. Passengers can connect to the 'Changi Airport WiFi' network for complimentary high-speed internet access without time limits.",
            "title": "Changi Airport WiFi Services",
            "url": "https://www.changiairport.com/passenger-guide/facilities-and-services/wifi",
            "chunk_index": 2,
            "content_hash": "ghi789"
        },
        {
            "chunk_id": "dining_001",
            "content": "Terminal 1 offers over 60 dining options including food courts, cafes, and restaurants. Popular choices include local Singapore cuisine like chicken rice and laksa, as well as international brands.",
            "title": "Terminal 1 Dining Options",
            "url": "https://www.changiairport.com/terminal1/dining",
            "chunk_index": 3,
            "content_hash": "jkl012"
        },
        {
            "chunk_id": "shopping_001",
            "content": "Shopping at Changi Airport includes duty-free stores, luxury brands, and local souvenirs. The airport features over 400 retail and service outlets across all terminals, making it a premier shopping destination.",
            "title": "Changi Airport Shopping",
            "url": "https://www.changiairport.com/shopping",
            "chunk_index": 4,
            "content_hash": "mno345"
        },
        {
            "chunk_id": "transport_001",
            "content": "Changi Airport is accessible by MRT, bus, taxi, and private car. The Changi Airport MRT station connects directly to the city center, with journey times of approximately 45 minutes to downtown Singapore.",
            "title": "Transportation to Changi Airport",
            "url": "https://www.changiairport.com/getting-here/by-train",
            "chunk_index": 5,
            "content_hash": "pqr678"
        },
        {
            "chunk_id": "terminal2_001",
            "content": "Terminal 2 at Changi Airport features the Orchid Garden, Entertainment Deck, and numerous shopping and dining options. The terminal serves as a hub for several international airlines.",
            "title": "Terminal 2 Facilities",
            "url": "https://www.changiairport.com/terminal2",
            "chunk_index": 6,
            "content_hash": "stu901"
        },
        {
            "chunk_id": "lounge_001",
            "content": "Changi Airport offers premium lounges across all terminals, including the award-winning Singapore Airlines SilverKris Lounge. Many lounges feature shower facilities, dining areas, and relaxation zones.",
            "title": "Airport Lounges",
            "url": "https://www.changiairport.com/passenger-guide/facilities-and-services/lounges",
            "chunk_index": 7,
            "content_hash": "vwx234"
        }
    ]
    
    return sample_docs

def process_documents_to_embeddings():
    """Process documents and generate embeddings"""
    print("🚀 Starting document embedding generation...")
    print("=" * 60)
    
    try:
        # Initialize embedding generator
        print("🔧 Initializing embedding generator...")
        embedding_gen = EmbeddingGenerator()
        
        # Load or create processed documents
        processed_docs_path = "data/processed/processed_documents.json"
        
        if not os.path.exists(processed_docs_path):
            print(f"📝 No processed documents found at {processed_docs_path}")
            print("📄 Creating sample documents for testing...")
            
            # Create sample documents
            sample_docs = create_sample_documents()
            
            # Create directory and save sample documents
            os.makedirs("data/processed", exist_ok=True)
            with open(processed_docs_path, 'w', encoding='utf-8') as f:
                json.dump(sample_docs, f, indent=2)
            
            print(f"✅ Created {len(sample_docs)} sample documents")
        
        print(f"📖 Loading documents from {processed_docs_path}")
        with open(processed_docs_path, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        print(f"📄 Loaded {len(documents)} document chunks")
        
        # Generate embeddings
        embeddings_data = []
        batch_size = 32
        
        total_batches = (len(documents) - 1) // batch_size + 1
        print(f"🔄 Processing {total_batches} batches...")
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_texts = [doc['content'] for doc in batch]
            
            batch_num = i//batch_size + 1
            print(f"\n📦 Processing batch {batch_num}/{total_batches}")
            
            # Generate embeddings
            batch_embeddings = embedding_gen.generate_embeddings_batch(batch_texts)
            
            # Combine with metadata
            for doc, embedding in zip(batch, batch_embeddings):
                embedding_item = {
                    'chunk_id': doc.get('chunk_id', f"chunk_{i}"),
                    'content': doc['content'],
                    'title': doc.get('title', ''),
                    'url': doc.get('url', ''),
                    'embedding': embedding.tolist(),
                    'chunk_index': doc.get('chunk_index', 0),
                    'content_hash': doc.get('content_hash', '')
                }
                embeddings_data.append(embedding_item)
        
        # Save embeddings
        os.makedirs("data/embeddings", exist_ok=True)
        embeddings_path = "data/embeddings/document_embeddings.json"
        
        print(f"\n💾 Saving embeddings to {embeddings_path}")
        with open(embeddings_path, 'w', encoding='utf-8') as f:
            json.dump(embeddings_data, f, indent=2)
        
        # Success summary
        print(f"\n🎉 SUCCESS! Embedding generation completed")
        print("=" * 60)
        print(f"📊 Generated embeddings for {len(embeddings_data)} document chunks")
        print(f"📐 Embedding dimension: {embedding_gen.dimension}")
        print(f"💾 Saved to: {embeddings_path}")
        print(f"📁 File size: {os.path.getsize(embeddings_path) / (1024*1024):.1f} MB")
        
        print(f"\n🎯 Next step: Upload to Pinecone with:")
        print(f"   python src/embeddings/vector_store.py")
        
    except Exception as e:
        print(f"\n❌ Error generating embeddings: {e}")
        import traceback
        traceback.print_exc()

def test_embedding_generator():
    """Test the embedding generator"""
    print("🧪 Testing Embedding Generator")
    print("=" * 50)
    
    try:
        # Initialize
        print("🔧 Initializing embedding generator...")
        embedding_gen = EmbeddingGenerator()
        
        # Test single embedding
        test_text = "What are the operating hours of Changi Airport?"
        print(f"\n📝 Test text: '{test_text}'")
        
        embedding = embedding_gen.generate_embedding(test_text)
        print(f"📐 Generated embedding shape: {embedding.shape}")
        print(f"🔢 Embedding sample (first 5 values): {embedding[:5]}")
        print(f"📊 Embedding stats: min={embedding.min():.3f}, max={embedding.max():.3f}, mean={embedding.mean():.3f}")
        
        # Test batch embeddings
        test_texts = [
            "Where can I find dining options at Changi Airport?",
            "How do I get to the airport from Singapore city?",
            "What shopping facilities are available?",
            "Are there lounges available for business travelers?",
            "What is the Rain Vortex at Jewel Changi Airport?"
        ]
        
        print(f"\n🔄 Testing batch embedding with {len(test_texts)} texts")
        batch_embeddings = embedding_gen.generate_embeddings_batch(test_texts)
        
        print(f"✅ Generated {len(batch_embeddings)} embeddings")
        for i, emb in enumerate(batch_embeddings):
            print(f"   📐 Embedding {i+1}: shape {emb.shape}, mean {emb.mean():.3f}")
        
        # Test similarity between embeddings
        if len(batch_embeddings) >= 2:
            print(f"\n🔍 Testing similarity calculation...")
            emb1, emb2 = batch_embeddings[0], batch_embeddings[1]
            
            # Cosine similarity
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            cosine_sim = dot_product / (norm1 * norm2)
            
            print(f"   🔢 Cosine similarity between first two embeddings: {cosine_sim:.3f}")
        
        print(f"\n🎉 Embedding generator test completed successfully!")
        print(f"✅ Model: {embedding_gen.model_name}")
        print(f"✅ Dimension: {embedding_gen.dimension}")
        
    except Exception as e:
        print(f"\n❌ Embedding generator test failed: {e}")
        import traceback
        traceback.print_exc()

def show_sample_documents():
    """Show sample documents that will be created"""
    print("📄 Sample Documents Preview")
    print("=" * 50)
    
    sample_docs = create_sample_documents()
    
    for i, doc in enumerate(sample_docs, 1):
        print(f"\n{i}. {doc['title']}")
        print(f"   ID: {doc['chunk_id']}")
        print(f"   Content: {doc['content'][:100]}...")
        print(f"   URL: {doc['url']}")

def main():
    """Main function to handle command line arguments"""
    import sys
    
    print("🤖 Changi RAG Chatbot - Embedding Generator")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        if command == "test":
            test_embedding_generator()
        elif command == "generate":
            process_documents_to_embeddings()
        elif command == "preview":
            show_sample_documents()
        else:
            print(f"❌ Unknown command: {command}")
            print("\nAvailable commands:")
            print("   python embedding_generator.py generate  - Generate embeddings")
            print("   python embedding_generator.py test      - Test embedding generator")
            print("   python embedding_generator.py preview   - Preview sample documents")
    else:
        # Default action: generate embeddings
        process_documents_to_embeddings()

if __name__ == "__main__":
    main()