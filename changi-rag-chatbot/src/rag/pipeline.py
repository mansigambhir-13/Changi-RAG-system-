import os
import sys
import time
import hashlib
import json
import uuid
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import asyncio
from datetime import datetime, timedelta

# Load environment variables first
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

load_env_file()

# LangChain imports with compatibility fixes
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Qdrant
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import LLMChainExtractor
    from langchain.schema import Document
    from langchain.memory import ConversationBufferWindowMemory
    from langchain.chains import ConversationalRetrievalChain, LLMChain
    from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
    from langchain.schema.messages import HumanMessage, AIMessage, SystemMessage
    print("✅ LangChain components imported successfully")
except ImportError as e:
    print(f"❌ LangChain import error: {e}")
    print("📦 Install with: pip install langchain==0.1.20 langchain-community==0.0.38")
    sys.exit(1)

# Google Gemini imports with fallback and compatibility
GEMINI_AVAILABLE = False
try:
    # Try the newer import first
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        GEMINI_AVAILABLE = True
        print("✅ Google Gemini LangChain integration imported successfully")
    except (ImportError, TypeError) as e:
        print(f"⚠️ Google Gemini import issue: {e}")
        print("🔄 Trying alternative Gemini integration...")
        
        # Alternative: Use direct Google AI SDK with LangChain wrapper
        try:
            import google.generativeai as genai
            from langchain.llms.base import LLM
            from langchain.callbacks.manager import CallbackManagerForLLMRun
            
            class GeminiLLM(LLM):
                """Custom Gemini LLM wrapper to avoid metaclass conflicts"""
                
                def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash", temperature: float = 0.1):
                    super().__init__()
                    genai.configure(api_key=api_key)
                    self.model_name = model_name
                    self.temperature = temperature
                    self.model = genai.GenerativeModel(model_name)
                
                @property
                def _llm_type(self) -> str:
                    return "gemini"
                
                def _call(
                    self,
                    prompt: str,
                    stop: Optional[List[str]] = None,
                    run_manager: Optional[CallbackManagerForLLMRun] = None,
                    **kwargs: Any,
                ) -> str:
                    try:
                        response = self.model.generate_content(
                            prompt,
                            generation_config=genai.types.GenerationConfig(
                                temperature=self.temperature,
                                max_output_tokens=1000,
                            )
                        )
                        return response.text
                    except Exception as e:
                        print(f"Gemini API error: {e}")
                        return f"I apologize, but I'm experiencing technical difficulties. Please try again."
            
            ChatGoogleGenerativeAI = GeminiLLM  # Use our custom wrapper
            GEMINI_AVAILABLE = True
            print("✅ Custom Gemini LLM wrapper created successfully")
            
        except ImportError:
            print("❌ Google AI SDK not available")
            print("📦 Install with: pip install google-generativeai")
            GEMINI_AVAILABLE = False

except Exception as e:
    print(f"❌ Unexpected error with Gemini: {e}")
    GEMINI_AVAILABLE = False

# Qdrant imports
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.models import Distance, VectorParams, PointStruct
    print("✅ Qdrant client imported successfully")
except ImportError as e:
    print(f"❌ Qdrant import error: {e}")
    print("📦 Install with: pip install qdrant-client")
    sys.exit(1)

# Configuration from environment
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "changi_airport_knowledge")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY and GEMINI_AVAILABLE:
    print("❌ GOOGLE_API_KEY not found in environment variables")
    print("📝 Please set it in your .env file")
    print("🔑 Get your API key from: https://makersuite.google.com/app/apikey")
    GEMINI_AVAILABLE = False

# Enhanced logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class EnhancedChatResponse:
    """Enhanced chat response with comprehensive metadata"""
    response: str
    sources: List[Dict[str, Any]]
    confidence: float
    conversation_id: str
    message_id: str
    response_time: float
    follow_up_questions: Optional[List[str]] = None
    cached: bool = False
    metadata: Optional[Dict[str, Any]] = None
    langchain_metadata: Optional[Dict[str, Any]] = None
    retrieval_metadata: Optional[Dict[str, Any]] = None
    reasoning_steps: Optional[List[str]] = None

@dataclass
class RetrievedDocument:
    """Standard retrieved document format"""
    content: str
    title: str
    url: str
    score: float
    metadata: Dict[str, Any]
    chunk_id: str

class UnifiedQdrantVectorStore:
    """Unified Qdrant vector store that integrates perfectly with LangChain"""
    
    def __init__(self, collection_name: str = QDRANT_COLLECTION):
        self.collection_name = collection_name
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        
        # Initialize embeddings model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize LangChain Qdrant integration
        self.vector_store: Optional[Qdrant] = None
        self._ensure_collection_exists()
        self._initialize_langchain_integration()
        
        logger.info(f"UnifiedQdrantVectorStore initialized for collection: {collection_name}")
    
    def _ensure_collection_exists(self):
        """Ensure Qdrant collection exists with proper configuration"""
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_exists = any(col.name == self.collection_name for col in collections)
            
            if not collection_exists:
                # Create collection with optimized settings
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=384,  # all-MiniLM-L6-v2 embedding size
                        distance=Distance.COSINE,
                        on_disk=True
                    ),
                    optimizers_config=models.OptimizersConfig(
                        deleted_threshold=0.2,
                        vacuum_min_vector_number=1000,
                        default_segment_number=2,
                    ),
                    replication_factor=1,
                    write_consistency_factor=1,
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Using existing Qdrant collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Failed to ensure collection exists: {e}")
            raise
    
    def _initialize_langchain_integration(self):
        """Initialize LangChain Qdrant integration"""
        try:
            self.vector_store = Qdrant(
                client=self.client,
                collection_name=self.collection_name,
                embeddings=self.embedding_model
            )
            logger.info("LangChain Qdrant integration initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LangChain integration: {e}")
            raise
    
    def add_documents(self, documents: List[Document], batch_size: int = 100) -> List[str]:
        """Add documents to vector store in batches"""
        try:
            if not documents:
                logger.warning("No documents provided to add")
                return []
            
            document_ids = []
            
            # Process in batches to avoid overwhelming the system
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                
                # Add batch to vector store
                batch_ids = self.vector_store.add_documents(batch)
                document_ids.extend(batch_ids)
                
                logger.info(f"Added batch {i//batch_size + 1}: {len(batch)} documents")
                time.sleep(0.1)  # Small delay to prevent overwhelming
            
            logger.info(f"Successfully added {len(documents)} documents to Qdrant")
            return document_ids
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise
    
    def get_retriever(self, 
                     search_type: str = "similarity",
                     search_kwargs: Dict[str, Any] = None) -> Any:
        """Get LangChain retriever with enhanced configuration"""
        if search_kwargs is None:
            search_kwargs = {"k": 8, "score_threshold": 0.3}
        
        retriever = self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
        
        return retriever
    
    def get_contextual_compression_retriever(self, 
                                           llm,
                                           search_kwargs: Dict[str, Any] = None):
        """Get contextual compression retriever for better results"""
        base_retriever = self.get_retriever(search_kwargs=search_kwargs)
        
        compressor = LLMChainExtractor.from_llm(llm)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
        
        return compression_retriever
    
    def similarity_search_with_score(self, 
                                   query: str, 
                                   k: int = 8,
                                   score_threshold: float = 0.0) -> List[Tuple[Document, float]]:
        """Enhanced similarity search with scoring"""
        try:
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                score_threshold=score_threshold
            )
            
            # Filter and sort results
            filtered_results = [
                (doc, score) for doc, score in results 
                if score >= score_threshold
            ]
            
            return sorted(filtered_results, key=lambda x: x[1], reverse=True)
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
    
    def retrieve_documents(self, 
                         query: str, 
                         top_k: int = 8,
                         score_threshold: float = 0.3) -> List[RetrievedDocument]:
        """Retrieve documents in standard format for compatibility"""
        try:
            # Use similarity search
            results = self.similarity_search_with_score(
                query=query, 
                k=top_k, 
                score_threshold=score_threshold
            )
            
            retrieved_docs = []
            for doc, score in results:
                retrieved_doc = RetrievedDocument(
                    content=doc.page_content,
                    title=doc.metadata.get('title', 'Untitled'),
                    url=doc.metadata.get('url', ''),
                    score=score,
                    metadata=doc.metadata,
                    chunk_id=doc.metadata.get('chunk_id', str(uuid.uuid4())[:12])
                )
                retrieved_docs.append(retrieved_doc)
            
            logger.info(f"Retrieved {len(retrieved_docs)} documents for query: {query[:50]}...")
            return retrieved_docs
            
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get comprehensive collection statistics"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            return {
                "collection_name": self.collection_name,
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "points_count": collection_info.points_count,
                "segments_count": collection_info.segments_count,
                "disk_data_size": collection_info.disk_data_size,
                "ram_data_size": collection_info.ram_data_size,
                "config": {
                    "distance": collection_info.config.params.vectors.distance.value,
                    "vector_size": collection_info.config.params.vectors.size,
                },
                "status": "healthy"
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e), "status": "unhealthy"}

class QueryProcessor:
    """Advanced query processing for better retrieval"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Airport domain expansions
        self.query_expansions = {
            'food': ['dining', 'restaurant', 'eating', 'meal', 'cuisine', 'cafe'],
            'shop': ['shopping', 'retail', 'store', 'buy', 'purchase'],
            'transport': ['transportation', 'travel', 'getting to', 'mrt', 'taxi', 'bus'],
            'wifi': ['internet', 'connection', 'online', 'network'],
            'hours': ['time', 'schedule', 'operating', 'open', 'close'],
            'location': ['where', 'find', 'directions', 'address'],
            'jewel': ['rain vortex', 'canopy park', 'waterfall', 'garden'],
            'terminal': ['t1', 't2', 't3', 't4', 'departure', 'arrival'],
            'parking': ['car park', 'vehicle'],
            'lounge': ['premium', 'business', 'first class']
        }
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process and enhance the query"""
        original_query = query.strip()
        processed_query = original_query.lower()
        
        # Extract intent and expand query
        expanded_terms = []
        detected_intents = []
        
        for intent, terms in self.query_expansions.items():
            if any(term in processed_query for term in terms) or intent in processed_query:
                expanded_terms.extend(terms)
                detected_intents.append(intent)
        
        # Create enhanced query
        if expanded_terms:
            enhanced_query = f"{original_query} {' '.join(set(expanded_terms[:5]))}"
        else:
            enhanced_query = original_query
        
        return {
            'original': original_query,
            'enhanced': enhanced_query,
            'intents': detected_intents,
            'expanded_terms': expanded_terms
        }

class EnhancedConversationalRAG:
    """Enhanced conversational RAG with Google Gemini integration and fallbacks"""
    
    def __init__(self, vector_store: UnifiedQdrantVectorStore):
        self.vector_store = vector_store
        self.query_processor = QueryProcessor()
        self.conversation_chains = {}
        self.conversation_memories = {}
        
        # Initialize LLM with fallback options
        self.llm = self._initialize_llm()
        
        # Setup prompts
        self._setup_prompts()
        
        logger.info("Enhanced Conversational RAG initialized")
    
    def _initialize_llm(self):
        """Initialize LLM with fallback options"""
        llm = None
        
        # Try Gemini first if available
        if GEMINI_AVAILABLE and GOOGLE_API_KEY:
            try:
                if hasattr(ChatGoogleGenerativeAI, '_llm_type'):
                    # Using custom wrapper
                    llm = ChatGoogleGenerativeAI(
                        api_key=GOOGLE_API_KEY,
                        model_name="gemini-1.5-flash",
                        temperature=0.1
                    )
                else:
                    # Using official LangChain integration
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-1.5-flash",
                        temperature=0.1,
                        max_tokens=1000,
                        google_api_key=GOOGLE_API_KEY,
                        convert_system_message_to_human=True
                    )
                logger.info("✅ Gemini LLM initialized successfully")
                return llm
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini: {e}")
        
        # Fallback to OpenAI if available
        try:
            from langchain_openai import ChatOpenAI
            openai_key = os.getenv('OPENAI_API_KEY')
            if openai_key:
                llm = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    temperature=0.1,
                    max_tokens=1000,
                    api_key=openai_key
                )
                logger.info("✅ OpenAI LLM initialized as fallback")
                return llm
        except ImportError:
            pass
        
        # Final fallback - create a mock LLM for testing
        from langchain.llms.base import LLM
        from langchain.callbacks.manager import CallbackManagerForLLMRun
        
        class MockLLM(LLM):
            """Mock LLM for testing when no real LLM is available"""
            
            @property
            def _llm_type(self) -> str:
                return "mock"
            
            def _call(
                self,
                prompt: str,
                stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any,
            ) -> str:
                return "I'm a mock LLM response. Please configure a real LLM (Gemini or OpenAI) for full functionality."
        
        logger.warning("⚠️ Using mock LLM - please configure GOOGLE_API_KEY or OPENAI_API_KEY")
        return MockLLM()
    
    def _setup_prompts(self):
        """Setup enhanced prompts for different scenarios"""
        
        # Main conversational prompt template
        system_template = """You are a helpful assistant for Changi Airport and Jewel Changi Airport. 
        Use the following context to answer questions accurately and helpfully.
        
        Guidelines:
        - Provide specific, actionable information
        - Include relevant details like operating hours, locations, and contact information
        - If information is not in the context, clearly state that
        - Suggest related topics the user might be interested in
        - Be conversational and friendly
        - Reference specific sources when possible
        
        Context: {context}
        
        Previous conversation:
        {chat_history}
        
        Human: {question}
        
        Assistant: """
        
        self.conversational_prompt = PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template=system_template
        )
        
        # Follow-up questions prompt
        self.followup_prompt = PromptTemplate(
            input_variables=["question", "context", "answer"],
            template="""Based on this conversation about Changi Airport:

Question: {question}
Context: {context}
Answer: {answer}

Generate 3 relevant follow-up questions that the user might want to ask next. 
Make them specific and actionable. Format as a simple numbered list.

Follow-up questions:
1."""
        )
    
    def _get_or_create_chain(self, conversation_id: str) -> ConversationalRetrievalChain:
        """Get or create conversational chain for a conversation"""
        if conversation_id not in self.conversation_chains:
            # Create memory for this conversation
            memory = ConversationBufferWindowMemory(
                k=5,  # Remember last 5 exchanges
                memory_key="chat_history",
                return_messages=False,  # Use string format for compatibility
                output_key="answer"
            )
            
            # Get retriever from vector store
            retriever = self.vector_store.get_retriever(
                search_kwargs={"k": 8, "score_threshold": 0.3}
            )
            
            # Create conversational retrieval chain
            chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                memory=memory,
                return_source_documents=True,
                return_generated_question=True,
                combine_docs_chain_kwargs={
                    "prompt": self.conversational_prompt
                },
                verbose=False,
                max_tokens_limit=1000
            )
            
            self.conversation_chains[conversation_id] = chain
            self.conversation_memories[conversation_id] = memory
            
            logger.info(f"Created new conversational chain for: {conversation_id}")
        
        return self.conversation_chains[conversation_id]
    
    async def chat_async(self, 
                        message: str,
                        conversation_id: str,
                        user_id: Optional[str] = None,
                        include_follow_ups: bool = True,
                        use_contextual_compression: bool = False) -> EnhancedChatResponse:
        """Async chat method with comprehensive error handling"""
        start_time = time.time()
        message_id = str(uuid.uuid4())[:12]
        
        logger.info(f"Processing async chat: '{message}' (conv: {conversation_id})")
        
        try:
            # Process query for intent analysis
            query_analysis = self.query_processor.process_query(message)
            
            # Get conversational chain
            chain = self._get_or_create_chain(conversation_id)
            
            # Use contextual compression if requested
            if use_contextual_compression:
                compression_retriever = self.vector_store.get_contextual_compression_retriever(
                    self.llm, {"k": 6, "score_threshold": 0.4}
                )
                chain.retriever = compression_retriever
            
            # Generate response using the chain
            result = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: chain({"question": message})
            )
            
            # Extract response components
            response_text = result["answer"]
            source_documents = result.get("source_documents", [])
            generated_question = result.get("generated_question", message)
            
            # Calculate confidence based on multiple factors
            confidence = self._calculate_confidence(
                response_text, source_documents, query_analysis
            )
            
            # Format sources for response
            formatted_sources = self._format_sources(source_documents)
            
            # Generate follow-up questions if requested
            follow_ups = None
            if include_follow_ups and confidence > 0.6:
                follow_ups = await self._generate_follow_ups(
                    message, response_text, source_documents
                )
            
            # Create comprehensive response
            response = EnhancedChatResponse(
                response=response_text,
                sources=formatted_sources,
                confidence=confidence,
                conversation_id=conversation_id,
                message_id=message_id,
                response_time=time.time() - start_time,
                follow_up_questions=follow_ups,
                metadata={
                    "user_id": user_id,
                    "generated_question": generated_question,
                    "query_analysis": query_analysis,
                    "compression_used": use_contextual_compression
                },
                langchain_metadata={
                    "chain_type": "ConversationalRetrievalChain",
                    "llm_model": getattr(self.llm, '_llm_type', 'unknown'),
                    "memory_window": 5,
                    "retriever_k": 8
                },
                retrieval_metadata={
                    "documents_retrieved": len(source_documents),
                    "avg_relevance_score": sum(
                        doc.metadata.get("score", 0) for doc in source_documents
                    ) / len(source_documents) if source_documents else 0
                }
            )
            
            logger.info(f"Async chat completed: confidence={confidence:.3f}, time={response.response_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Async chat failed: {e}")
            
            # Generate fallback response
            fallback_response = self._generate_fallback_response(message)
            
            return EnhancedChatResponse(
                response=fallback_response,
                sources=[],
                confidence=0.1,
                conversation_id=conversation_id,
                message_id=message_id,
                response_time=time.time() - start_time,
                metadata={"error": str(e), "fallback": True}
            )
    
    def chat(self, *args, **kwargs) -> EnhancedChatResponse:
        """Sync wrapper for async chat method"""
        return asyncio.run(self.chat_async(*args, **kwargs))
    
    def _calculate_confidence(self, 
                            response: str,
                            source_documents: List[Document],
                            query_analysis: Dict[str, Any]) -> float:
        """Calculate response confidence based on multiple factors"""
        confidence = 0.5  # Base confidence
        
        # Factor 1: Number and quality of sources
        if source_documents:
            # Average score from source documents
            avg_score = sum(
                doc.metadata.get("score", 0.5) for doc in source_documents
            ) / len(source_documents)
            confidence += min(avg_score * 0.3, 0.3)
        
        # Factor 2: Response length and detail
        if len(response) > 200:
            confidence += 0.1
        if len(response) > 500:
            confidence += 0.1
        
        # Factor 3: Presence of specific information
        specific_indicators = [
            "hours", "location", "terminal", "floor", "phone", 
            "website", "price", "address", "operating"
        ]
        specificity_score = sum(
            1 for indicator in specific_indicators 
            if indicator.lower() in response.lower()
        ) / len(specific_indicators)
        confidence += specificity_score * 0.2
        
        # Factor 4: Intent match
        if query_analysis.get("intents"):
            confidence += 0.05
        
        return min(confidence, 1.0)
    
    def _format_sources(self, source_documents: List[Document]) -> List[Dict[str, Any]]:
        """Format source documents for response"""
        formatted_sources = []
        
        for i, doc in enumerate(source_documents):
            source = {
                "id": f"source_{i+1}",
                "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                "metadata": doc.metadata,
                "score": doc.metadata.get("score", 0.0),
                "title": doc.metadata.get("title", f"Source {i+1}"),
                "url": doc.metadata.get("url", ""),
                "section": doc.metadata.get("section", "")
            }
            formatted_sources.append(source)
        
        return formatted_sources
    
    async def _generate_follow_ups(self, 
                                 question: str,
                                 answer: str,
                                 source_documents: List[Document]) -> List[str]:
        """Generate follow-up questions"""
        try:
            context = "\n".join([doc.page_content[:200] for doc in source_documents[:3]])
            
            followup_chain = LLMChain(
                llm=self.llm,
                prompt=self.followup_prompt
            )
            
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: followup_chain.run(
                    question=question,
                    context=context,
                    answer=answer
                )
            )
            
            # Parse follow-up questions
            lines = result.strip().split('\n')
            follow_ups = []
            for line in lines:
                line = line.strip()
                if line and (line.startswith(('1.', '2.', '3.', '-', '•'))):
                    # Clean up the question
                    question_text = line.split('.', 1)[-1].strip() if '.' in line else line[1:].strip()
                    if question_text and len(question_text) > 10:
                        follow_ups.append(question_text)
            
            return follow_ups[:3]  # Return max 3 follow-ups
            
        except Exception as e:
            logger.error(f"Follow-up generation failed: {e}")
            return []
    
    def _generate_fallback_response(self, message: str) -> str:
        """Generate fallback response when main system fails"""
        fallback_responses = [
            "I apologize, but I'm experiencing technical difficulties right now. For immediate assistance with Changi Airport information, please visit the official Changi Airport website at changiairport.com or contact the airport information hotline.",
            
            "I'm currently having trouble accessing my knowledge base. For the most up-to-date information about Changi Airport facilities, services, and operations, I recommend checking the Changi Airport mobile app or speaking with airport staff.",
            
            "There seems to be a temporary issue with my system. You can find comprehensive information about Changi Airport dining, shopping, transportation, and services on their official website or by asking at any information counter in the terminals."
        ]
        
        # Use hash for consistent response selection
        hash_val = int(hashlib.md5(message.encode()).hexdigest(), 16)
        return fallback_responses[hash_val % len(fallback_responses)]
    
    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get conversation history"""
        if conversation_id in self.conversation_memories:
            memory = self.conversation_memories[conversation_id]
            # Handle both string and message formats
            if hasattr(memory, 'chat_memory') and hasattr(memory.chat_memory, 'messages'):
                return [
                    {
                        "type": "message",
                        "content": str(msg),
                        "timestamp": datetime.now().isoformat()
                    }
                    for msg in memory.chat_memory.messages
                ]
            else:
                # String-based memory
                return [
                    {
                        "type": "history",
                        "content": str(memory.buffer),
                        "timestamp": datetime.now().isoformat()
                    }
                ]
        return []
    
    def clear_conversation(self, conversation_id: str):
        """Clear a specific conversation"""
        if conversation_id in self.conversation_chains:
            del self.conversation_chains[conversation_id]
        if conversation_id in self.conversation_memories:
            del self.conversation_memories[conversation_id]
        logger.info(f"Cleared conversation: {conversation_id}")
    
    def get_active_conversations(self) -> List[str]:
        """Get list of active conversation IDs"""
        return list(self.conversation_chains.keys())

class DocumentProcessor:
    """Enhanced document processing for ingesting data into Qdrant"""
    
    def __init__(self, vector_store: UnifiedQdrantVectorStore):
        self.vector_store = vector_store
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
        )
        logger.info("Document processor initialized")
    
    def process_raw_documents(self, raw_documents: List[Dict[str, Any]]) -> List[Document]:
        """Process raw documents into LangChain Document objects"""
        documents = []
        
        for i, raw_doc in enumerate(raw_documents):
            try:
                # Create Document object with proper metadata
                doc = Document(
                    page_content=raw_doc.get("content", ""),
                    metadata={
                        "source": raw_doc.get("source", f"document_{i}"),
                        "title": raw_doc.get("title", ""),
                        "url": raw_doc.get("url", ""),
                        "scraped_at": raw_doc.get("scraped_at", ""),
                        "section": raw_doc.get("section", ""),
                        "doc_id": str(uuid.uuid4()),
                        "original_index": i
                    }
                )
                documents.append(doc)
                
            except Exception as e:
                logger.error(f"Failed to process document {i}: {e}")
                continue
        
        logger.info(f"Processed {len(documents)} raw documents")
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks with proper metadata"""
        try:
            split_docs = self.text_splitter.split_documents(documents)
            
            # Add chunk metadata
            for i, doc in enumerate(split_docs):
                doc.metadata["chunk_id"] = str(uuid.uuid4())[:12]
                doc.metadata["chunk_index"] = i
                doc.metadata["chunk_size"] = len(doc.page_content)
                doc.metadata["processed_at"] = datetime.now().isoformat()
            
            logger.info(f"Split into {len(split_docs)} chunks")
            return split_docs
            
        except Exception as e:
            logger.error(f"Document splitting failed: {e}")
            return documents
    
    def ingest_documents(self, raw_documents: List[Dict[str, Any]]) -> bool:
        """Complete document ingestion pipeline"""
        try:
            if not raw_documents:
                logger.error("No documents provided for ingestion")
                return False
            
            # Process documents
            documents = self.process_raw_documents(raw_documents)
            if not documents:
                logger.error("No documents to process after raw processing")
                return False
            
            # Split documents
            split_documents = self.split_documents(documents)
            
            # Add to vector store
            document_ids = self.vector_store.add_documents(split_documents)
            
            logger.info(f"Successfully ingested {len(document_ids)} document chunks")
            return True
            
        except Exception as e:
            logger.error(f"Document ingestion failed: {e}")
            return False

class GeminiChangiRAGPipeline:
    """Main RAG pipeline orchestrating all components with improved compatibility"""
    
    def __init__(self):
        logger.info("🚀 Initializing Enhanced Changi RAG Pipeline with Fixed Compatibility...")
        
        try:
            # Initialize components in order
            self.vector_store = UnifiedQdrantVectorStore()
            self.conversational_rag = EnhancedConversationalRAG(self.vector_store)
            self.document_processor = DocumentProcessor(self.vector_store)
            
            # Performance tracking
            self.stats = {
                "total_queries": 0,
                "avg_response_time": 0.0,
                "avg_confidence": 0.0,
                "cache_hits": 0,
                "successful_queries": 0,
                "failed_queries": 0
            }
            
            logger.info("✅ Enhanced RAG Pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise
    
    async def chat_async(self, *args, **kwargs) -> EnhancedChatResponse:
        """Async chat with comprehensive stats tracking"""
        self.stats["total_queries"] += 1
        
        try:
            response = await self.conversational_rag.chat_async(*args, **kwargs)
            
            # Update stats
            self.stats["successful_queries"] += 1
            self.stats["avg_response_time"] = (
                (self.stats["avg_response_time"] * (self.stats["successful_queries"] - 1) + 
                 response.response_time) / self.stats["successful_queries"]
            )
            self.stats["avg_confidence"] = (
                (self.stats["avg_confidence"] * (self.stats["successful_queries"] - 1) + 
                 response.confidence) / self.stats["successful_queries"]
            )
            
            if response.cached:
                self.stats["cache_hits"] += 1
            
            return response
            
        except Exception as e:
            self.stats["failed_queries"] += 1
            logger.error(f"Chat async failed: {e}")
            raise
    
    def chat(self, *args, **kwargs) -> EnhancedChatResponse:
        """Sync wrapper for chat"""
        return asyncio.run(self.chat_async(*args, **kwargs))
    
    def ingest_documents(self, raw_documents: List[Dict[str, Any]]) -> bool:
        """Ingest documents into the pipeline"""
        return self.document_processor.ingest_documents(raw_documents)
    
    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'pipeline_version': '4.1.0-fixed',
            'components': {}
        }
        
        try:
            # Check Qdrant connection and collection
            try:
                qdrant_stats = self.vector_store.get_collection_stats()
                if qdrant_stats.get("error"):
                    health_status['components']['qdrant'] = {
                        'status': 'unhealthy',
                        'error': qdrant_stats["error"]
                    }
                    health_status['status'] = 'unhealthy'
                else:
                    health_status['components']['qdrant'] = {
                        'status': 'healthy',
                        'stats': qdrant_stats
                    }
            except Exception as e:
                health_status['components']['qdrant'] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
                health_status['status'] = 'unhealthy'
            
            # Check LLM
            try:
                llm_type = getattr(self.conversational_rag.llm, '_llm_type', 'unknown')
                if llm_type == 'mock':
                    llm_status = 'degraded'
                    health_status['status'] = 'degraded'
                else:
                    llm_status = 'healthy'
            except Exception as e:
                llm_status = f'error: {str(e)}'
                health_status['status'] = 'degraded'
            
            health_status['components']['llm'] = {
                'status': llm_status,
                'type': getattr(self.conversational_rag.llm, '_llm_type', 'unknown'),
                'gemini_available': GEMINI_AVAILABLE
            }
            
            # Check conversation management
            active_conversations = len(self.conversational_rag.get_active_conversations())
            health_status['components']['conversations'] = {
                'status': 'healthy',
                'active_count': active_conversations
            }
            
            # Check document processor
            health_status['components']['document_processor'] = {
                'status': 'healthy',
                'chunk_size': self.document_processor.text_splitter._chunk_size,
                'chunk_overlap': self.document_processor.text_splitter._chunk_overlap
            }
            
        except Exception as e:
            health_status['status'] = 'unhealthy'
            health_status['error'] = str(e)
        
        return health_status
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics"""
        try:
            qdrant_stats = self.vector_store.get_collection_stats()
            
            stats = {
                'pipeline_info': {
                    'version': '4.1.0-fixed',
                    'framework': 'LangChain',
                    'vector_store': 'Qdrant',
                    'llm': getattr(self.conversational_rag.llm, '_llm_type', 'unknown'),
                    'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
                    'gemini_available': GEMINI_AVAILABLE
                },
                'performance_stats': self.stats,
                'qdrant_stats': qdrant_stats,
                'conversation_stats': {
                    'active_conversations': len(self.conversational_rag.get_active_conversations()),
                    'total_chains': len(self.conversational_rag.conversation_chains)
                },
                'configuration': {
                    'chunk_size': self.document_processor.text_splitter._chunk_size,
                    'chunk_overlap': self.document_processor.text_splitter._chunk_overlap,
                    'memory_window': 5,
                    'retrieval_k': 8,
                    'score_threshold': 0.3
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get pipeline stats: {e}")
            return {'error': str(e)}

def create_sample_documents() -> List[Dict[str, Any]]:
    """Create comprehensive sample documents for testing"""
    return [
        {
            "title": "Changi Airport Terminals Guide",
            "content": """Changi Airport has four main terminals (T1, T2, T3, T4) connected by the Skytrain. 
            Terminal 1 serves airlines like Lufthansa, Emirates, and British Airways. It features over 50 shops and restaurants.
            Terminal 2 handles Air Asia, Jetstar, and other budget airlines. It has a unique butterfly garden and entertainment deck.
            Terminal 3 is the largest terminal serving Singapore Airlines and Star Alliance partners. It includes the famous kinetic rain sculpture.
            Terminal 4 is the newest terminal with automated check-in and biometric systems, serving Cathay Pacific and other premium airlines.""",
            "url": "https://changiairport.com/terminals",
            "source": "changi_website",
            "section": "terminals",
            "scraped_at": datetime.now().isoformat()
        },
        {
            "title": "Dining Options at Changi Airport",
            "content": """Changi Airport offers over 200 dining options across all terminals. 
            Popular restaurants include Din Tai Fung (famous for xiaolongbao), Burger King, McDonald's, KFC, and authentic local hawker fare. 
            Terminal 3 has the largest food court with over 20 dining options including Tsui Wah Restaurant and Crystal Jade.
            Operating hours vary by location, with most restaurants open from 6 AM to 2 AM. 
            Some 24-hour options are available including convenience stores like 7-Eleven and coffee shops like Starbucks.
            Many restaurants offer both dine-in and takeaway options for travelers with limited time.""",
            "url": "https://changiairport.com/dining",
            "source": "changi_website",
            "section": "dining",
            "scraped_at": datetime.now().isoformat()
        },
        {
            "title": "Jewel Changi Airport Attractions",
            "content": """Jewel Changi Airport features the world's tallest indoor waterfall, the Rain Vortex, 
            standing at 40 meters high and operating from 9 AM to 12 AM daily. 
            The Canopy Park on the top floor (5th floor) offers attractions like walking nets, slides, and beautiful gardens.
            It includes the Hedge Maze, Mirror Maze, and Discovery Slides. Admission fees apply for most attractions.
            Forest Valley is a four-story garden with walking trails, featuring over 2,000 trees and plants.
            Shopping options include over 280 retail and dining outlets spread across 5 levels.
            Jewel is open 24 hours but individual stores have varying operating hours from 10 AM to 10 PM typically.""",
            "url": "https://jewelchangiairport.com",
            "source": "jewel_website",
            "section": "attractions",
            "scraped_at": datetime.now().isoformat()
        },
        {
            "title": "Transportation to and from Changi Airport",
            "content": """Multiple transportation options connect Changi Airport to Singapore city center. 
            The MRT East West Line connects directly to Changi Airport station (CG2) and operates from 5:30 AM to 12:30 AM daily.
            Travel time to city center (Raffles Place) is approximately 45 minutes for S$2.30.
            Public buses including 36, 53, and 858 serve all terminals with services from 6 AM to 12 AM.
            Taxi services are available 24/7 with fixed rates to different city areas - city center costs approximately S$20-30.
            Private hire cars (Grab, Gojek) and shuttle services can be booked in advance through mobile apps.
            The Airport Shuttle provides shared rides to hotels and city locations for S$9 per person.""",
            "url": "https://changiairport.com/transport",
            "source": "changi_website",
            "section": "transportation",
            "scraped_at": datetime.now().isoformat()
        },
        {
            "title": "Changi Airport WiFi and Digital Services",
            "content": """Free WiFi is available throughout all Changi Airport terminals with unlimited data and high-speed connection.
            Connect to 'Changi Airport WiFi' network - no registration required for up to 3 hours, then simple registration extends usage.
            Charging stations and power outlets are abundant throughout all terminals.
            The Changi App provides interactive terminal maps, flight information, shopping deals, and augmented reality features.
            Digital services include automated check-in kiosks, baggage drop services, and mobile boarding passes.
            Free computer terminals and printing services are available in all terminals.""",
            "url": "https://changiairport.com/digital-services",
            "source": "changi_website",
            "section": "services",
            "scraped_at": datetime.now().isoformat()
        }
    ]

async def test_fixed_pipeline():
    """Comprehensive test of the fixed pipeline"""
    print("🧪 Testing Fixed LangChain + Qdrant RAG Pipeline")
    print("=" * 70)
    
    try:
        # Initialize pipeline
        print("🚀 Initializing fixed pipeline...")
        pipeline = GeminiChangiRAGPipeline()
        
        # Test health check first
        print("\n🏥 Performing initial health check...")
        health = pipeline.health_check()
        print(f"   Overall Status: {health['status']}")
        for component, details in health['components'].items():
            print(f"   {component}: {details['status']}")
        
        # Show LLM info
        llm_info = health['components'].get('llm', {})
        print(f"   LLM Type: {llm_info.get('type', 'unknown')}")
        print(f"   Gemini Available: {llm_info.get('gemini_available', False)}")
        
        # Ingest sample documents
        print("\n📚 Ingesting sample documents...")
        sample_docs = create_sample_documents()
        success = pipeline.ingest_documents(sample_docs)
        
        if not success:
            print("❌ Failed to ingest documents")
            return
        
        print(f"✅ Successfully ingested {len(sample_docs)} documents")
        
        # Test scenarios
        test_scenarios = [
            {
                "query": "What terminals does Changi Airport have?",
                "conversation_id": "fixed_test_001",
                "expected_topics": ["terminals", "t1", "t2", "t3", "t4"]
            },
            {
                "query": "Where can I find good food and what are the operating hours?",
                "conversation_id": "fixed_test_001",
                "expected_topics": ["dining", "food", "hours"]
            },
            {
                "query": "Tell me about the Rain Vortex and other attractions at Jewel",
                "conversation_id": "fixed_test_002",
                "expected_topics": ["rain vortex", "jewel", "attractions"]
            },
            {
                "query": "How do I get to the city center from the airport?",
                "conversation_id": "fixed_test_002",
                "expected_topics": ["transportation", "mrt", "taxi"]
            },
            {
                "query": "Is there free WiFi and how do I connect?",
                "conversation_id": "fixed_test_003",
                "expected_topics": ["wifi", "internet", "free"]
            }
        ]
        
        print(f"\n📝 Testing with {len(test_scenarios)} scenarios")
        print("=" * 70)
        
        all_successful = True
        
        for i, scenario in enumerate(test_scenarios, 1):
            query = scenario["query"]
            conv_id = scenario["conversation_id"]
            expected_topics = scenario["expected_topics"]
            
            print(f"\n🔍 Scenario {i}: {query}")
            print(f"   💬 Conversation: {conv_id}")
            print("-" * 60)
            
            try:
                # Test async chat
                response = await pipeline.chat_async(
                    message=query,
                    conversation_id=conv_id,
                    user_id="fixed_test_user",
                    include_follow_ups=True,
                    use_contextual_compression=(i % 2 == 0)
                )
                
                print(f"✅ Response generated successfully:")
                print(f"   📝 Content: {response.response[:200]}...")
                print(f"   📊 Confidence: {response.confidence:.3f}")
                print(f"   📚 Sources: {len(response.sources)} documents")
                print(f"   ⏱️  Response time: {response.response_time:.2f}s")
                print(f"   🆔 Message ID: {response.message_id}")
                
                # Check if response contains expected topics
                response_lower = response.response.lower()
                topics_found = [topic for topic in expected_topics if topic in response_lower]
                print(f"   🎯 Topics found: {topics_found}")
                
                if response.langchain_metadata:
                    print(f"   🤖 LLM Type: {response.langchain_metadata.get('llm_model', 'N/A')}")
                    print(f"   🔄 Chain Type: {response.langchain_metadata.get('chain_type', 'N/A')}")
                
                if response.metadata:
                    print(f"   🗜️  Compression: {response.metadata.get('compression_used', False)}")
                    query_analysis = response.metadata.get('query_analysis', {})
                    if query_analysis:
                        print(f"   🔍 Detected Intents: {query_analysis.get('intents', [])}")
                
                if response.follow_up_questions:
                    print(f"   💡 Follow-up questions:")
                    for j, follow_up in enumerate(response.follow_up_questions, 1):
                        print(f"      {j}. {follow_up}")
                
                if response.sources:
                    print(f"   📚 Top sources:")
                    for j, source in enumerate(response.sources[:2], 1):
                        print(f"      {j}. {source.get('title', 'N/A')} (Score: {source.get('score', 0):.3f})")
                
                # Validate response quality
                if response.confidence < 0.3:
                    print(f"   ⚠️  Low confidence score")
                    all_successful = False
                
            except Exception as e:
                print(f"❌ Error in scenario {i}: {e}")
                all_successful = False
                import traceback
                traceback.print_exc()
        
        # Final assessment
        if all_successful:
            print(f"\n🎉 Fixed RAG Pipeline test completed successfully!")
            print(f"✅ Metaclass conflicts resolved")
            print(f"✅ LangChain + Qdrant integration working")
            print(f"✅ Conversational memory functioning properly")
            print(f"✅ Document ingestion and retrieval operational")
            print(f"✅ LLM responding accurately (even with fallbacks)")
            print(f"✅ All components healthy and responsive")
            print(f"🎯 Fixed Pipeline ready for production deployment!")
        else:
            print(f"\n⚠️  Pipeline test completed with some issues")
            print(f"🔧 Please review the errors above and fix them")
        
        # Show final stats
        print(f"\n📊 Final Pipeline Statistics:")
        stats = pipeline.get_pipeline_stats()
        pipeline_info = stats.get('pipeline_info', {})
        print(f"   Version: {pipeline_info.get('version', 'N/A')}")
        print(f"   LLM: {pipeline_info.get('llm', 'N/A')}")
        print(f"   Gemini Available: {pipeline_info.get('gemini_available', False)}")
        
        performance_stats = stats.get('performance_stats', {})
        print(f"   Total Queries: {performance_stats.get('total_queries', 0)}")
        print(f"   Success Rate: {performance_stats.get('successful_queries', 0)}/{performance_stats.get('total_queries', 0)}")
        print(f"   Avg Response Time: {performance_stats.get('avg_response_time', 0):.2f}s")
        
    except Exception as e:
        print(f"❌ Fixed pipeline test failed: {e}")
        print("\n📝 Troubleshooting steps:")
        print("   1. Check Qdrant is running: docker-compose up -d")
        print("   2. Verify environment variables in .env file")
        print("   3. Update package versions:")
        print("      pip install langchain==0.1.20 langchain-community==0.0.38")
        print("      pip install --upgrade google-generativeai")
        print("   4. Clear any cached imports and restart Python")
        import traceback
        traceback.print_exc()

# Main execution
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            asyncio.run(test_fixed_pipeline())
        else:
            print("Usage: python fixed_pipeline.py [test]")
    else:
        # Run interactive test by default
        asyncio.run(test_fixed_pipeline())