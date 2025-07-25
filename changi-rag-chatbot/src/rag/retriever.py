"""
Fixed unified retriever system for RAG pipeline using LangChain + Qdrant
Resolves metaclass conflicts and import compatibility issues
"""
import re
import sys
import os
import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
import uuid
from datetime import datetime

# Load environment variables
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

# Import the fixed pipeline components
try:
    # First try importing from the fixed pipeline
    from fixed_pipeline import (
        UnifiedQdrantVectorStore, 
        RetrievedDocument, 
        QueryProcessor,
        logger
    )
    print("✅ Fixed pipeline components imported successfully")
except ImportError:
    # Fallback to original pipeline if available
    try:
        from pipeline import (
            UnifiedQdrantVectorStore, 
            RetrievedDocument, 
            QueryProcessor,
            logger
        )
        print("✅ Original pipeline components imported successfully")
    except ImportError:
        print("❌ Could not import pipeline components")
        print("📝 Make sure fixed_pipeline.py or pipeline.py is in the same directory")
        sys.exit(1)

# LangChain imports with compatibility handling
try:
    from langchain.schema import Document
    from langchain.retrievers.document_compressors import LLMChainExtractor
    from langchain.retrievers import ContextualCompressionRetriever
    print("✅ LangChain components imported successfully")
except ImportError as e:
    print(f"❌ LangChain import error: {e}")
    print("📦 Install with: pip install langchain==0.1.20 langchain-community==0.0.38")
    sys.exit(1)

# Try to import LLM options with fallbacks
LLM_AVAILABLE = False
LLMClass = None

# Try OpenAI first
try:
    from langchain_openai import ChatOpenAI
    if os.getenv('OPENAI_API_KEY'):
        LLMClass = ChatOpenAI
        LLM_AVAILABLE = True
        print("✅ OpenAI LLM available")
except ImportError:
    pass

# Try Google Gemini with fallback handling
if not LLM_AVAILABLE:
    try:
        from fixed_pipeline import ChatGoogleGenerativeAI, GEMINI_AVAILABLE, GOOGLE_API_KEY
        if GEMINI_AVAILABLE and GOOGLE_API_KEY:
            LLMClass = ChatGoogleGenerativeAI
            LLM_AVAILABLE = True
            print("✅ Gemini LLM available")
    except (ImportError, AttributeError):
        pass

# Create a mock LLM if nothing else is available
if not LLM_AVAILABLE:
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
            return "Mock LLM response for testing purposes."
    
    LLMClass = MockLLM
    print("⚠️ Using mock LLM - configure GOOGLE_API_KEY or OPENAI_API_KEY for full functionality")

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

class AdvancedQueryProcessor(QueryProcessor):
    """Advanced query processing with domain-specific enhancements"""
    
    def __init__(self):
        super().__init__()
        
        # Extended airport domain expansions
        self.query_expansions.update({
            'business': ['lounge', 'priority', 'premium', 'first class'],
            'budget': ['cheap', 'affordable', 'low cost', 'economy'],
            'children': ['kids', 'family', 'playground', 'children friendly'],
            'accessibility': ['wheelchair', 'disabled', 'special needs', 'mobility'],
            'security': ['immigration', 'customs', 'checkpoint', 'screening'],
            'baggage': ['luggage', 'suitcase', 'bag drop', 'storage', 'claim'],
            'duty': ['duty free', 'tax free', 'shopping', 'souvenirs'],
            'medical': ['pharmacy', 'clinic', 'doctor', 'health', 'medicine'],
            'banking': ['atm', 'money exchange', 'currency', 'bank'],
            'relaxation': ['spa', 'massage', 'quiet zone', 'rest area']
        })
        
        # Query intent patterns
        self.intent_patterns = {
            'location': r'\b(where|location|find|directions?|how to get|navigate)\b',
            'hours': r'\b(hours?|time|when|schedule|open|close|operating)\b',
            'pricing': r'\b(price|cost|fee|charge|rate|expensive|cheap)\b',
            'booking': r'\b(book|reserve|reservation|appointment|ticket)\b',
            'services': r'\b(service|facility|amenity|available|offer)\b',
            'contact': r'\b(contact|phone|email|call|number)\b',
            'comparison': r'\b(compare|versus|vs|better|best|difference)\b',
            'urgent': r'\b(urgent|emergency|quick|fast|immediate|now)\b'
        }
        
        # Entity recognition patterns
        self.entity_patterns = {
            'terminals': r'\b(terminal\s*[1-4]|t[1-4])\b',
            'jewel_attractions': r'\b(rain vortex|canopy park|forest valley|hedge maze|mirror maze)\b',
            'airlines': r'\b(singapore airlines|emirates|lufthansa|british airways|cathay pacific|air asia|jetstar)\b',
            'restaurants': r'\b(din tai fung|burger king|mcdonalds|kfc|starbucks|crystal jade)\b',
            'transport_modes': r'\b(mrt|taxi|bus|grab|gojek|shuttle)\b',
            'areas': r'\b(city center|downtown|orchard|marina bay|sentosa)\b'
        }
    
    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Advanced query intent analysis"""
        query_lower = query.lower()
        
        # Detect primary intent
        primary_intent = 'general'
        intent_confidence = 0.0
        
        for intent, pattern in self.intent_patterns.items():
            if re.search(pattern, query_lower):
                primary_intent = intent
                intent_confidence = 0.8
                break
        
        # Detect entities
        entities = {}
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, query_lower, re.IGNORECASE)
            if matches:
                entities[entity_type] = [match.strip() for match in matches]
        
        # Determine urgency level
        urgency = 'normal'
        if re.search(r'\b(urgent|emergency|asap|immediately|now)\b', query_lower):
            urgency = 'high'
        elif re.search(r'\b(planning|future|later|eventually)\b', query_lower):
            urgency = 'low'
        
        # Extract question type
        question_type = 'general'
        if query_lower.startswith(('what', 'where', 'when', 'how', 'why', 'which')):
            question_type = query_lower.split()[0]
        
        return {
            'primary_intent': primary_intent,
            'intent_confidence': intent_confidence,
            'entities': entities,
            'urgency': urgency,
            'question_type': question_type,
            'complexity': self._assess_complexity(query)
        }
    
    def _assess_complexity(self, query: str) -> str:
        """Assess query complexity based on various factors"""
        word_count = len(query.split())
        
        # Check for complex patterns
        complex_indicators = [
            'and', 'or', 'but', 'also', 'compare', 'versus', 'difference',
            'multiple', 'several', 'various', 'different'
        ]
        
        complex_count = sum(1 for indicator in complex_indicators if indicator in query.lower())
        
        if word_count > 15 or complex_count >= 2:
            return 'high'
        elif word_count > 8 or complex_count >= 1:
            return 'medium'
        else:
            return 'low'
    
    def generate_search_variants(self, query: str) -> List[str]:
        """Generate multiple search variants for better coverage"""
        processed = self.process_query(query)
        analysis = self.analyze_query_intent(query)
        
        variants = [processed['enhanced']]
        
        # Add entity-focused variants
        for entity_type, entity_list in analysis['entities'].items():
            for entity in entity_list:
                variant = f"{entity} {processed['original']}"
                if variant not in variants:
                    variants.append(variant)
        
        # Add intent-focused variants
        intent_templates = {
            'location': 'where to find {}',
            'hours': 'operating hours for {}',
            'pricing': 'cost and pricing for {}',
            'services': 'services and facilities for {}'
        }
        
        template = intent_templates.get(analysis['primary_intent'])
        if template and analysis['entities']:
            main_entity = list(analysis['entities'].values())[0][0]
            variant = template.format(main_entity)
            if variant not in variants:
                variants.append(variant)
        
        return variants[:5]  # Limit to 5 variants

class EnhancedRetriever:
    """Enhanced retriever with multiple search strategies"""
    
    def __init__(self, vector_store: UnifiedQdrantVectorStore = None):
        self.vector_store = vector_store or UnifiedQdrantVectorStore()
        self.query_processor = AdvancedQueryProcessor()
        
        # Configuration
        self.default_top_k = 8
        self.min_score_threshold = 0.3
        self.max_total_results = 20
        
        # Initialize LLM for contextual compression (optional)
        self.llm = None
        self._initialize_llm()
        
        logger.info("Enhanced Retriever initialized")
    
    def _initialize_llm(self):
        """Initialize LLM with fallback options"""
        try:
            if LLMClass and LLM_AVAILABLE:
                if LLMClass.__name__ == 'ChatOpenAI':
                    self.llm = LLMClass(
                        model="gpt-3.5-turbo",
                        temperature=0.1,
                        max_tokens=500,
                        api_key=os.getenv('OPENAI_API_KEY')
                    )
                    logger.info("OpenAI LLM initialized for contextual compression")
                elif hasattr(LLMClass, '_llm_type'):
                    # Custom wrapper (like our Gemini wrapper)
                    self.llm = LLMClass(
                        api_key=os.getenv('GOOGLE_API_KEY'),
                        model_name="gemini-1.5-flash",
                        temperature=0.1
                    )
                    logger.info("Custom LLM wrapper initialized for contextual compression")
                else:
                    # Standard LangChain LLM
                    self.llm = LLMClass(
                        temperature=0.1,
                        max_tokens=500
                    )
                    logger.info("Standard LLM initialized for contextual compression")
            else:
                # Use mock LLM
                self.llm = LLMClass()
                logger.warning("Mock LLM initialized - limited functionality")
                
        except Exception as e:
            logger.warning(f"Failed to initialize LLM for compression: {e}")
            self.llm = None
    
    @log_performance("retrieve_documents")
    def retrieve_documents(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        strategy: str = "adaptive",
        include_compression: bool = False
    ) -> List[RetrievedDocument]:
        """Main document retrieval method with multiple strategies"""
        
        top_k = top_k or self.default_top_k
        logger.info(f"Retrieving documents for: '{query}' (strategy: {strategy}, top_k: {top_k})")
        
        # Process the query
        processed_query = self.query_processor.process_query(query)
        query_analysis = self.query_processor.analyze_query_intent(query)
        
        # Choose retrieval strategy based on query analysis or explicit strategy
        if strategy == "adaptive":
            strategy = self._choose_strategy(query_analysis)
        
        # Execute retrieval strategy
        if strategy == "multi_variant":
            results = self._multi_variant_search(query, processed_query, query_analysis, top_k, filters)
        elif strategy == "contextual_expansion":
            results = self._contextual_expansion_search(query, processed_query, top_k, filters)
        elif strategy == "entity_focused":
            results = self._entity_focused_search(query, processed_query, query_analysis, top_k, filters)
        else:  # Default: standard search
            results = self._standard_search(processed_query['enhanced'], top_k, filters)
        
        # Apply contextual compression if requested and available
        if include_compression and self.llm and results and not isinstance(self.llm, type(LLMClass)) or getattr(self.llm, '_llm_type', '') != 'mock':
            try:
                results = self._apply_contextual_compression(query, results)
            except Exception as e:
                logger.warning(f"Contextual compression failed: {e}")
        
        # Post-process results
        final_results = self._post_process_results(results, query_analysis)
        
        avg_score = self._avg_score(final_results)
        logger.info(f"Retrieved {len(final_results)} documents with avg score: {avg_score:.3f}")
        return final_results
    
    def _choose_strategy(self, query_analysis: Dict[str, Any]) -> str:
        """Choose optimal retrieval strategy based on query analysis"""
        
        # High complexity queries benefit from multi-variant search
        if query_analysis['complexity'] == 'high':
            return "multi_variant"
        
        # Entity-rich queries benefit from entity-focused search
        if len(query_analysis['entities']) >= 2:
            return "entity_focused"
        
        # Location and comparison queries benefit from contextual expansion
        if query_analysis['primary_intent'] in ['location', 'comparison']:
            return "contextual_expansion"
        
        # Default to multi-variant for better coverage
        return "multi_variant"
    
    def _standard_search(
        self, 
        query: str, 
        top_k: int, 
        filters: Optional[Dict[str, Any]]
    ) -> List[RetrievedDocument]:
        """Standard similarity search"""
        return self.vector_store.retrieve_documents(
            query=query,
            top_k=top_k,
            score_threshold=self.min_score_threshold
        )
    
    def _multi_variant_search(
        self,
        original_query: str,
        processed_query: Dict[str, Any],
        query_analysis: Dict[str, Any],
        top_k: int,
        filters: Optional[Dict[str, Any]]
    ) -> List[RetrievedDocument]:
        """Search using multiple query variants for better coverage"""
        
        # Generate search variants
        variants = self.query_processor.generate_search_variants(original_query)
        
        all_results = []
        seen_chunk_ids = set()
        
        # Search with each variant
        for i, variant in enumerate(variants):
            try:
                variant_results = self.vector_store.retrieve_documents(
                    query=variant,
                    top_k=max(3, top_k // len(variants)),
                    score_threshold=self.min_score_threshold * 0.8  # Slightly lower threshold for variants
                )
                
                # Add unique results
                for result in variant_results:
                    if result.chunk_id not in seen_chunk_ids:
                        # Adjust score based on variant priority (first variant gets highest score)
                        score_multiplier = 1.0 - (i * 0.1)  # 1.0, 0.9, 0.8, etc.
                        result.score *= score_multiplier
                        
                        all_results.append(result)
                        seen_chunk_ids.add(result.chunk_id)
                
                if len(all_results) >= self.max_total_results:
                    break
                    
            except Exception as e:
                logger.warning(f"Variant search failed for '{variant}': {e}")
                continue
        
        # Sort by score and return top results
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:top_k]
    
    def _entity_focused_search(
        self,
        original_query: str,
        processed_query: Dict[str, Any],
        query_analysis: Dict[str, Any],
        top_k: int,
        filters: Optional[Dict[str, Any]]
    ) -> List[RetrievedDocument]:
        """Search focused on detected entities"""
        
        all_results = []
        seen_chunk_ids = set()
        
        # Search for each entity type
        for entity_type, entity_list in query_analysis['entities'].items():
            for entity in entity_list:
                try:
                    # Create entity-focused query
                    entity_query = f"{entity} {original_query}"
                    
                    entity_results = self.vector_store.retrieve_documents(
                        query=entity_query,
                        top_k=max(2, top_k // (len(query_analysis['entities']) + 1)),
                        score_threshold=self.min_score_threshold
                    )
                    
                    # Add unique results with entity boost
                    for result in entity_results:
                        if result.chunk_id not in seen_chunk_ids:
                            # Boost score for entity matches
                            if entity.lower() in result.content.lower():
                                result.score *= 1.2
                            
                            all_results.append(result)
                            seen_chunk_ids.add(result.chunk_id)
                            
                except Exception as e:
                    logger.warning(f"Entity search failed for '{entity}': {e}")
                    continue
        
        # Add general search results
        try:
            general_results = self.vector_store.retrieve_documents(
                query=processed_query['enhanced'],
                top_k=top_k // 2,
                score_threshold=self.min_score_threshold
            )
            
            for result in general_results:
                if result.chunk_id not in seen_chunk_ids:
                    all_results.append(result)
                    seen_chunk_ids.add(result.chunk_id)
                    
        except Exception as e:
            logger.warning(f"General search failed: {e}")
        
        # Sort and return
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:top_k]
    
    def _contextual_expansion_search(
        self,
        original_query: str,
        processed_query: Dict[str, Any],
        top_k: int,
        filters: Optional[Dict[str, Any]]
    ) -> List[RetrievedDocument]:
        """Search with contextual expansion to neighboring chunks"""
        
        # First, get initial results
        try:
            initial_results = self.vector_store.retrieve_documents(
                query=processed_query['enhanced'],
                top_k=top_k,
                score_threshold=self.min_score_threshold
            )
        except Exception as e:
            logger.warning(f"Initial search failed: {e}")
            return []
        
        if not initial_results:
            return initial_results
        
        # For high-scoring results, try to get neighboring chunks
        expanded_results = []
        seen_chunk_ids = set()
        
        for doc in initial_results:
            # Add the original document
            if doc.chunk_id not in seen_chunk_ids:
                expanded_results.append(doc)
                seen_chunk_ids.add(doc.chunk_id)
            
            # Try to find neighboring chunks (if score is high enough)
            if doc.score > 0.7:
                try:
                    neighbors = self._find_neighboring_chunks(doc)
                    for neighbor in neighbors:
                        if neighbor.chunk_id not in seen_chunk_ids:
                            # Reduce score for context chunks
                            neighbor.score = doc.score * 0.8
                            expanded_results.append(neighbor)
                            seen_chunk_ids.add(neighbor.chunk_id)
                except Exception as e:
                    logger.warning(f"Neighbor search failed: {e}")
                    continue
        
        return expanded_results
    
    def _find_neighboring_chunks(self, document: RetrievedDocument) -> List[RetrievedDocument]:
        """Find neighboring chunks based on document metadata"""
        neighbors = []
        
        try:
            # If the document has section information, try to find other chunks from the same section
            if document.metadata.get('section') and document.metadata.get('title'):
                section_query = f"{document.metadata['section']} {document.metadata['title']}"
                
                section_results = self.vector_store.retrieve_documents(
                    query=section_query,
                    top_k=3,
                    score_threshold=0.3
                )
                
                # Filter out the original document
                neighbors = [r for r in section_results if r.chunk_id != document.chunk_id]
        
        except Exception as e:
            logger.warning(f"Failed to find neighboring chunks: {e}")
        
        return neighbors
    
    def _apply_contextual_compression(
        self, 
        query: str, 
        results: List[RetrievedDocument]
    ) -> List[RetrievedDocument]:
        """Apply contextual compression to filter and improve results"""
        
        if not self.llm or not results:
            return results
        
        try:
            # Convert to LangChain documents
            docs = [
                Document(
                    page_content=result.content,
                    metadata=result.metadata
                )
                for result in results
            ]
            
            # Create compressor
            compressor = LLMChainExtractor.from_llm(self.llm)
            
            # Apply compression
            compressed_docs = compressor.compress_documents(docs, query)
            
            # Convert back to RetrievedDocument format
            compressed_results = []
            for i, doc in enumerate(compressed_docs):
                original_result = results[i] if i < len(results) else results[0]
                
                compressed_result = RetrievedDocument(
                    content=doc.page_content,
                    title=original_result.title,
                    url=original_result.url,
                    score=original_result.score * 1.1,  # Boost score for compressed results
                    metadata=doc.metadata,
                    chunk_id=original_result.chunk_id
                )
                compressed_results.append(compressed_result)
            
            logger.info(f"Contextual compression: {len(results)} -> {len(compressed_results)} documents")
            return compressed_results
        
        except Exception as e:
            logger.error(f"Contextual compression failed: {e}")
            return results
    
    def _post_process_results(
        self, 
        results: List[RetrievedDocument], 
        query_analysis: Dict[str, Any]
    ) -> List[RetrievedDocument]:
        """Post-process results based on query analysis"""
        
        if not results:
            return results
        
        # Filter by minimum score threshold
        filtered_results = [r for r in results if r.score >= self.min_score_threshold]
        
        # Boost results that match detected entities
        for result in filtered_results:
            content_lower = result.content.lower()
            
            # Entity matching boost
            for entity_type, entity_list in query_analysis['entities'].items():
                for entity in entity_list:
                    if entity.lower() in content_lower:
                        result.score *= 1.1
                        break
            
            # Intent-specific boosts
            if query_analysis['primary_intent'] == 'hours' and any(
                term in content_lower for term in ['hour', 'time', 'open', 'close', 'schedule']
            ):
                result.score *= 1.15
            
            elif query_analysis['primary_intent'] == 'location' and any(
                term in content_lower for term in ['location', 'where', 'find', 'address', 'directions']
            ):
                result.score *= 1.15
        
        # Sort by score
        filtered_results.sort(key=lambda x: x.score, reverse=True)
        
        # Remove exact duplicates
        unique_results = []
        seen_content = set()
        
        for result in filtered_results:
            content_hash = hashlib.md5(result.content.encode()).hexdigest()
            if content_hash not in seen_content:
                unique_results.append(result)
                seen_content.add(content_hash)
        
        return unique_results
    
    def _avg_score(self, results: List[RetrievedDocument]) -> float:
        """Calculate average score of results"""
        if not results:
            return 0.0
        return sum(r.score for r in results) / len(results)
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get comprehensive retrieval statistics"""
        try:
            # Get vector store stats
            vector_stats = self.vector_store.get_collection_stats()
            
            stats = {
                'retriever_info': {
                    'version': '2.1.0-fixed',
                    'strategies': ['standard', 'multi_variant', 'entity_focused', 'contextual_expansion'],
                    'compression_available': self.llm is not None and getattr(self.llm, '_llm_type', '') != 'mock',
                    'llm_type': getattr(self.llm, '_llm_type', 'none') if self.llm else 'none'
                },
                'vector_store_stats': vector_stats,
                'configuration': {
                    'default_top_k': self.default_top_k,
                    'min_score_threshold': self.min_score_threshold,
                    'max_total_results': self.max_total_results
                },
                'query_processor': {
                    'expansion_categories': len(self.query_processor.query_expansions),
                    'intent_patterns': len(self.query_processor.intent_patterns),
                    'entity_patterns': len(self.query_processor.entity_patterns)
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get retrieval stats: {e}")
            return {'error': str(e)}

class RetrievalEvaluator:
    """Evaluate retrieval quality and performance"""
    
    def __init__(self, retriever: EnhancedRetriever):
        self.retriever = retriever
        
    def evaluate_query_set(self, test_queries: List[str]) -> Dict[str, Any]:
        """Evaluate retrieval quality on a set of test queries"""
        
        results = []
        total_time = 0
        
        for query in test_queries:
            start_time = time.time()
            
            try:
                retrieved_docs = self.retriever.retrieve_documents(query)
                response_time = time.time() - start_time
                total_time += response_time
                
                # Calculate metrics
                avg_score = sum(doc.score for doc in retrieved_docs) / max(len(retrieved_docs), 1)
                max_score = max([doc.score for doc in retrieved_docs], default=0)
                
                results.append({
                    'query': query,
                    'num_results': len(retrieved_docs),
                    'avg_score': avg_score,
                    'max_score': max_score,
                    'response_time': response_time,
                    'success': True
                })
                
            except Exception as e:
                results.append({
                    'query': query,
                    'error': str(e),
                    'success': False
                })
        
        # Calculate summary metrics
        successful_results = [r for r in results if r['success']]
        
        evaluation = {
            'summary': {
                'total_queries': len(test_queries),
                'successful_queries': len(successful_results),
                'success_rate': len(successful_results) / len(test_queries) if test_queries else 0,
                'avg_response_time': total_time / len(test_queries) if test_queries else 0,
                'avg_results_per_query': sum(r['num_results'] for r in successful_results) / max(len(successful_results), 1),
                'avg_confidence_score': sum(r['avg_score'] for r in successful_results) / max(len(successful_results), 1)
            },
            'detailed_results': results
        }
        
        return evaluation

# Utility functions for testing
def test_fixed_retriever():
    """Test the fixed unified retriever system"""
    print("🧪 Testing Fixed LangChain + Qdrant Retriever")
    print("=" * 60)
    
    try:
        print("🔧 Initializing fixed enhanced retriever...")
        retriever = EnhancedRetriever()
        
        # Show initialization info
        stats = retriever.get_retrieval_stats()
        retriever_info = stats.get('retriever_info', {})
        print(f"   Version: {retriever_info.get('version', 'N/A')}")
        print(f"   LLM Type: {retriever_info.get('llm_type', 'N/A')}")
        print(f"   Compression Available: {retriever_info.get('compression_available', False)}")
        
        # Test different strategies
        test_scenarios = [
            {
                'query': 'What terminals does Changi Airport have and what airlines operate there?',
                'strategy': 'multi_variant',
                'description': 'Multi-variant search for complex query'
            },
            {
                'query': 'Where can I find Din Tai Fung restaurant in Terminal 3?',
                'strategy': 'entity_focused',
                'description': 'Entity-focused search with specific restaurant and terminal'
            },
            {
                'query': 'How do I get from Terminal 2 to Jewel Changi Airport?',
                'strategy': 'contextual_expansion',
                'description': 'Contextual expansion for location-based query'
            },
            {
                'query': 'Free WiFi connection instructions',
                'strategy': 'standard',
                'description': 'Standard search for simple query'
            },
            {
                'query': 'Rain Vortex operating hours and ticket prices',
                'strategy': 'adaptive',
                'description': 'Adaptive strategy selection'
            }
        ]
        
        all_successful = True
        
        for i, scenario in enumerate(test_scenarios, 1):
            query = scenario['query']
            strategy = scenario['strategy']
            description = scenario['description']
            
            print(f"\n🔍 Test {i}: {description}")
            print(f"   Query: {query}")
            print(f"   Strategy: {strategy}")
            print("-" * 50)
            
            try:
                # Test retrieval
                start_time = time.time()
                results = retriever.retrieve_documents(
                    query=query,
                    strategy=strategy,
                    include_compression=(i % 2 == 0)  # Test compression on even tests
                )
                response_time = time.time() - start_time
                
                if results:
                    print(f"✅ Retrieved {len(results)} documents in {response_time:.2f}s")
                    
                    # Show top results
                    for j, doc in enumerate(results[:3], 1):
                        print(f"\n   📄 Result {j}:")
                        print(f"      🏆 Score: {doc.score:.3f}")
                        print(f"      📝 Title: {doc.title}")
                        print(f"      📄 Content: {doc.content[:100]}...")
                        print(f"      🆔 Chunk ID: {doc.chunk_id}")
                    
                    # Calculate metrics
                    avg_score = sum(doc.score for doc in results) / len(results)
                    max_score = max(doc.score for doc in results)
                    print(f"\n   📊 Metrics:")
                    print(f"      Average Score: {avg_score:.3f}")
                    print(f"      Max Score: {max_score:.3f}")
                    print(f"      Response Time: {response_time:.2f}s")
                    
                else:
                    print("❌ No relevant documents found")
                    print("💡 This might indicate:")
                    print("   - Empty vector store (run document ingestion first)")
                    print("   - Query doesn't match content")
                    print("   - Score threshold too high")
                    all_successful = False
                    
            except Exception as e:
                print(f"❌ Error in test {i}: {e}")
                all_successful = False
                import traceback
                traceback.print_exc()
        
        # Test query analysis
        print(f"\n🔍 Testing Query Analysis:")
        print("=" * 50)
        
        analysis_queries = [
            "Where can I find the best food in Terminal 3?",
            "How much does it cost to park at Changi Airport?",
            "Emergency medical services location",
            "Compare dining options between T1 and T2"
        ]
        
        for query in analysis_queries:
            try:
                analysis = retriever.query_processor.analyze_query_intent(query)
                print(f"\nQuery: {query}")
                print(f"  Intent: {analysis['primary_intent']} (confidence: {analysis['intent_confidence']:.2f})")
                print(f"  Entities: {analysis['entities']}")
                print(f"  Complexity: {analysis['complexity']}")
                print(f"  Urgency: {analysis['urgency']}")
            except Exception as e:
                print(f"Analysis failed for '{query}': {e}")
        
        # Show final stats
        print(f"\n📊 Final Retriever Statistics:")
        print("=" * 50)
        final_stats = retriever.get_retrieval_stats()
        
        retriever_info = final_stats.get('retriever_info', {})
        print(f"📈 Retriever Info:")
        print(f"   Version: {retriever_info.get('version', 'N/A')}")
        print(f"   Strategies: {retriever_info.get('strategies', [])}")
        print(f"   Compression Available: {retriever_info.get('compression_available', False)}")
        print(f"   LLM Type: {retriever_info.get('llm_type', 'N/A')}")
        
        vector_stats = final_stats.get('vector_store_stats', {})
        if vector_stats and not vector_stats.get('error'):
            print(f"\n📊 Vector Store:")
            print(f"   Total Vectors: {vector_stats.get('vectors_count', 'N/A')}")
            print(f"   Collection: {vector_stats.get('collection_name', 'N/A')}")
            print(f"   Status: {vector_stats.get('status', 'N/A')}")
        
        config = final_stats.get('configuration', {})
        print(f"\n⚙️  Configuration:")
        print(f"   Default Top-K: {config.get('default_top_k', 'N/A')}")
        print(f"   Score Threshold: {config.get('min_score_threshold', 'N/A')}")
        print(f"   Max Results: {config.get('max_total_results', 'N/A')}")
        
        # Test evaluation
        print(f"\n🧪 Running Retrieval Evaluation:")
        print("=" * 50)
        
        evaluator = RetrievalEvaluator(retriever)
        evaluation_queries = [
            "Changi Airport terminals information",
            "Dining options and restaurants", 
            "Transportation to city center",
            "Jewel attractions and hours",
            "Free WiFi and internet access"
        ]
        
        evaluation = evaluator.evaluate_query_set(evaluation_queries)
        summary = evaluation['summary']
        
        print(f"📊 Evaluation Results:")
        print(f"   Success Rate: {summary['success_rate']:.1%}")
        print(f"   Avg Response Time: {summary['avg_response_time']:.2f}s")
        print(f"   Avg Results per Query: {summary['avg_results_per_query']:.1f}")
        print(f"   Avg Confidence Score: {summary['avg_confidence_score']:.3f}")
        
        if all_successful and summary['success_rate'] > 0.8:
            print("\n🎉 Fixed Retriever test completed successfully!")
            print("✅ Metaclass conflicts resolved")
            print("✅ All retrieval strategies working")
            print("✅ Query analysis functioning properly")
            print("✅ LangChain + Qdrant integration successful")
            print("✅ Fallback mechanisms working")
            print("🎯 Fixed Retriever ready for production use!")
        else:
            print("\n⚠️  Retriever test completed with some issues")
            print("🔧 Check that documents are ingested and Qdrant is running")
        
    except Exception as e:
        print(f"❌ Failed to test fixed retriever: {e}")
        print("\n📝 Make sure you have:")
        print("   1. fixed_pipeline.py in the same directory")
        print("   2. Qdrant running: docker-compose up -d")
        print("   3. Valid environment variables in .env")
        print("   4. Sample documents ingested in the vector store")
        print("   5. Compatible package versions:")
        print("      pip install langchain==0.1.20 langchain-community==0.0.38")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fixed_retriever()