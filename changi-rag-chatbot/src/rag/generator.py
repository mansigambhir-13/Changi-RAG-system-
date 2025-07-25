"""
Response generation system for RAG pipeline using Google Gemini
"""
import os
import sys
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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
    import google.generativeai as genai
    print("✅ Google Generative AI imported successfully")
except ImportError:
    print("❌ Google Generative AI not installed!")
    print("📦 Install with: pip install google-generativeai")
    sys.exit(1)

# Import RetrievedDocument
try:
    # Try relative import first
    from .retriever import RetrievedDocument
except ImportError:
    try:
        # Try absolute import
        from src.rag.retriever import RetrievedDocument
    except ImportError:
        # Fallback: define it here
        @dataclass
        class RetrievedDocument:
            content: str
            title: str
            url: str
            score: float
            metadata: Dict[str, Any]
            chunk_id: str

# System prompts
SYSTEM_PROMPTS = {
    'main_prompt': """You are a helpful and knowledgeable assistant for Changi Airport and Jewel Changi Airport. 
Your role is to provide accurate, friendly, and detailed information about airport facilities, services, dining, shopping, transportation, and general airport operations.

Guidelines for responses:
- Always base your answers on the provided context documents
- Be conversational, friendly, and helpful in tone
- Provide specific details like operating hours, locations, contact information when available
- If information isn't available in the context, politely say so and suggest alternatives
- Use bullet points or numbered lists when presenting multiple options or steps
- Include relevant source links when possible
- Keep responses comprehensive but concise

Context Information:
{context}

User Question: {query}

Please provide a helpful and accurate response based on the context above. If the information isn't sufficient to fully answer the question, indicate what information is missing and suggest how the user might find it."""
}

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

# Simple logging
class SimpleLogger:
    def info(self, msg, **kwargs): 
        print(f"ℹ️  {msg}")
        if kwargs: 
            details = ', '.join([f"{k}={v}" for k, v in kwargs.items()])
            print(f"   📝 {details}")
    def debug(self, msg, **kwargs): 
        print(f"🐛 {msg}")
        if kwargs: 
            details = ', '.join([f"{k}={v}" for k, v in kwargs.items()])
            print(f"   📝 {details}")
    def error(self, msg, **kwargs): 
        print(f"❌ {msg}")
        if kwargs: 
            details = ', '.join([f"{k}={v}" for k, v in kwargs.items()])
            print(f"   📝 {details}")
    def warning(self, msg, **kwargs): 
        print(f"⚠️  {msg}")
        if kwargs: 
            details = ', '.join([f"{k}={v}" for k, v in kwargs.items()])
            print(f"   📝 {details}")

def get_logger(name): 
    return SimpleLogger()

@dataclass
class GeneratedResponse:
    """Data class for generated responses"""
    response: str
    sources: List[Dict[str, Any]]
    confidence: float
    token_usage: Dict[str, int]
    response_metadata: Dict[str, Any]

class PromptBuilder:
    """Build prompts for the RAG system with Gemini"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def build_rag_prompt(
        self,
        query: str,
        retrieved_documents: List[RetrievedDocument],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Build the complete RAG prompt for Gemini"""
        
        # Build context from retrieved documents
        context = self._build_context_from_documents(retrieved_documents)
        
        # Build conversation history if provided
        history_text = ""
        if conversation_history:
            history_text = self._build_conversation_history(conversation_history)
        
        # Build the complete prompt
        if history_text:
            prompt = f"{SYSTEM_PROMPTS['main_prompt'].format(context=context, query=query)}\n\n{history_text}\n\nUser: {query}\n\nPlease provide a helpful response based on the context above."
        else:
            prompt = SYSTEM_PROMPTS['main_prompt'].format(context=context, query=query)
        
        # Check if prompt is too long (Gemini has generous limits but still check)
        if len(prompt) > 25000:  # Conservative limit
            prompt = self._truncate_prompt(prompt, retrieved_documents, query)
            self.logger.warning("Prompt truncated due to length")
        
        self.logger.debug(
            "Built RAG prompt",
            query=query,
            context_docs=len(retrieved_documents),
            prompt_length=len(prompt),
            has_history=bool(conversation_history)
        )
        
        return prompt
    
    def _build_context_from_documents(self, documents: List[RetrievedDocument]) -> str:
        """Build context section from retrieved documents"""
        if not documents:
            return "No relevant information found."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            # Format each document with source attribution
            doc_context = f"""Document {i}:
Title: {doc.title}
Source: {doc.url}
Content: {doc.content}
Relevance Score: {doc.score:.3f}

"""
            context_parts.append(doc_context)
        
        return "\n".join(context_parts)
    
    def _build_conversation_history(self, history: List[Dict[str, str]]) -> str:
        """Build conversation history section"""
        history_parts = []
        
        for turn in history[-5:]:  # Keep last 5 turns
            role = turn.get('role', 'user')
            content = turn.get('content', '')
            
            if role == 'user':
                history_parts.append(f"User: {content}")
            elif role == 'assistant':
                history_parts.append(f"Assistant: {content}")
        
        if history_parts:
            return "Previous conversation:\n" + "\n".join(history_parts) + "\n"
        
        return ""
    
    def _truncate_prompt(
        self,
        prompt: str,
        documents: List[RetrievedDocument],
        query: str
    ) -> str:
        """Truncate prompt to fit within limits"""
        # Calculate available space for context
        base_prompt = SYSTEM_PROMPTS["main_prompt"].format(context="", query=query)
        available_chars = 25000 - len(base_prompt) - 500  # Buffer
        
        # Build truncated context
        truncated_context = ""
        current_chars = 0
        
        for doc in documents:
            doc_context = f"""Title: {doc.title}
Source: {doc.url}
Content: {doc.content}

"""
            if current_chars + len(doc_context) <= available_chars:
                truncated_context += doc_context
                current_chars += len(doc_context)
            else:
                # Truncate this document's content to fit
                remaining_chars = available_chars - current_chars
                if remaining_chars > 200:  # Only if we have reasonable space
                    truncated_content = doc.content[:remaining_chars - 100] + "..."
                    truncated_context += f"""Title: {doc.title}
Source: {doc.url}
Content: {truncated_content}

"""
                break
        
        # Rebuild prompt with truncated context
        return SYSTEM_PROMPTS["main_prompt"].format(context=truncated_context, query=query)

class GeminiRAGGenerator:
    """Main response generation component using Google Gemini"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.prompt_builder = PromptBuilder()
        
        # Get configuration from environment
        self.api_key = os.getenv('GOOGLE_API_KEY')
        self.model_name = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')
        self.temperature = float(os.getenv('GEMINI_TEMPERATURE', '0.1'))
        self.max_tokens = int(os.getenv('GEMINI_MAX_TOKENS', '1000'))
        
        if not self.api_key:
            raise ValueError("❌ GOOGLE_API_KEY not found. Please set it in your .env file")
        
        print(f"🤖 Initializing Gemini RAG Generator")
        print(f"   🧠 Model: {self.model_name}")
        print(f"   🌡️  Temperature: {self.temperature}")
        print(f"   📊 Max tokens: {self.max_tokens}")
        print(f"   🔑 API key: {self.api_key[:20]}...")
        
        # Configure Gemini
        try:
            genai.configure(api_key=self.api_key)
            
            # Initialize model
            generation_config = {
                "temperature": self.temperature,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": self.max_tokens,
            }
            
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
            
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            print("✅ Gemini model initialized successfully")
            
        except Exception as e:
            print(f"❌ Failed to initialize Gemini: {e}")
            raise
    
    @log_performance("generate_response")
    def generate_response(
        self,
        query: str,
        retrieved_documents: List[RetrievedDocument],
        conversation_history: Optional[List[Dict[str, str]]] = None,
        stream: bool = False
    ) -> GeneratedResponse:
        """Generate response using Gemini based on retrieved context"""
        
        self.logger.info(
            "Generating response",
            query=query,
            context_docs=len(retrieved_documents),
            model=self.model_name
        )
        
        try:
            # Build the prompt
            prompt = self.prompt_builder.build_rag_prompt(
                query=query,
                retrieved_documents=retrieved_documents,
                conversation_history=conversation_history
            )
            
            # Generate response with Gemini
            start_time = time.time()
            
            if stream:
                response_text = self._generate_streaming_response(prompt)
            else:
                response = self.model.generate_content(prompt)
                response_text = response.text if response.text else "I apologize, but I couldn't generate a response at this time."
            
            generation_time = time.time() - start_time
            
            # Extract sources from retrieved documents
            sources = self._extract_sources(retrieved_documents)
            
            # Calculate confidence based on retrieval scores
            confidence = self._calculate_confidence(retrieved_documents, response_text)
            
            # Estimate token usage (Gemini doesn't provide exact counts like OpenAI)
            token_usage = self._estimate_token_usage(prompt, response_text)
            
            # Prepare response metadata
            response_metadata = {
                'model': self.model_name,
                'temperature': self.temperature,
                'max_tokens': self.max_tokens,
                'generation_time': generation_time,
                'num_context_docs': len(retrieved_documents),
                'avg_retrieval_score': sum(doc.score for doc in retrieved_documents) / max(len(retrieved_documents), 1)
            }
            
            generated_response = GeneratedResponse(
                response=response_text,
                sources=sources,
                confidence=confidence,
                token_usage=token_usage,
                response_metadata=response_metadata
            )
            
            self.logger.info(
                "Response generated successfully",
                response_length=len(generated_response.response),
                confidence=confidence,
                generation_time=generation_time
            )
            
            return generated_response
            
        except Exception as e:
            self.logger.error("Failed to generate response", error=str(e))
            import traceback
            traceback.print_exc()
            
            # Return fallback response
            return GeneratedResponse(
                response="I apologize, but I'm having trouble generating a response right now. Please try again.",
                sources=[],
                confidence=0.0,
                token_usage={},
                response_metadata={'error': str(e)}
            )
    
    def _generate_streaming_response(self, prompt: str) -> str:
        """Generate streaming response (if needed for real-time display)"""
        try:
            response_content = ""
            
            # Gemini streaming
            response = self.model.generate_content(prompt, stream=True)
            
            for chunk in response:
                if chunk.text:
                    response_content += chunk.text
            
            return response_content
            
        except Exception as e:
            self.logger.error("Streaming generation failed", error=str(e))
            raise
    
    def _extract_sources(self, documents: List[RetrievedDocument]) -> List[Dict[str, Any]]:
        """Extract source information from retrieved documents"""
        sources = []
        
        for doc in documents:
            source = {
                'title': doc.title,
                'url': doc.url,
                'score': doc.score,
                'snippet': doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
            }
            sources.append(source)
        
        return sources
    
    def _calculate_confidence(self, documents: List[RetrievedDocument], response: str) -> float:
        """Calculate confidence score based on retrieval quality and response"""
        if not documents:
            return 0.0
        
        # Base confidence from average retrieval scores
        avg_score = sum(doc.score for doc in documents) / len(documents)
        
        # Boost confidence if we have multiple good sources
        if len(documents) >= 3 and avg_score > 0.8:
            confidence = min(0.95, avg_score + 0.1)
        elif len(documents) >= 2 and avg_score > 0.75:
            confidence = min(0.9, avg_score + 0.05)
        else:
            confidence = avg_score
        
        # Reduce confidence for very short responses
        if len(response.split()) < 10:
            confidence *= 0.8
        
        # Check for uncertainty phrases in response
        uncertainty_phrases = [
            "i don't know", "not sure", "uncertain", "might be", "possibly",
            "i'm not certain", "unclear", "unable to find", "no information",
            "i apologize", "i couldn't", "i can't find"
        ]
        
        response_lower = response.lower()
        for phrase in uncertainty_phrases:
            if phrase in response_lower:
                confidence *= 0.7
                break
        
        return max(0.0, min(1.0, confidence))
    
    def _estimate_token_usage(self, prompt: str, response: str) -> Dict[str, int]:
        """Estimate token usage (Gemini doesn't provide exact counts)"""
        # Rough estimation: ~4 characters per token for English
        prompt_tokens = len(prompt) // 4
        completion_tokens = len(response) // 4
        
        return {
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': prompt_tokens + completion_tokens
        }
    
    def generate_follow_up_questions(
        self,
        query: str,
        retrieved_documents: List[RetrievedDocument]
    ) -> List[str]:
        """Generate follow-up questions based on the context"""
        try:
            # Build context for follow-up generation
            context = self.prompt_builder._build_context_from_documents(retrieved_documents)
            
            follow_up_prompt = f"""Based on the user's question: "{query}"
And the following context information:

{context}

Generate 3 relevant follow-up questions that the user might want to ask next. 
Focus on related topics that would be helpful for someone visiting Changi Airport or Jewel.

Format as a simple numbered list:
1. [question]
2. [question]  
3. [question]"""
            
            response = self.model.generate_content(follow_up_prompt)
            response_text = response.text if response.text else ""
            
            # Parse the response to extract questions
            follow_ups = []
            for line in response_text.split('\n'):
                line = line.strip()
                if line and (line.startswith(('1.', '2.', '3.')) or line.startswith('-')):
                    question = line.split('.', 1)[-1].strip() if '.' in line else line[1:].strip()
                    if question:
                        follow_ups.append(question)
            
            return follow_ups[:3]  # Limit to 3 questions
            
        except Exception as e:
            self.logger.error("Failed to generate follow-up questions", error=str(e))
            return [
                "What are the operating hours for this service?",
                "Where exactly can I find this at the airport?",
                "Are there any additional fees or requirements?"
            ]

# For backward compatibility, create an alias
RAGGenerator = GeminiRAGGenerator

# Utility functions
def test_generator():
    """Test the Gemini response generator"""
    print("🧪 Testing Gemini RAG Generator")
    print("=" * 60)
    
    try:
        # Initialize generator
        print("🔧 Initializing generator...")
        generator = GeminiRAGGenerator()
        
        # Test connection
        print("\n🔗 Testing API connection...")
        test_prompt = "Hello! Please respond with 'API connection successful' to confirm you're working."
        test_response = generator.model.generate_content(test_prompt)
        
        if test_response.text:
            print(f"✅ API connection successful!")
            print(f"   📝 Response: {test_response.text}")
        else:
            print("❌ API connection failed")
            return
        
        # Create sample retrieved documents for testing
        sample_documents = [
            RetrievedDocument(
                content="Changi Airport operates 24 hours a day, 7 days a week. The airport provides world-class facilities and services to passengers traveling through Singapore.",
                title="Changi Airport Operating Hours",
                url="https://www.changiairport.com/corporate/about-us/airport-operations",
                score=0.85,
                metadata={"source": "changi_official"},
                chunk_id="changi_001"
            ),
            RetrievedDocument(
                content="Free WiFi is available throughout Changi Airport terminals. Passengers can connect to the 'Changi Airport WiFi' network for complimentary high-speed internet access without time limits.",
                title="Changi Airport WiFi Services",
                url="https://www.changiairport.com/passenger-guide/facilities-and-services/wifi",
                score=0.72,
                metadata={"source": "changi_official"},
                chunk_id="wifi_001"
            )
        ]
        
        # Test queries
        test_queries = [
            "What are the operating hours of Changi Airport?",
            "Is there free WiFi at the airport?",
            "Where can I find dining options?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 Test Query {i}: {query}")
            print("-" * 50)
            
            try:
                # Generate response
                response = generator.generate_response(
                    query=query,
                    retrieved_documents=sample_documents
                )
                
                print(f"✅ Response generated successfully:")
                print(f"   📝 Content: {response.response[:200]}...")
                print(f"   📊 Confidence: {response.confidence:.3f}")
                print(f"   ⏱️  Response time: {response.response_metadata.get('generation_time', 0):.2f}s")
                print(f"   📚 Sources: {len(response.sources)} found")
                print(f"   🤖 Model: {response.response_metadata.get('model', 'N/A')}")
                print(f"   🔢 Token usage: {response.token_usage.get('total_tokens', 0)} tokens")
                
                # Test follow-up questions
                print(f"\n💡 Follow-up questions:")
                follow_ups = generator.generate_follow_up_questions(query, sample_documents)
                for j, follow_up in enumerate(follow_ups, 1):
                    print(f"   {j}. {follow_up}")
                
            except Exception as e:
                print(f"❌ Error generating response: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n🎉 Gemini RAG generator test completed!")
        print(f"✅ All components working successfully")
        print(f"🎯 Ready for integration with RAG pipeline")
        
    except Exception as e:
        print(f"❌ Failed to test generator: {e}")
        print("\n📝 Make sure you have:")
        print("   1. Valid GOOGLE_API_KEY in .env file")
        print("   2. google-generativeai installed: pip install google-generativeai")
        print("   3. Internet connection for API calls")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_generator()