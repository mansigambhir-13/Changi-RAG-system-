"""
Offline Smart Chunker - No API key required
Uses intelligent heuristics to create semantic chunks
"""
import json
import re
import hashlib
from typing import List, Dict, Any
from pathlib import Path
import time
from dataclasses import dataclass, asdict

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class SmartChunk:
    """Data class for smart chunks"""
    chunk_id: str
    content: str
    topic_summary: str
    semantic_type: str
    source_url: str
    source_title: str
    chunk_index: int
    total_chunks: int
    metadata: Dict[str, Any]
    created_at: str
    chunk_hash: str

class OfflineSmartChunker:
    """Smart chunker using intelligent heuristics (no API required)"""
    
    def __init__(self, 
                 target_chunk_size: int = 800,
                 max_chunk_size: int = 1500,
                 min_chunk_size: int = 200):
        self.target_chunk_size = target_chunk_size
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.logger = get_logger(__name__)
        
        # Topic keywords for semantic classification
        self.topic_keywords = {
            'introduction': ['welcome', 'hello', 'introduction', 'overview', 'about', 'discover'],
            'attractions': ['attraction', 'garden', 'waterfall', 'rain vortex', 'forest valley', 'canopy park', 'experience', 'adventure'],
            'shopping': ['shop', 'shopping', 'retail', 'store', 'boutique', 'brands', 'merchandise', 'purchase'],
            'dining': ['dine', 'dining', 'restaurant', 'food', 'eat', 'cuisine', 'meal', 'cafe'],
            'services': ['service', 'facility', 'information', 'guide', 'assistance', 'help', 'support'],
            'transportation': ['transport', 'getting to', 'travel', 'terminal', 'airport', 'train', 'bus'],
            'entertainment': ['entertainment', 'show', 'event', 'performance', 'music', 'light show'],
            'accommodation': ['hotel', 'stay', 'room', 'accommodation', 'yotelair', 'rest'],
            'promotions': ['promotion', 'deal', 'offer', 'discount', 'campaign', 'win', 'prize'],
            'nature': ['nature', 'plant', 'tree', 'garden', 'green', 'natural', 'botanical']
        }
    
    def _classify_content_type(self, content: str) -> str:
        """Classify content into semantic type based on keywords"""
        content_lower = content.lower()
        
        type_scores = {}
        for content_type, keywords in self.topic_keywords.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            if score > 0:
                type_scores[content_type] = score
        
        if type_scores:
            return max(type_scores, key=type_scores.get)
        return 'general'
    
    def _generate_topic_summary(self, content: str, semantic_type: str) -> str:
        """Generate a topic summary based on content analysis"""
        content_lower = content.lower()
        
        # Extract key phrases based on semantic type
        if semantic_type == 'attractions':
            if 'rain vortex' in content_lower:
                return "Rain Vortex waterfall attraction and viewing experience"
            elif 'forest valley' in content_lower:
                return "Forest Valley garden and walking trails information"
            elif 'canopy park' in content_lower:
                return "Canopy Park recreational activities and attractions"
            else:
                return "Airport attractions and entertainment experiences"
        
        elif semantic_type == 'shopping':
            if 'jewel' in content_lower:
                return "Shopping experience and retail options at Jewel"
            else:
                return "Shopping facilities and retail opportunities"
        
        elif semantic_type == 'dining':
            if 'jewel' in content_lower:
                return "Dining options and restaurants at Jewel"
            else:
                return "Food and beverage facilities at the airport"
        
        elif semantic_type == 'services':
            return "Airport services, facilities and passenger information"
        
        elif semantic_type == 'transportation':
            return "Transportation options and terminal navigation"
        
        elif semantic_type == 'introduction':
            if 'jewel' in content_lower:
                return "Introduction to Jewel Changi Airport"
            else:
                return "Welcome and overview of Changi Airport"
        
        elif semantic_type == 'promotions':
            return "Current promotions, deals and special offers"
        
        elif semantic_type == 'nature':
            return "Natural attractions and green spaces"
        
        else:
            return f"General information about {semantic_type}"
    
    def _smart_split_text(self, text: str) -> List[str]:
        """Intelligently split text into semantic chunks"""
        chunks = []
        
        # First, try to split by strong semantic boundaries
        # Look for section headers, bullet points, or major topic changes
        
        # Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n', text)
        
        current_chunk = ""
        current_topic = None
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
            
            # Detect topic changes
            paragraph_type = self._classify_content_type(paragraph)
            
            # If we're switching topics and current chunk is substantial
            if (current_topic and 
                current_topic != paragraph_type and 
                len(current_chunk) > self.min_chunk_size):
                
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
                current_topic = paragraph_type
            
            # If adding this paragraph would exceed max size
            elif len(current_chunk) + len(paragraph) > self.max_chunk_size:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph
                current_topic = paragraph_type
            
            # If we're close to target size and at a good break point
            elif (len(current_chunk) + len(paragraph) > self.target_chunk_size and
                  len(current_chunk) > self.min_chunk_size):
                
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
                current_topic = paragraph_type
            
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                if not current_topic:
                    current_topic = paragraph_type
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # If no good splits found, fall back to sentence-based splitting
        if not chunks or len(chunks) == 1:
            return self._sentence_based_split(text)
        
        return chunks
    
    def _sentence_based_split(self, text: str) -> List[str]:
        """Fallback: split by sentences when paragraph splitting fails"""
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > self.target_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def chunk_document(self, document: Dict[str, Any]) -> List[SmartChunk]:
        """Chunk a document using offline intelligence"""
        content = document.get('content', '')
        url = document.get('url', '')
        title = document.get('title', '')
        metadata = document.get('metadata', {})
        
        if not content or len(content.strip()) < self.min_chunk_size:
            self.logger.warning(f"Insufficient content for document: {url}")
            return []
        
        self.logger.info(f"Smart chunking document: {title} ({len(content)} chars)")
        
        # Split content intelligently
        chunk_texts = self._smart_split_text(content)
        
        # Create SmartChunk objects
        smart_chunks = []
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        
        for i, chunk_text in enumerate(chunk_texts):
            if len(chunk_text) < self.min_chunk_size:
                continue
            
            # Classify and summarize
            semantic_type = self._classify_content_type(chunk_text)
            topic_summary = self._generate_topic_summary(chunk_text, semantic_type)
            
            # Generate unique chunk ID
            chunk_id_data = f"{url}_{i}_{chunk_text[:50]}"
            chunk_id = hashlib.md5(chunk_id_data.encode()).hexdigest()
            chunk_hash = hashlib.md5(chunk_text.encode()).hexdigest()
            
            # Enhanced metadata
            enhanced_metadata = {
                **metadata,
                'offline_chunking': True,
                'chunking_method': 'semantic_heuristics',
                'original_content_length': len(content),
                'chunk_character_count': len(chunk_text),
                'chunk_word_count': len(chunk_text.split())
            }
            
            smart_chunk = SmartChunk(
                chunk_id=chunk_id,
                content=chunk_text,
                topic_summary=topic_summary,
                semantic_type=semantic_type,
                source_url=url,
                source_title=title,
                chunk_index=i,
                total_chunks=len(chunk_texts),
                metadata=enhanced_metadata,
                created_at=timestamp,
                chunk_hash=chunk_hash
            )
            
            smart_chunks.append(smart_chunk)
        
        self.logger.info(f"Generated {len(smart_chunks)} smart chunks for: {title}")
        return smart_chunks
    
    def process_file(self, input_file: str, output_file: str = None) -> str:
        """Process a file using offline smart chunking"""
        if not Path(input_file).exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        if not output_file:
            input_path = Path(input_file)
            output_file = f"data/processed/offline_smart_chunks_{input_path.stem}.jsonl"
        
        # Ensure output directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Starting offline smart chunking: {input_file}")
        
        all_chunks = []
        total_documents = 0
        successful_documents = 0
        
        # Process each document
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    document = json.loads(line.strip())
                    total_documents += 1
                    
                    self.logger.info(f"Processing document {line_num}: {document.get('title', 'Unknown')}")
                    
                    chunks = self.chunk_document(document)
                    all_chunks.extend(chunks)
                    
                    if chunks:
                        successful_documents += 1
                
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Invalid JSON on line {line_num}: {e}")
                    continue
                except Exception as e:
                    self.logger.error(f"Error processing document {line_num}: {e}")
                    continue
        
        # Save chunks
        with open(output_file, 'w', encoding='utf-8') as f:
            for chunk in all_chunks:
                chunk_dict = asdict(chunk)
                f.write(json.dumps(chunk_dict, ensure_ascii=False) + '\n')
        
        # Generate statistics
        stats = self._generate_stats(all_chunks, total_documents, successful_documents)
        stats_file = output_file.replace('.jsonl', '_stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"✅ Offline smart chunking completed!")
        self.logger.info(f"📊 {len(all_chunks)} chunks from {successful_documents}/{total_documents} documents")
        self.logger.info(f"📁 Output: {output_file}")
        
        return output_file
    
    def _generate_stats(self, chunks: List[SmartChunk], total_docs: int, successful_docs: int) -> Dict[str, Any]:
        """Generate statistics"""
        if not chunks:
            return {}
        
        chunk_sizes = [len(chunk.content) for chunk in chunks]
        semantic_types = [chunk.semantic_type for chunk in chunks]
        
        # Count semantic types
        type_counts = {}
        for semantic_type in semantic_types:
            type_counts[semantic_type] = type_counts.get(semantic_type, 0) + 1
        
        stats = {
            'total_chunks': len(chunks),
            'unique_sources': len(set(chunk.source_url for chunk in chunks)),
            'documents_processed': successful_docs,
            'total_documents': total_docs,
            'success_rate': successful_docs / total_docs if total_docs > 0 else 0,
            'chunking_method': 'offline_semantic_heuristics',
            'chunk_size_stats': {
                'min_size': min(chunk_sizes),
                'max_size': max(chunk_sizes),
                'avg_size': sum(chunk_sizes) / len(chunk_sizes),
                'target_size': self.target_chunk_size
            },
            'semantic_type_distribution': type_counts,
            'content_stats': {
                'total_characters': sum(chunk_sizes),
                'avg_chunks_per_source': len(chunks) / len(set(chunk.source_url for chunk in chunks)),
                'topics_identified': len(set(chunk.topic_summary for chunk in chunks))
            }
        }
        
        return stats

def run_offline_smart_chunker(target_size: int = 800):
    """Run offline smart chunker"""
    print(f"🧠 Offline Smart Chunker (No API Required)")
    print(f"📏 Target chunk size: {target_size} characters")
    print(f"🔑 API Key: Not needed ✅")
    print()
    
    # Auto-detect input file
    print("🔍 Auto-detecting scraped data file...")
    
    raw_data_dir = Path("data/raw")
    if raw_data_dir.exists():
        # Find comprehensive files first
        comprehensive_files = list(raw_data_dir.glob("changi_comprehensive_*.jsonl"))
        if comprehensive_files:
            input_file = str(comprehensive_files[0])
            print(f"✅ Found comprehensive file: {input_file}")
        else:
            # Fallback to any scraped files
            scraped_files = list(raw_data_dir.glob("*scraped*.jsonl"))
            if scraped_files:
                input_file = str(scraped_files[0])
                print(f"✅ Found scraped file: {input_file}")
            else:
                print("❌ No scraped data files found")
                return
    else:
        print("❌ data/raw directory not found")
        return
    
    try:
        # Initialize chunker
        chunker = OfflineSmartChunker(target_chunk_size=target_size)
        
        # Process file
        output_file = chunker.process_file(input_file)
        
        # Show results
        stats_file = output_file.replace('.jsonl', '_stats.json')
        if Path(stats_file).exists():
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            
            print(f"✅ Offline smart chunking completed!")
            print(f"📁 Output: {output_file}")
            print(f"📊 Generated {stats['total_chunks']} smart chunks")
            print(f"📄 From {stats['documents_processed']}/{stats['total_documents']} documents")
            print(f"📏 Average chunk size: {stats['chunk_size_stats']['avg_size']:.0f} chars")
            print(f"🎯 Topics identified: {stats['content_stats']['topics_identified']}")
            print(f"📈 Semantic types: {', '.join(stats['semantic_type_distribution'].keys())}")
        
    except Exception as e:
        print(f"❌ Error during chunking: {e}")

if __name__ == "__main__":
    import sys
    
    target_size = int(sys.argv[1]) if len(sys.argv) > 1 else 800
    run_offline_smart_chunker(target_size)