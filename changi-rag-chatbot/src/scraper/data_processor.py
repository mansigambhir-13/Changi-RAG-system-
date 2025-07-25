"""
Data processing pipeline for Changi Airport scraped content
"""
import json
import re
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
import time

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class ProcessedChunk:
    """Data class for processed content chunks"""
    chunk_id: str
    source_url: str
    source_title: str
    content: str
    chunk_index: int
    chunk_hash: str
    metadata: Dict[str, Any]
    processed_at: str

class ContentCleaner:
    """Clean and normalize scraped content"""
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.ContentCleaner")
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""
            
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.,!?;:()\-"]', ' ', text)
        
        # Fix common issues
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([.!?])\s*([a-z])', r'\1 \2', text)  # Ensure space after sentence endings
        
        # Remove URLs and email addresses
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\S+@\S+\.\S+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'\+?\d{1,4}?[-.\s]?\(?\d{1,3}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}', '', text)
        
        # Clean up multiple spaces again
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def is_valid_content(self, text: str, min_length: int = 50) -> bool:
        """Check if content is valid and meaningful"""
        if not text or len(text.strip()) < min_length:
            return False
            
        # Check if content is mostly navigation/menu items
        common_nav_terms = ['home', 'about', 'contact', 'login', 'search', 'menu', 'navigation']
        words = text.lower().split()
        nav_word_count = sum(1 for word in words if word in common_nav_terms)
        
        if nav_word_count > len(words) * 0.3:  # More than 30% navigation words
            return False
            
        # Check if content has reasonable word distribution
        unique_words = set(words)
        if len(unique_words) < len(words) * 0.3:  # Less than 30% unique words
            return False
            
        return True

class ContentChunker:
    """Split content into chunks for embeddings"""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.logger = get_logger(f"{__name__}.ContentChunker")
        
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Find the end of this chunk
            end = start + self.chunk_size
            
            # If we're not at the end of the text, try to break at a sentence boundary
            if end < len(text):
                # Look for sentence endings within the last 200 characters
                search_start = max(end - 200, start)
                sentence_end = self._find_sentence_boundary(text, search_start, end)
                
                if sentence_end > start:
                    end = sentence_end
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            
            # Ensure we don't get stuck in infinite loop
            if start <= 0:
                start = end
                
        return chunks
    
    def _find_sentence_boundary(self, text: str, start: int, end: int) -> int:
        """Find the best place to break text at sentence boundary"""
        # Look for sentence endings
        sentence_endings = ['. ', '! ', '? ', '.\n', '!\n', '?\n']
        
        best_pos = end
        for ending in sentence_endings:
            pos = text.rfind(ending, start, end)
            if pos > start:
                return pos + len(ending)
        
        # If no sentence ending found, look for other break points
        break_points = [', ', '; ', ' - ', '\n', ' ']
        
        for break_point in break_points:
            pos = text.rfind(break_point, start, end)
            if pos > start:
                return pos + len(break_point)
        
        return end

class DataProcessor:
    """Main data processing pipeline"""
    
    def __init__(self):
        self.cleaner = ContentCleaner()
        self.chunker = ContentChunker()
        self.logger = get_logger(__name__)
        
    def process_scraped_file(self, input_file: str, output_file: str = None) -> str:
        """Process a single scraped data file"""
        input_path = Path(input_file)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        if not output_file:
            output_file = str(settings.PROCESSED_DATA_DIR / f"processed_{input_path.name}")
        
        # Ensure output directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        processed_chunks = []
        total_items = 0
        processed_items = 0
        
        self.logger.info(f"Processing file: {input_file}")
        
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        # Parse JSON line
                        item = json.loads(line.strip())
                        total_items += 1
                        
                        # Process the item
                        chunks = self._process_item(item)
                        processed_chunks.extend(chunks)
                        processed_items += 1
                        
                        if line_num % 10 == 0:
                            self.logger.info(f"Processed {line_num} items, generated {len(processed_chunks)} chunks")
                            
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Invalid JSON on line {line_num}: {e}")
                        continue
                    except Exception as e:
                        self.logger.error(f"Error processing line {line_num}: {e}")
                        continue
        
        except Exception as e:
            self.logger.error(f"Error reading file {input_file}: {e}")
            raise
        
        # Save processed chunks
        self._save_processed_chunks(processed_chunks, output_file)
        
        self.logger.info(f"Processing complete: {processed_items}/{total_items} items processed, {len(processed_chunks)} chunks generated")
        self.logger.info(f"Output saved to: {output_file}")
        
        return output_file
    
    def _process_item(self, item: Dict[str, Any]) -> List[ProcessedChunk]:
        """Process a single scraped item into chunks"""
        chunks = []
        
        # Extract basic information
        url = item.get('url', '')
        title = item.get('title', '')
        content = item.get('content', '')
        metadata = item.get('metadata', {})
        
        # Clean the content
        cleaned_content = self.cleaner.clean_text(content)
        
        # Validate content
        if not self.cleaner.is_valid_content(cleaned_content):
            self.logger.debug(f"Skipping invalid content from {url}")
            return chunks
        
        # Split into chunks
        text_chunks = self.chunker.chunk_text(cleaned_content)
        
        # Create ProcessedChunk objects
        for i, chunk_text in enumerate(text_chunks):
            # Generate unique chunk ID
            chunk_data = f"{url}_{i}_{chunk_text[:50]}"
            chunk_id = hashlib.md5(chunk_data.encode()).hexdigest()
            chunk_hash = hashlib.md5(chunk_text.encode()).hexdigest()
            
            # Enhanced metadata
            chunk_metadata = {
                **metadata,
                'chunk_length': len(chunk_text),
                'chunk_word_count': len(chunk_text.split()),
                'total_chunks': len(text_chunks),
                'original_content_length': len(content)
            }
            
            processed_chunk = ProcessedChunk(
                chunk_id=chunk_id,
                source_url=url,
                source_title=title,
                content=chunk_text,
                chunk_index=i,
                chunk_hash=chunk_hash,
                metadata=chunk_metadata,
                processed_at=time.strftime('%Y-%m-%d %H:%M:%S')
            )
            
            chunks.append(processed_chunk)
        
        return chunks
    
    def _save_processed_chunks(self, chunks: List[ProcessedChunk], output_file: str):
        """Save processed chunks to file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                json.dump(asdict(chunk), f, ensure_ascii=False)
                f.write('\n')
    
    def process_all_scraped_files(self) -> List[str]:
        """Process all scraped files in the raw data directory"""
        raw_data_dir = Path(settings.RAW_DATA_DIR)
        output_files = []
        
        # Find all scraped data files
        scraped_files = list(raw_data_dir.glob("*_scraped_data.jsonl"))
        
        if not scraped_files:
            self.logger.warning(f"No scraped data files found in {raw_data_dir}")
            return output_files
        
        for scraped_file in scraped_files:
            try:
                output_file = self.process_scraped_file(str(scraped_file))
                output_files.append(output_file)
            except Exception as e:
                self.logger.error(f"Failed to process {scraped_file}: {e}")
        
        return output_files
    
    def get_processing_stats(self, processed_file: str) -> Dict[str, Any]:
        """Get statistics about processed data"""
        if not Path(processed_file).exists():
            return {}
        
        stats = {
            'total_chunks': 0,
            'total_content_length': 0,
            'average_chunk_length': 0,
            'unique_sources': set(),
            'chunk_length_distribution': {'small': 0, 'medium': 0, 'large': 0}
        }
        
        with open(processed_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    chunk = json.loads(line.strip())
                    stats['total_chunks'] += 1
                    content_length = len(chunk['content'])
                    stats['total_content_length'] += content_length
                    stats['unique_sources'].add(chunk['source_url'])
                    
                    # Categorize chunk size
                    if content_length < 500:
                        stats['chunk_length_distribution']['small'] += 1
                    elif content_length < 1000:
                        stats['chunk_length_distribution']['medium'] += 1
                    else:
                        stats['chunk_length_distribution']['large'] += 1
                        
                except json.JSONDecodeError:
                    continue
        
        if stats['total_chunks'] > 0:
            stats['average_chunk_length'] = stats['total_content_length'] / stats['total_chunks']
        
        stats['unique_sources'] = len(stats['unique_sources'])
        
        return stats

def run_data_processing(input_file: str = None):
    """Run data processing from command line"""
    processor = DataProcessor()
    
    if input_file:
        # Process specific file
        if not Path(input_file).exists():
            print(f"Input file not found: {input_file}")
            return
            
        output_file = processor.process_scraped_file(input_file)
        stats = processor.get_processing_stats(output_file)
        
        print(f"✅ Processing completed!")
        print(f"📁 Output file: {output_file}")
        print(f"📊 Stats: {stats['total_chunks']} chunks from {stats['unique_sources']} sources")
        print(f"📏 Average chunk length: {stats['average_chunk_length']:.0f} characters")
        
    else:
        # Process all scraped files
        output_files = processor.process_all_scraped_files()
        
        if not output_files:
            print("❌ No scraped files found to process")
            return
        
        print(f"✅ Processing completed for {len(output_files)} files!")
        
        for output_file in output_files:
            stats = processor.get_processing_stats(output_file)
            print(f"\n📁 {Path(output_file).name}:")
            print(f"   📊 {stats['total_chunks']} chunks from {stats['unique_sources']} sources")
            print(f"   📏 Average chunk length: {stats['average_chunk_length']:.0f} characters")

if __name__ == "__main__":
    import sys
    
    input_file = sys.argv[1] if len(sys.argv) > 1 else None
    run_data_processing(input_file)