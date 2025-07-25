"""
Enhanced web scraper for comprehensive Changi Airport content
"""
import requests
from bs4 import BeautifulSoup
import json
import time
import hashlib
from pathlib import Path
from urllib.parse import urljoin, urlparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

class EnhancedChangiScraper:
    """Enhanced scraper with more comprehensive content extraction"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.scraped_urls = set()
        self.scraped_data = []
        
    def scrape_url(self, url, max_retries=3):
        """Scrape a single URL with retry logic"""
        for attempt in range(max_retries):
            try:
                logger.info(f"Scraping: {url} (attempt {attempt + 1})")
                
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract content
                content_data = self.extract_content(soup, url)
                if content_data:
                    self.scraped_data.append(content_data)
                    logger.info(f"✅ Successfully scraped: {content_data['title']}")
                    return content_data
                else:
                    logger.warning(f"No content extracted from {url}")
                    
            except Exception as e:
                logger.error(f"Error scraping {url} (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    
        return None
    
    def extract_content(self, soup, url):
        """Extract comprehensive content from a page"""
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            element.decompose()
        
        # Extract title
        title = self.extract_title(soup)
        if not title:
            return None
            
        # Extract main content
        content = self.extract_main_content(soup)
        if not content or len(content.strip()) < 100:
            return None
        
        # Generate content hash
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        # Extract metadata
        metadata = self.extract_metadata(soup, url)
        
        return {
            'url': url,
            'title': title,
            'content': content,
            'metadata': metadata,
            'scraped_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'content_hash': content_hash
        }
    
    def extract_title(self, soup):
        """Extract page title with multiple fallbacks"""
        # Try different title sources
        selectors = [
            'h1.page-title',
            'h1.main-title', 
            'h1',
            'title',
            '.page-header h1',
            '.content-header h1'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                title = element.get_text().strip()
                if title and len(title) > 3:
                    return title
        
        return None
    
    def extract_main_content(self, soup):
        """Extract main content with better text extraction"""
        # Remove unwanted sections
        unwanted = [
            '.advertisement', '.ad', '.sidebar', '.menu', '.navigation',
            '.footer', '.header', '.breadcrumb', '.social-share',
            '.related-articles', '.comments', '.cookie-notice',
            '.newsletter', '.subscription', '.popup'
        ]
        
        for selector in unwanted:
            for element in soup.select(selector):
                element.decompose()
        
        # Try main content selectors
        main_selectors = [
            'main',
            '.main-content',
            '.content',
            '.article-content',
            '.page-content',
            '.entry-content',
            '#content',
            '.container .content'
        ]
        
        content_element = None
        for selector in main_selectors:
            element = soup.select_one(selector)
            if element:
                content_element = element
                break
        
        if not content_element:
            content_element = soup.find('body')
        
        if not content_element:
            return ""
        
        # Extract text content more intelligently
        text_parts = []
        
        # Get paragraphs and headings
        for element in content_element.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']):
            text = element.get_text().strip()
            if text and len(text) > 10:  # Filter out very short text
                text_parts.append(text)
        
        # If no structured content found, fall back to all text
        if not text_parts:
            text_parts = [content_element.get_text()]
        
        # Join and clean
        content = ' '.join(text_parts)
        content = ' '.join(content.split())  # Normalize whitespace
        
        return content.strip()
    
    def extract_metadata(self, soup, url):
        """Extract comprehensive metadata"""
        metadata = {
            'domain': urlparse(url).netloc,
            'path': urlparse(url).path,
        }
        
        # Meta tags
        meta_tags = {}
        for meta in soup.find_all('meta'):
            name = meta.get('name') or meta.get('property') or meta.get('itemprop')
            content = meta.get('content')
            if name and content:
                meta_tags[name] = content
        
        if meta_tags:
            metadata['meta_tags'] = meta_tags
        
        # Extract headings
        headings = []
        for i in range(1, 4):  # h1, h2, h3
            for heading in soup.find_all(f'h{i}'):
                text = heading.get_text().strip()
                if text:
                    headings.append({'level': i, 'text': text})
        
        if headings:
            metadata['headings'] = headings[:10]  # Limit to 10 headings
        
        return metadata
    
    def scrape_changi_comprehensive(self):
        """Scrape comprehensive Changi Airport content"""
        
        # Main Changi Airport URLs
        changi_urls = [
            'https://www.changiairport.com/',
            'https://www.changiairport.com/en/airport-guide.html',
            'https://www.changiairport.com/en/shop.html',
            'https://www.changiairport.com/en/dine.html',
            'https://www.changiairport.com/en/experience.html',
            'https://www.changiairport.com/en/airport-guide/arrival-guide.html',
            'https://www.changiairport.com/en/airport-guide/departure-guide.html',
            'https://www.changiairport.com/en/airport-guide/transit-guide.html',
            'https://www.changiairport.com/en/discover/attractions.html',
            'https://www.changiairport.com/en/discover/services.html'
        ]
        
        # Jewel Changi URLs
        jewel_urls = [
            'https://www.jewelchangiairport.com/',
            'https://www.jewelchangiairport.com/en/shop.html',
            'https://www.jewelchangiairport.com/en/dine.html',
            'https://www.jewelchangiairport.com/en/attractions.html',
            'https://www.jewelchangiairport.com/en/attractions/rain-vortex.html',
            'https://www.jewelchangiairport.com/en/attractions/forest-valley.html'
        ]
        
        all_urls = changi_urls + jewel_urls
        
        logger.info(f"Starting comprehensive scraping of {len(all_urls)} URLs")
        
        for i, url in enumerate(all_urls, 1):
            if url not in self.scraped_urls:
                logger.info(f"Progress: {i}/{len(all_urls)} - {url}")
                self.scrape_url(url)
                self.scraped_urls.add(url)
                time.sleep(1)  # Be respectful to the server
        
        logger.info(f"Scraping complete! Collected {len(self.scraped_data)} pages")
        return self.scraped_data
    
    def save_data(self, filename=None):
        """Save scraped data to file"""
        if not filename:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f'data/raw/changi_comprehensive_{timestamp}.jsonl'
        
        # Ensure directory exists
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            for item in self.scraped_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"Data saved to: {filename}")
        return filename

def run_comprehensive_scraping():
    """Run comprehensive scraping"""
    scraper = EnhancedChangiScraper()
    
    try:
        # Scrape content
        data = scraper.scrape_changi_comprehensive()
        
        if not data:
            print("❌ No data scraped")
            return None
        
        # Save data
        filename = scraper.save_data()
        
        print(f"✅ Comprehensive scraping completed!")
        print(f"📊 Scraped {len(data)} pages")
        print(f"📁 Saved to: {filename}")
        
        # Show sample of what was scraped
        print(f"\n📄 Sample content:")
        for i, item in enumerate(data[:3]):
            print(f"  {i+1}. {item['title']} ({len(item['content'])} chars)")
        
        return filename
        
    except Exception as e:
        logger.error(f"Error in comprehensive scraping: {e}")
        print(f"❌ Scraping failed: {e}")
        return None

if __name__ == "__main__":
    run_comprehensive_scraping()