import json
import os

# Ensure directory exists
os.makedirs('data/raw', exist_ok=True)

# Test data
test_data = [
    {
        "url": "https://www.changiairport.com/",
        "title": "Changi Airport Main", 
        "content": "Changi Airport is Singapore main international airport and one of the worlds busiest and most awarded airports. Located in the eastern part of Singapore, it serves as a major aviation hub for Southeast Asia and connects Singapore to over 380 cities worldwide. The airport features multiple terminals with extensive shopping, dining, and entertainment options. Terminal 3 houses the famous Butterfly Garden and Terminal 1 features a rooftop swimming pool.",
        "metadata": {"domain": "changiairport.com", "site_name": "changi_airport"},
        "scraped_at": "2025-07-24 08:45:00",
        "content_hash": "abc123"
    },
    {
        "url": "https://www.changiairport.com/shop",
        "title": "Shopping at Changi",
        "content": "Changi Airport offers world-class shopping experiences across all terminals. From luxury brands to local Singapore products, passengers can find everything they need. Popular shopping areas include duty-free stores, fashion boutiques, electronics stores, and souvenir shops. Terminal 2 features the largest collection of duty-free liquor and tobacco products.",
        "metadata": {"domain": "changiairport.com", "site_name": "changi_airport"},
        "scraped_at": "2025-07-24 08:45:01", 
        "content_hash": "def456"
    },
    {
        "url": "https://www.changiairport.com/dine",
        "title": "Dining at Changi",
        "content": "Changi Airport provides an exceptional dining experience with over 200 food and beverage outlets across all terminals. From local Singapore hawker fare to international fine dining, there is something for every palate and budget. Popular local dishes include Hainanese chicken rice, laksa, satay, and bak kut teh.",
        "metadata": {"domain": "changiairport.com", "site_name": "changi_airport"},
        "scraped_at": "2025-07-24 08:45:02",
        "content_hash": "ghi789"
    }
]

# Write to JSONL file
with open('data/raw/changi_airport_scraped_data.jsonl', 'w', encoding='utf-8') as f:
    for item in test_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print("✅ Test file created successfully!")
print("📁 File location: data/raw/changi_airport_scraped_data.jsonl")
