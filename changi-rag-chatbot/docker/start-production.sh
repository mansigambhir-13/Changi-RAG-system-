#!/bin/bash

# Production startup script for Changi RAG Chatbot

set -e

echo "🚀 Starting Changi RAG Chatbot in Production Mode..."

# Wait for dependencies
echo "⏳ Waiting for dependencies..."

# Wait for Redis
if [ ! -z "$REDIS_URL" ]; then
    echo "Waiting for Redis..."
    timeout 30s bash -c 'until curl -f $REDIS_URL > /dev/null 2>&1; do sleep 1; done' || {
        echo "❌ Redis not available"
        exit 1
    }
    echo "✅ Redis is ready"
fi

# Initialize data if needed
echo "📊 Checking data initialization..."
if [ ! -f "/app/data/.initialized" ]; then
    echo "🔄 Running initial data setup..."
    python -c "
from src.scraper.changi_scraper import ChangiScraper
from src.embeddings.embedding_generator import EmbeddingGenerator
import asyncio

async def initialize_data():
    # Run scraper
    scraper = ChangiScraper()
    await scraper.scrape_and_process()
    
    # Generate embeddings
    embedder = EmbeddingGenerator()
    await embedder.process_all_documents()
    
    # Mark as initialized
    with open('/app/data/.initialized', 'w') as f:
        f.write('initialized')

asyncio.run(initialize_data())
"
    echo "✅ Data initialization complete"
fi

# Start the application
echo "🌐 Starting API server..."

if [ "$ENVIRONMENT" = "production" ]; then
    # Production: Use Gunicorn with Uvicorn workers
    exec gunicorn src.api.main:app \
        --bind 0.0.0.0:8000 \
        --workers $WORKERS \
        --worker-class uvicorn.workers.UvicornWorker \
        --access-logfile - \
        --error-logfile - \
        --log-level info \
        --preload \
        --max-requests 1000 \
        --max-requests-jitter 50 \
        --timeout 120
else
    # Development: Use Uvicorn directly
    exec uvicorn src.api.main:app \
        --host 0.0.0.0 \
        --port 8000 \
        --reload
fi