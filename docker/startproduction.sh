#!/bin/bash
# docker/start-production.sh

set -e

echo "Starting Changi Airport RAG Chatbot in production mode..."

# Set defaults
export WORKERS=${WORKERS:-4}
export HOST=${HOST:-0.0.0.0}
export PORT=${PORT:-8000}
export TIMEOUT=${TIMEOUT:-120}
export KEEPALIVE=${KEEPALIVE:-2}

# Validate required environment variables
required_vars=("OPENAI_API_KEY" "PINECONE_API_KEY" "API_TOKEN")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "Error: Required environment variable $var is not set"
        exit 1
    fi
done

# Create log directory if it doesn't exist
mkdir -p /app/logs

# Wait for dependencies (Redis, etc.)
echo "Waiting for dependencies..."
python -c "
import time
import redis
import os

# Wait for Redis
redis_host = os.getenv('REDIS_HOST', 'redis')
redis_port = int(os.getenv('REDIS_PORT', 6379))

for i in range(30):
    try:
        r = redis.Redis(host=redis_host, port=redis_port, socket_timeout=2)
        r.ping()
        print('Redis is ready!')
        break
    except:
        print(f'Waiting for Redis... ({i+1}/30)')
        time.sleep(2)
else:
    print('Warning: Redis connection timeout, continuing without Redis')
"

# Initialize the application
echo "Initializing RAG pipeline..."
python -c "
import asyncio
from src.rag.pipeline import RAGPipeline

async def init():
    try:
        pipeline = RAGPipeline()
        await pipeline.initialize()
        print('RAG pipeline initialized successfully!')
    except Exception as e:
        print(f'Warning: RAG pipeline initialization failed: {e}')

asyncio.run(init())
"

# Start the application with Gunicorn
echo "Starting application with $WORKERS workers..."
exec gunicorn src.api.main:app \
    --bind $HOST:$PORT \
    --workers $WORKERS \
    --worker-class uvicorn.workers.UvicornWorker \
    --timeout $TIMEOUT \
    --keepalive $KEEPALIVE \
    --max-requests 1000 \
    --max-requests-jitter 100 \
    --preload \
    --access-logfile /app/logs/access.log \
    --error-logfile /app/logs/error.log \
    --log-level info \
    --capture-output