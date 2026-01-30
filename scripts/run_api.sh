#!/bin/bash

# Run the FastAPI server

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Set defaults
API_HOST=${API_HOST:-0.0.0.0}
API_PORT=${API_PORT:-8000}

echo "Starting RAG Screenplay API..."
echo "Host: $API_HOST"
echo "Port: $API_PORT"
echo "Docs: http://$API_HOST:$API_PORT/docs"

# Run uvicorn
python -m uvicorn src.presentation.api:app \
    --host $API_HOST \
    --port $API_PORT \
    --reload \
    --log-level info
