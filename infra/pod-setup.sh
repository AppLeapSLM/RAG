#!/bin/bash
# AppLeap RAG — RunPod Startup Script
# Set as Docker Command: /workspace/start.sh
# Runs on every pod start. Installs deps, starts all services.

echo "[AppLeap] Installing system packages..."
apt-get update > /dev/null 2>&1
apt-get install -y postgresql postgresql-server-dev-all git make gcc zstd poppler-utils tesseract-ocr libmagic-dev pandoc unzip curl > /dev/null 2>&1

echo "[AppLeap] Starting PostgreSQL..."
service postgresql start
su - postgres -c "psql -c \"ALTER USER postgres PASSWORD 'postgres';\"" 2>/dev/null
su - postgres -c "psql -c 'CREATE DATABASE appleap_rag;'" 2>/dev/null

echo "[AppLeap] Installing pgvector..."
cd /tmp && git clone https://github.com/pgvector/pgvector.git 2>/dev/null
cd /tmp/pgvector && make > /dev/null 2>&1 && make install > /dev/null 2>&1
su - postgres -c "psql -d appleap_rag -c 'CREATE EXTENSION IF NOT EXISTS vector;'"

echo "[AppLeap] Downloading app code..."
if [ ! -d /workspace/appleap-rag ]; then
    cd /workspace
    curl -L https://github.com/AppLeapSLM/RAG/archive/refs/heads/main.zip -o rag.zip
    unzip -q rag.zip
    mv RAG-main appleap-rag
    rm rag.zip
fi

echo "[AppLeap] Installing Python dependencies..."
cd /workspace/appleap-rag/appleap-rag && pip install -e ".[parsing]" > /dev/null 2>&1

echo "[AppLeap] Installing Ollama..."
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh > /dev/null 2>&1
fi
export OLLAMA_MODELS=/workspace/ollama
mkdir -p /workspace/ollama
echo "[AppLeap] Starting Ollama..."
ollama serve &
sleep 5

echo "[AppLeap] Pulling models (skip if cached)..."
ollama pull phi4 2>/dev/null
ollama pull nomic-embed-text 2>/dev/null

echo "[AppLeap] Starting idle shutdown watchdog (30 min)..."
RUNPOD_API_KEY="${RUNPOD_API_KEY}"
POD_ID="${RUNPOD_POD_ID:-$(cat /etc/hostname)}"
IDLE_MINUTES=30
STAMP_FILE="/tmp/appleap-last-request"
touch "$STAMP_FILE"

(while true; do
    sleep 300
    if [ -f "$STAMP_FILE" ]; then
        LAST_MODIFIED=$(stat -c %Y "$STAMP_FILE")
        NOW=$(date +%s)
        AGE_MINUTES=$(( (NOW - LAST_MODIFIED) / 60 ))
        if [ "$AGE_MINUTES" -ge "$IDLE_MINUTES" ]; then
            echo "[AppLeap] Idle for ${AGE_MINUTES} minutes. Stopping pod."
            curl -s -H "Authorization: Bearer ${RUNPOD_API_KEY}" -H "Content-Type: application/json" -d "{\"query\":\"mutation { podStop(input: {podId: \\\"${POD_ID}\\\"}) { id } }\"}" https://api.runpod.io/graphql > /dev/null 2>&1
        fi
    fi
done) &

echo "[AppLeap] Starting server..."
cd /workspace/appleap-rag/appleap-rag && uvicorn backend.main:app --host 0.0.0.0 --port 8000
