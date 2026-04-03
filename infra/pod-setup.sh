#!/bin/bash
# AppLeap RAG — RunPod Pod Setup Script
# Run this once on a fresh pod to install all dependencies.
# Template: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

set -e

echo "=== System packages ==="
apt-get update
apt-get install -y \
    postgresql \
    postgresql-server-dev-all \
    git \
    make \
    gcc \
    zstd \
    poppler-utils \
    tesseract-ocr \
    libmagic-dev \
    pandoc

echo "=== PostgreSQL setup ==="
service postgresql start
su - postgres -c "psql -c \"ALTER USER postgres PASSWORD 'postgres';\""
su - postgres -c "psql -c 'CREATE DATABASE appleap_rag;'"

echo "=== pgvector ==="
cd /tmp && git clone https://github.com/pgvector/pgvector.git && cd pgvector && make && make install
su - postgres -c "psql -d appleap_rag -c 'CREATE EXTENSION vector;'"

echo "=== Ollama ==="
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
sleep 5
ollama pull phi4
ollama pull nomic-embed-text

echo "=== App dependencies ==="
cd ~/appleap-rag/appleap-rag
pip install -e ".[parsing]"

echo "=== Done ==="
echo "Start the server with:"
echo "  cd ~/appleap-rag/appleap-rag && uvicorn backend.main:app --host 0.0.0.0 --port 8000"
