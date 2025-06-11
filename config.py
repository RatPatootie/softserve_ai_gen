import os
from typing import Dict, Any

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Data directories
DATA_DIR = "data"
ARTICLES_DIR = os.path.join(DATA_DIR, "articles")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# Vector store configuration
VECTOR_STORE_CONFIG = {
    "persist_directory": "./chroma_db",
    "collection_name": "batch_articles",
    "embedding_model": "all-MiniLM-L6-v2"
}

# Scraping configuration
SCRAPING_CONFIG = {
    "base_url": "https://www.deeplearning.ai/the-batch/",
    "max_articles": 100,
    "rate_limit": 1,  # seconds between requests
    "timeout": 30
}

# Text processing configuration
TEXT_PROCESSING_CONFIG = {
    "chunk_size": 500,
    "chunk_overlap": 50,
    "min_chunk_length": 100
}

# LLM configuration
LLM_CONFIG = {
    "provider": "openai",  # or "anthropic"
    "model": "gpt-3.5-turbo",
    "max_tokens": 500,
    "temperature": 0.3
}

# UI configuration
UI_CONFIG = {
    "page_title": "The Batch RAG System",
    "page_icon": "🤖",
    "layout": "wide"
}