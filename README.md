# The Batch Multimodal RAG System

## Overview
A comprehensive Retrieval-Augmented Generation system for The Batch news articles, supporting both text and image content.

## Features
- Web scraping from The Batch
- Multimodal content processing (text + images)
- Vector-based semantic search
- LLM integration (OpenAI/Anthropic)
- Interactive Streamlit UI
- Hybrid retrieval system

## Installation
### Prerequisites
- Python 3.10.11
- Git
- API keys for OpenAI or Anthropic (optional but recommended)

### Step 1: Clone Repository
```bash
git clone https://github.com/your-username/multimodal-rag-system.git
cd multimodal-rag-system
```
### Step 2: Create Virtual Environment
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```
### Step 3: Install Dependencies
``` bash
pip install -r requirements.txt
```
### Step 4: Environment Setup
Create a .env file in the root directory:
``` bash
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```
### Step 5: Download NLTK Data
```bash
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```
### Step 6: Run the Application
``` bash
streamlit run app.py
```