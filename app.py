import streamlit as st
import os
import json
from datetime import datetime
import pandas as pd
from PIL import Image
import logging

# Import our custom modules
from src.data_ingestion import BatchScraper
from src.preprocessing import TextPreprocessor, ImagePreprocessor
from src.vector_store import MultimodalDB
from src.multimodal_retriever import MultimodalRetriever
from src.llm_integration import LLMIntegration

# Page configuration
st.set_page_config(
    page_title="The Batch RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'articles_loaded' not in st.session_state:
    st.session_state.articles_loaded = False
def init_data():
    scrapper = BatchScraper()
    scrapper.download_articles()
    # Load the cleaned articles
    with open(os.path.join('data/articles', "scraped_articles.json"), "r", encoding="utf-8") as f:
        articles = json.load(f)
    print("Processing articles...")
    results = st.session_state.vector_store.add_articles_batch(articles)  # Start with first 5 articles as a test
    print(f"Processed {len(results)} articles")
    st.session_state.articles_loaded = True

def initialize_system():
    
    """Initialize the RAG system components"""
    with st.spinner("Initializing RAG system..."):
        # Initialize vector store
        st.session_state.vector_store = MultimodalDB(persist_directory="./multimodal_db")
        
        # Initialize retriever
        st.session_state.retriever = MultimodalRetriever(st.session_state.vector_store)
        
        # Initialize LLM
        llm_provider = st.sidebar.selectbox("Select LLM Provider", ["openai", "anthropic"])
        st.session_state.llm = LLMIntegration(provider=llm_provider)
        init_data()
        
        st.success("System initialized successfully!")

def main():
    n_results = 5  # default
    st.title("🤖 The Batch Multimodal RAG System")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("System Controls")
    
    # Initialize system
    if st.sidebar.button("Initialize System"):
        initialize_system()
    
    # Load sample data
    #if st.sidebar.button("Load Sample Data") and st.session_state.vector_store:
        
    
    # Main interface
    if st.session_state.vector_store and st.session_state.articles_loaded:
        st.header("Query Interface")
        
        # Query input
        query = st.text_input("Enter your question about The Batch articles:", 
                             placeholder="e.g., What are the latest AI developments in healthcare?")
        
        # Search parameters
        col1, col2 = st.columns(2)
        with col1:
            n_results = st.slider("Number of results", min_value=1, max_value=10, value=5)
        with col2:
            include_images = st.checkbox("Include images in results", value=True)
        
        # Search button
        if st.button("Search", type="primary") and query:
            with st.spinner("Searching and generating response..."):
                # Retrieve relevant content
                retrieval_results = st.session_state.retriever.retrieve(query, n_results)
                
                # Generate LLM response
                context = st.session_state.retriever.get_context_for_llm(retrieval_results)
                llm_response = st.session_state.llm.generate_response(query, context)
                
                # Display results
                st.subheader("🤖 AI Response")
                st.write(llm_response)
                
                st.subheader("📄 Retrieved Articles")
                
                for i, result in enumerate(retrieval_results['results'], 1):
                    with st.expander(f"Result {i} - Relevance: {result['relevance_score']:.2f}"):
                        if result['type'] == 'text':
                            st.write("**Content:**")
                            st.write(result['content'])
                            
                            if 'title' in result['metadata']:
                                st.write(f"**Source:** {result['metadata']['title']}")
                            if 'url' in result['metadata']:
                                st.write(f"**URL:** {result['metadata']['url']}")
                        
                        elif result['type'] == 'image' and include_images:
                            st.write("**Image Result:**")
                            if 'description' in result['metadata']:
                                st.write(f"**Description:** {result['metadata']['description']}")
                            if 'alt' in result['metadata']:
                                st.write(f"**Alt Text:** {result['metadata']['alt']}")
    
    else:
        st.info("Please initialize the system and load data to begin using the RAG system.")
        
        # System status
        st.subheader("System Status")
        status_col1, status_col2, status_col3 = st.columns(3)
        
        with status_col1:
            st.metric("Vector Store", "✅" if st.session_state.vector_store else "❌")
        with status_col2:
            st.metric("Retriever", "✅" if st.session_state.retriever else "❌")
        with status_col3:
            st.metric("Data Loaded", "✅" if st.session_state.articles_loaded else "❌")

    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit, ChromaDB, and Sentence Transformers")

if __name__ == "__main__":
    main()