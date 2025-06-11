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
from src.vector_store import MultimodalVectorStore
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

def initialize_system():
    """Initialize the RAG system components"""
    with st.spinner("Initializing RAG system..."):
        # Initialize vector store
        st.session_state.vector_store = MultimodalVectorStore()
        
        # Initialize retriever
        st.session_state.retriever = MultimodalRetriever(st.session_state.vector_store)
        
        # Initialize LLM
        llm_provider = st.sidebar.selectbox("Select LLM Provider", ["openai", "anthropic"])
        st.session_state.llm = LLMIntegration(provider=llm_provider)
        
        st.success("System initialized successfully!")

def load_sample_data():
    """Scrape articles from The Batch and load them into the vector store"""
    
    
   
    
    # Initialize the scraper - this is where the magic begins
    scraper = BatchScraper()
    
    # Create a progress tracking system so users know what's happening
    progress_container = st.container()
    
    with progress_container:
        st.info(f"Starting to scrape up articles from The Batch...")
        
        # Create progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # This is the core transformation - real scraping instead of fake data
            status_text.text("Discovering articles on The Batch website...")
            scraped_articles = scraper.scrape_articles()
            
            if not scraped_articles:
                st.error("No articles were found. This might indicate that The Batch website structure has changed, or there's a connectivity issue.")
                return
            
            st.success(f"Successfully discovered {len(scraped_articles)} articles!")
            
            # Optional image downloading - this adds richness to your dataset
            
            status_text.text("Downloading images from articles...")
            scraper.download_images(scraped_articles)
            st.info("Images downloaded and ready for processing!")
            
        except Exception as e:
            st.error(f"Scraping encountered an error: {str(e)}")
            st.info("This might be due to network issues or changes in The Batch website structure.")
            return
    
    # Now we process the scraped data through your existing pipeline
    # This part is beautiful because it shows how well your architecture separates concerns
    text_processor = TextPreprocessor()
    image_processor = ImagePreprocessor()  # You'll need this for the scraped images
    
    # Process each article with a progress indicator
    total_chunks_added = 0
    
    for i, article in enumerate(scraped_articles):
        # Update progress
        progress_bar.progress((i + 1) / len(scraped_articles))
        status_text.text(f"Processing article {i + 1}: {article['title'][:50]}...")
        
        # Text processing (this remains identical to your sample data approach)
        cleaned_text = text_processor.clean_text(article['content'])
        chunks = text_processor.create_chunks(cleaned_text)
        if chunks:
            metadata_list = [
                {
                    'title': article['title'],
                    'url': article['url'],
                    'publication_date': article['publication_date'],
                    'scraped_at': article['scraped_at'],
                    'chunk_index': j,
                    'total_chunks': len(chunks)
                }
                for j in range(len(chunks))
            ]
            
            st.session_state.vector_store.add_text_chunks(chunks, metadata_list)
            total_chunks_added += len(chunks)
        else:
            logging.warning(f"Skipped article with no text chunks: {article['url']}")

       
        # Process images if they exist - this is new functionality your sample data didn't have
        if article['images']:
            image_metadata = []
            for img_idx, img_data in enumerate(article['images']):
                img_metadata = {
                    'title': article['title'],
                    'article_url': article['url'],
                    'image_url': img_data['url'],
                    'alt_text': img_data['alt'],
                    'caption': img_data['caption'],
                    'local_path': img_data.get('local_path', ''),  # Only if images were downloaded
                    'image_index': img_idx
                }
                image_metadata.append(img_metadata)
            
            # Add images to your multimodal vector store
            try:
                st.session_state.vector_store.add_images(article['images'], image_metadata)
            except Exception as e:
                st.warning(f"Could not process images for '{article['title']}': {str(e)}")
    
    # Update session state and provide comprehensive feedback
    st.session_state.articles_loaded = True
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Provide detailed success information
    success_container = st.container()
    with success_container:
        st.success("Data loading completed successfully!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Articles Loaded", len(scraped_articles))
        with col2:
            st.metric("Text Chunks Created", total_chunks_added)
        with col3:
            total_images = sum(len(article.get('images', [])) for article in scraped_articles)
            st.metric("Images Processed", total_images)
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
    if st.sidebar.button("Load Sample Data") and st.session_state.vector_store:
        load_sample_data()
    
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