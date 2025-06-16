import chromadb
from chromadb.config import Settings
import numpy as np
import json
import uuid
import torch
from typing import List, Dict, Any, Optional, Tuple
from src.preprocessing import TextPreprocessor, ImagePreprocessor

class MultimodalDB:
    def __init__(self, persist_directory: str = "./chroma_db"):
        """Initialize the MultimodalDB with text and image processors and vector store"""
        self.text_processor = TextPreprocessor()
        self.image_processor = ImagePreprocessor()
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Create collections for text and image embeddings
        self.text_collection = self.client.get_or_create_collection(
            name="text_embeddings",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.image_collection = self.client.get_or_create_collection(
            name="image_embeddings",
            metadata={"hnsw:space": "cosine"}
        )

    def process_article(self, article: Dict[str, Any]) -> Tuple[List[np.ndarray], List[np.ndarray], Dict[str, Any]]:
        """Process a single article to extract text and image embeddings"""
        # Process text
        text_embeddings = []
        chunks = self.text_processor.chunk_text_with_title(
            article, 
            tokenizer=self.text_processor.clip_processor.tokenizer
        )
        
        text_embeddings = []
        valid_chunks = []
        for chunk in chunks:
            chunk = self.text_processor.clean_text(chunk)
            embedding = self.text_processor.get_clip_text_embedding(chunk)
            if embedding is not None:
                text_embeddings.append(embedding)
                valid_chunks.append(chunk)

        # Process images
        image_embeddings = []
        for img_data in article.get('images', []):
            image = self.image_processor.download_image_from_url(img_data['url'])
            if image:
                embedding = self.image_processor.process_image(image)
                if embedding is not None:
                    image_embeddings.append(embedding)
                    img_data['processed'] = True
                else:
                    img_data['processed'] = False
            else:
                img_data['processed'] = False

        # Create metadata
        metadata = {
            'article_url': article.get('url', ''),
            'title': article.get('title', ''),
            'publication_date': article.get('publication_date', ''),
            'processed_at': article.get('scraped_at', '')
        }

        return text_embeddings, image_embeddings, metadata,valid_chunks

    def add_article(self, article: Dict[str, Any]) -> Dict[str, List[str]]:
        """Add an article to the database, processing both text and images"""
        text_embeddings, image_embeddings, metadata,chunks = self.process_article(article)
        
        # Add text embeddings
        text_ids = []
        for i, embedding in enumerate(text_embeddings):
            chunk_id = str(uuid.uuid4())
            chunk_metadata = {
                **metadata,
                'chunk_index': i,
                'total_chunks': len(text_embeddings),
                'chunk_text': chunks[i] if i < len(chunks) else ''
            }
            
            self.text_collection.add(
                embeddings=[embedding.flatten().tolist()],
                metadatas=[chunk_metadata],
                ids=[chunk_id],
                documents=[chunks[i]] 
            )
            text_ids.append(chunk_id)

        # Add image embeddings
        image_ids = []
        for i, embedding in enumerate(image_embeddings):
            image_id = str(uuid.uuid4())
            image_metadata = {
                **metadata,
                'image_index': i,
                'image_url': article['images'][i]['url'] if i < len(article['images']) else '',
                'total_images': len(image_embeddings)
            }
            
            self.image_collection.add(
                embeddings=[embedding.flatten().tolist()],
                metadatas=[image_metadata],
                ids=[image_id]
            )
            image_ids.append(image_id)

        return {
            'text_ids': text_ids,
            'image_ids': image_ids
        }

    def search(self, query: str, n_results: int = 5, modality: str = 'both') -> Dict[str, Any]:
        """Search the database using a text query"""
        # Get query embedding
        query_text_embedding = self.text_processor.get_clip_text_embedding(query)
        
        results = {}
        
        if modality in ['text', 'both']:
            # Search text collection
            text_results = self.text_collection.query(
                query_embeddings=[query_text_embedding.flatten().tolist()],
                n_results=n_results
            )
            results['text_results'] = text_results
        
        if modality in ['image', 'both']:
            # Search image collection using the same text embedding
            image_results = self.image_collection.query(
                query_embeddings=[query_text_embedding.flatten().tolist()],
                n_results=n_results
            )
            results['image_results'] = image_results
        
        return results

    def article_exists(self, article_url: str) -> bool:
        """Check if an article already exists in the database"""
        try:
            # Search in text collection for the URL
            results = self.text_collection.get(
                where={"article_url": article_url},
                limit=1
            )
            return len(results['ids']) > 0
        except Exception as e:
            print(f"Error checking article existence: {e}")
            return False

    def add_articles_batch(self, articles: List[Dict[str, Any]]) -> List[Dict[str, List[str]]]:
        """Process and add multiple articles in batch, skipping existing ones"""
        results = []
        skipped = 0
        
        for article in articles:
            try:
                article_url = article.get('url', '')
                if not article_url:
                    print(f"Warning: Article has no URL, skipping: {article.get('title', 'Unknown')}")
                    results.append({'text_ids': [], 'image_ids': []})
                    continue
                
                # Check if article already exists
                if self.article_exists(article_url):
                    print(f"Article already exists, skipping: {article_url}")
                    results.append({'text_ids': [], 'image_ids': []})
                    skipped += 1
                    continue
                
                # Process and add new article
                article_ids = self.add_article(article)
                results.append(article_ids)
                
            except Exception as e:
                print(f"Error processing article {article.get('title', 'Unknown')}: {e}")
                results.append({'text_ids': [], 'image_ids': []})
                import traceback
                traceback.print_exc()
        
        print(f"Processed articles: {len(articles)}, Skipped existing: {skipped}")
        return results