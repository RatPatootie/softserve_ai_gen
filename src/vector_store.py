import chromadb
from chromadb.config import Settings
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import uuid

class MultimodalVectorStore:
    def __init__(self, persist_directory="./chroma_db"):
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Initialize sentence transformer for text embeddings
        self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create collections
        self.text_collection = self.client.get_or_create_collection(
            name="text_chunks",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.image_collection = self.client.get_or_create_collection(
            name="image_features",
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_text_chunks(self, chunks, metadata_list):
        """Add text chunks to vector store"""
        # Generate embeddings
        embeddings = self.text_encoder.encode(chunks).tolist()
        
        # Generate IDs
        ids = [str(uuid.uuid4()) for _ in chunks]
        
        # Add to collection
        self.text_collection.add(
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadata_list,
            ids=ids
        )
        
        return ids
    
    def add_image_features(self, image_features, metadata_list):
        """Add image features to vector store"""
        # Normalize features
        normalized_features = []
        for features in image_features:
            if features is not None:
                norm_features = features / np.linalg.norm(features)
                normalized_features.append(norm_features.flatten().tolist())
            else:
                # Create zero vector for failed images
                normalized_features.append([0.0] * 512)  # CLIP feature size
        
        # Generate IDs
        ids = [str(uuid.uuid4()) for _ in image_features]
        
        # Add to collection
        self.image_collection.add(
            embeddings=normalized_features,
            metadatas=metadata_list,
            ids=ids
        )
        
        return ids
    
    def search_text(self, query, n_results=5):
        """Search for similar text chunks"""
        query_embedding = self.text_encoder.encode([query]).tolist()
        
        results = self.text_collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        return results
    
    def search_images(self, query, n_results=5):
        """Search for similar images using text query"""
        # For text-to-image search, we'd need CLIP text encoder
        # For now, return empty results
        return {"documents": [], "metadatas": [], "distances": []}
    
    def hybrid_search(self, query, n_results=5, text_weight=0.7):
        """Perform hybrid search across text and images"""
        text_results = self.search_text(query, n_results)
        image_results = self.search_images(query, n_results)
        
        # Combine results (simplified approach)
        combined_results = {
            'text_results': text_results,
            'image_results': image_results
        }
        
        return combined_results