from src.vector_store import MultimodalDB
import json
from typing import List, Dict, Any

class MultimodalRetriever:
    def __init__(self, vector_store: MultimodalDB):
        self.vector_store = vector_store
    
    def retrieve(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """Retrieve relevant content for a given query"""
        
        # Perform hybrid search
        search_results = self.vector_store.search(query, n_results)
        # Process and rank results
        processed_results = self.process_search_results(search_results, query, n_results)
        
        return processed_results
    
    def process_search_results(self, search_results: Dict, query: str,n_results:int) -> Dict[str, Any]:
        """Process and structure search results"""
        
        text_results = search_results.get('text_results', {})
        image_results = search_results.get('image_results', {})
        
        # Structure text results
        text_chunks = []
        if text_results.get('documents'):
            for i, (doc, metadata, distance) in enumerate(zip(
                text_results['documents'][0],
                text_results['metadatas'][0],
                text_results['distances'][0]
            )):
                text_chunks.append({
                    'content': doc,
                    'metadata': metadata,
                    'relevance_score': 1 - distance,  # Convert distance to similarity
                    'type': 'text'
                })
        
        # Structure image results
        image_items = []
        if image_results.get('metadatas'):
            for metadata, distance in zip(
                image_results['metadatas'][0],
                image_results['distances'][0]
            ):
                image_items.append({
                    'metadata': metadata,
                    'relevance_score': 1 - distance,
                    'type': 'image'
                })
        
        # Combine and sort by relevance
        all_results = text_chunks + image_items
        all_results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return {
            'query': query,
            'results': all_results[:n_results],
            'total_found': len(all_results)
        }
    
    def get_context_for_llm(self, retrieval_results: Dict[str, Any]) -> str:
        """Format retrieval results as context for LLM"""
        
        context_parts = []
        context_parts.append(f"Query: {retrieval_results['query']}\n")
        context_parts.append("Relevant Information:\n")
        
        for i, result in enumerate(retrieval_results['results'][:5], 1):
            if result['type'] == 'text':
                context_parts.append(f"{i}. {result['content']}")
                if 'title' in result['metadata']:
                    context_parts.append(f"   Source: {result['metadata']['title']}")
            elif result['type'] == 'image':
                context_parts.append(f"{i}. Image: {result['metadata'].get('description', 'No description')}")
                if 'alt' in result['metadata']:
                    context_parts.append(f"   Alt text: {result['metadata']['alt']}")
                    
            context_parts.append("")  # Empty line for separation
        
        return "\n".join(context_parts)