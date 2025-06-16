import openai
from anthropic import Anthropic
import os
from openai import OpenAI
from typing import Dict, List, Optional
from dotenv import load_dotenv


class LLMIntegration:
    def __init__(self, provider="openai", model="gpt-3.5-turbo"):
        load_dotenv()
        self.provider = provider
        self.model = model
        
        if provider == "openai":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif provider == "anthropic":
            self.anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    def generate_response(self, query: str, context: str, max_tokens: int = 500) -> str:
        """Generate response using LLM"""
        
        system_prompt = """You are an AI assistant that helps users find information from The Batch news articles. 
        Use the provided context to answer questions accurately. If the context doesn't contain enough information 
        to answer the question, say so clearly. Always cite which articles or sources you're referencing."""
        
        user_prompt = f"""
        Context from The Batch articles:
        {context}
        
        User Question: {query}
        
        Please provide a comprehensive answer based on the context above.
        """
        
        if self.provider == "openai":
            return self._generate_openai_response(system_prompt, user_prompt, max_tokens)
        elif self.provider == "anthropic":
            return self._generate_anthropic_response(system_prompt, user_prompt, max_tokens)
        else:
            return "LLM provider not supported."
    
    def _generate_openai_response(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        """Generate response using OpenAI"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _generate_anthropic_response(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        """Generate response using Anthropic Claude"""
        try:
            response = self.anthropic.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=max_tokens,
                temperature=0.3,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def summarize_article(self, article_content: str) -> str:
        """Generate summary of an article"""
        prompt = f"""
        Please provide a concise summary of the following article from The Batch:
        
        {article_content}
        
        Summary should be 2-3 sentences highlighting the key points.
        """
        
        return self.generate_response("Summarize this article", prompt, max_tokens=200)