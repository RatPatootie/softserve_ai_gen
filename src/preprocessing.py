import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from PIL import Image
import torch 
from transformers import CLIPProcessor, CLIPModel
import numpy as np

class TextPreprocessor:
    def __init__(self):
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """Clean and normalize text"""
        # Remove HTML tags
        text = re.sub('<[^<]+?>', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize_and_lemmatize(self, text):
        """Tokenize and lemmatize text"""
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        processed_tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]
        
        return processed_tokens
    
    def create_chunks(self, text, chunk_size=500, overlap=50):
        """Split text into overlapping chunks"""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                
                # Create overlap
                overlap_sentences = current_chunk[-overlap//10:] if len(current_chunk) > overlap//10 else current_chunk
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s.split()) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks

class ImagePreprocessor:
    def __init__(self):
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    def process_image(self, image_path):
        """Process image and extract features"""
        try:
            image = Image.open(image_path).convert('RGB')
            
            # Resize if too large
            if image.size[0] > 1024 or image.size[1] > 1024:
                image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            
            # Extract CLIP features
            inputs = self.clip_processor(images=image, return_tensors="pt")
            
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                
            return {
                'features': image_features.numpy(),
                'size': image.size,
                'mode': image.mode
            }
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None
    
    def generate_image_description(self, image_path, caption=""):
        """Generate description for image using CLIP"""
        try:
            image = Image.open(image_path).convert('RGB')
            
            # Use existing caption or generate generic description
            if caption:
                return f"Image showing: {caption}"
            
            # For demo purposes, return a generic description
            # In production, you might use a more sophisticated image captioning model
            return "Image from The Batch article"
            
        except Exception as e:
            return f"Image processing error: {str(e)}"