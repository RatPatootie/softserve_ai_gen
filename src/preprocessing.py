import re
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import torch
from io import BytesIO
import requests


class TextPreprocessor:
    def __init__(self):
        self.clip_model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.clip_processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

    def clean_text(self, text):
        """Clean text from HTML, emojis, extra spaces, and known noise"""
        # Remove HTML tags
        text = re.sub(r'<[^<]+?>', '', text)
        # Remove known repetitive marketing phrases
        text = re.sub(r'✨\s*New course!', '', text,flags=re.IGNORECASE)

        # Remove emojis
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # Emoticons
            u"\U0001F300-\U0001F5FF"  # Symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # Transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # Flags
            "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def get_clip_text_embedding(self, text):
        """Get CLIP text embedding for a given text """
        inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)
        return text_features.numpy()

    @staticmethod
    def chunk_text_with_title(article_json, chunk_size=256, overlap=16, tokenizer=None):
        title = article_json.get("title", "")
        content = article_json.get("content", "").strip()

        content_tokens = tokenizer.encode(content, add_special_tokens=False)
        chunks = []
        start = 0

        while start < len(content_tokens):
            end = min(start + chunk_size, len(content_tokens))
            chunk_tokens = content_tokens[start:end]
            chunk_text = tokenizer.decode(chunk_tokens)

            # Додаємо title
            full_chunk = f"Title: {title}\nContent: {chunk_text}"

            # Перевіряємо загальну кількість токенів у цьому об'єднаному тексті
            tokenized_full = tokenizer(full_chunk, return_tensors="pt", truncation=True, max_length=77)
            input_ids = tokenized_full['input_ids'][0]

            # Якщо результат перевищує 77 навіть після обрізки — скоротимо chunk_text
            while len(input_ids) > 77 and len(chunk_tokens) > 5:
                chunk_tokens = chunk_tokens[:-1]
                chunk_text = tokenizer.decode(chunk_tokens)
                full_chunk = f"Title: {title}\nContent: {chunk_text}"
                input_ids = tokenizer(full_chunk, return_tensors="pt")['input_ids'][0]

            chunks.append(full_chunk)
            start += chunk_size - overlap

        return chunks




class ImagePreprocessor:
    def __init__(self):
        self.clip_model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.clip_processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    
    def process_image(self, image):
        """Process image and extract features"""
        try:
            
            
            # Resize if too large
            if image.size[0] > 1024 or image.size[1] > 1024:
                image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            
            # Extract CLIP features
            inputs = self.clip_processor(images=image, return_tensors="pt")
            
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                
            return image_features.numpy()
          
            
        except Exception as e:
            print(f"Error processing image {image}: {e}")
            return None
    def download_image_from_url(self,url):
        try:
            response = requests.get(url)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            return image
        except Exception as e:
            print(f"Error downloading image: {e}")
            return None

