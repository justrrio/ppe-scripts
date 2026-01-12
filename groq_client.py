"""
Groq API Client Module
Wrapper for Groq Vision API with batch processing support.
"""
import base64
import json
import os
import time
from groq import Groq

try:
    from .config import GROQ_API_KEY, MODELS, BATCH_SIZE, IMAGE_ANALYSIS_PROMPT
except ImportError:
    from config import GROQ_API_KEY, MODELS, BATCH_SIZE, IMAGE_ANALYSIS_PROMPT


class GroqVisionClient:
    """
    Client for Groq Vision API with batch processing and model alternation.
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize Groq client.
        
        Args:
            api_key: Groq API key (uses config if not provided)
        """
        self.api_key = api_key or GROQ_API_KEY
        self.client = Groq(api_key=self.api_key)
        self.model_names = list(MODELS.keys())
        self.current_model_idx = 0
        self.request_count = 0
    
    def _get_current_model(self) -> str:
        """Get the current model ID."""
        model_key = self.model_names[self.current_model_idx]
        return MODELS[model_key]
    
    def _rotate_model(self):
        """Rotate to the next model for load balancing."""
        self.current_model_idx = (self.current_model_idx + 1) % len(self.model_names)
    
    def _encode_image(self, image_path: str) -> str:
        """
        Encode an image file to base64.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Base64 encoded string
        """
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def analyze_images_batch(
        self,
        image_paths: list[str]
    ) -> list[dict]:
        """
        Analyze multiple images for PPE dataset suitability.
        
        Checks: human presence + PPE visibility + image quality
        
        Args:
            image_paths: List of image paths (max BATCH_SIZE)
            
        Returns:
            List of results with is_suitable boolean for each image
        """
        if not image_paths:
            return []
        
        # Limit to max batch size
        image_paths = image_paths[:BATCH_SIZE]
        
        prompt = f"""Analyze these {len(image_paths)} images for PPE dataset training suitability.

An image is SUITABLE if ALL criteria are met:
1. At least one human/person is clearly visible
2. At least one PPE item is visible (helmet, vest, gloves, glasses, shoes, mask, harness)
3. Image is NOT blurry
4. Proper exposure (not too dark or overexposed)
5. Person/PPE is large enough to annotate

Respond with a JSON object containing an "images" array with one entry per image in order:
{{
    "images": [
        {{"index": 0, "is_suitable": true}},
        {{"index": 1, "is_suitable": false}},
        ...
    ]
}}"""
        
        model = self._get_current_model()
        
        # Build message content with all images
        content = [{"type": "text", "text": prompt}]
        
        for path in image_paths:
            base64_image = self._encode_image(path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                temperature=0.1,
                max_completion_tokens=1024,
                response_format={"type": "json_object"}
            )
            
            self.request_count += 1
            self._rotate_model()
            
            result_content = response.choices[0].message.content
            parsed = json.loads(result_content)
            
            # Map results back to image paths
            results = []
            images_data = parsed.get("images", [])
            
            for i, path in enumerate(image_paths):
                if i < len(images_data):
                    img_result = images_data[i]
                    results.append({
                        "image_path": path,
                        "is_suitable": img_result.get("is_suitable", True),
                        "model_used": model
                    })
                else:
                    # Fallback if response doesn't include this image
                    results.append({
                        "image_path": path,
                        "is_suitable": True,  # Default to keeping if uncertain
                        "error": "Missing in response"
                    })
            
            return results
            
        except json.JSONDecodeError as e:
            return [
                {
                    "image_path": path,
                    "is_suitable": True,
                    "error": f"JSON parse error: {str(e)}"
                }
                for path in image_paths
            ]
        except Exception as e:
            return [
                {
                    "image_path": path,
                    "is_suitable": True,
                    "error": str(e)
                }
                for path in image_paths
            ]
    
    def get_stats(self) -> dict:
        """Get client usage statistics."""
        return {
            "total_requests": self.request_count,
            "current_model": self._get_current_model()
        }
