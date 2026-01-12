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
    from .config import GROQ_API_KEY, MODELS, BATCH_SIZE, HUMAN_DETECTION_PROMPT, DATASET_SUITABILITY_PROMPT
except ImportError:
    from config import GROQ_API_KEY, MODELS, BATCH_SIZE, HUMAN_DETECTION_PROMPT, DATASET_SUITABILITY_PROMPT


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
    
    def analyze_single_image(
        self,
        image_path: str,
        prompt: str = HUMAN_DETECTION_PROMPT
    ) -> dict:
        """
        Analyze a single image for human detection.
        
        Args:
            image_path: Path to image file
            prompt: Analysis prompt
            
        Returns:
            Dictionary with analysis result
        """
        model = self._get_current_model()
        base64_image = self._encode_image(image_path)
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.1,
                max_completion_tokens=256,
                response_format={"type": "json_object"}
            )
            
            self.request_count += 1
            self._rotate_model()
            
            content = response.choices[0].message.content
            result = json.loads(content)
            result["image_path"] = image_path
            result["model_used"] = model
            return result
            
        except json.JSONDecodeError as e:
            return {
                "image_path": image_path,
                "has_human": True,  # Default to keeping image if uncertain
                "confidence": 0.0,
                "error": f"JSON parse error: {str(e)}",
                "raw_response": response.choices[0].message.content if response else None
            }
        except Exception as e:
            return {
                "image_path": image_path,
                "has_human": True,  # Default to keeping image if uncertain
                "confidence": 0.0,
                "error": str(e)
            }
    
    def analyze_images_batch(
        self,
        image_paths: list[str],
        prompt: str = None
    ) -> list[dict]:
        """
        Analyze multiple images in a batch (up to 5 images).
        
        This sends all images in a single API request, which is more efficient
        than individual requests.
        
        Args:
            image_paths: List of image paths (max 5)
            prompt: Analysis prompt (uses default if None)
            
        Returns:
            List of analysis results for each image
        """
        if not image_paths:
            return []
        
        # Limit to max batch size
        image_paths = image_paths[:BATCH_SIZE]
        
        if prompt is None:
            prompt = f"""Analyze these {len(image_paths)} images. For EACH image, determine if it contains any human beings.

Respond with a JSON object containing an "images" array with one entry per image in order:
{{
    "images": [
        {{"index": 0, "has_human": true, "confidence": 0.95}},
        {{"index": 1, "has_human": false, "confidence": 0.90}},
        ...
    ]
}}

Rules:
- has_human: true if any human, person, or body part is visible
- confidence: your confidence level from 0.0 to 1.0
- Process images in the exact order provided"""
        
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
                        "has_human": img_result.get("has_human", True),
                        "confidence": img_result.get("confidence", 0.0),
                        "model_used": model
                    })
                else:
                    # Fallback if response doesn't include this image
                    results.append({
                        "image_path": path,
                        "has_human": True,
                        "confidence": 0.0,
                        "error": "Missing in response"
                    })
            
            return results
            
        except json.JSONDecodeError as e:
            # Return default results on error
            return [
                {
                    "image_path": path,
                    "has_human": True,
                    "confidence": 0.0,
                    "error": f"JSON parse error: {str(e)}"
                }
                for path in image_paths
            ]
        except Exception as e:
            return [
                {
                    "image_path": path,
                    "has_human": True,
                    "confidence": 0.0,
                    "error": str(e)
                }
                for path in image_paths
            ]
    
    def analyze_dataset_suitability_batch(
        self,
        image_paths: list[str]
    ) -> list[dict]:
        """
        Analyze multiple images for PPE dataset suitability.
        
        Determines if each image is suitable for training PPE object detection models.
        
        Args:
            image_paths: List of image paths (max 5)
            
        Returns:
            List of results with is_suitable boolean for each image
        """
        if not image_paths:
            return []
        
        # Limit to max batch size
        image_paths = image_paths[:BATCH_SIZE]
        
        prompt = f"""Analyze these {len(image_paths)} images to determine if each is SUITABLE for training a PPE object detection model.

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

