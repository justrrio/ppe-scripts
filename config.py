"""
Configuration settings for PPE Frame Extraction Pipeline
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

# Groq API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Model IDs
MODELS = {
    "scout": "meta-llama/llama-4-scout-17b-16e-instruct",
    "maverick": "meta-llama/llama-4-maverick-17b-128e-instruct"
}

# Processing settings
BATCH_SIZE = 5  # Max images per Groq API request
FRAME_INTERVAL_SECONDS = 0.2  # 1 / 5 (5 FPS)

# Human detection prompt
HUMAN_DETECTION_PROMPT = """Analyze this image carefully. Does it contain any human beings (people, workers, crew members, or any part of a human body)?

Respond ONLY with a JSON object in this exact format:
{"has_human": true, "confidence": 0.95}
or
{"has_human": false, "confidence": 0.95}

Where:
- has_human: boolean indicating if humans are visible
- confidence: float between 0.0 and 1.0 indicating your confidence level"""


# PPE Dataset Suitability prompt
DATASET_SUITABILITY_PROMPT = """Analyze this image to determine if it is SUITABLE for training a PPE (Personal Protective Equipment) object detection AI model like YOLO or RT-DETR.

An image is SUITABLE if ALL of these criteria are met:
1. At least one human/person is clearly visible
2. At least one PPE item is visible (helmet, safety vest, gloves, safety glasses, safety shoes, face shield, mask, or harness)
3. Image is NOT blurry or motion-blurred
4. Image has proper exposure (not too dark or overexposed)
5. The person/PPE is large enough to annotate (not too far/small)

An image is NOT SUITABLE if ANY of these are true:
- No humans visible at all
- Humans visible but NO PPE items can be seen
- Image is too blurry to annotate accurately
- Image is too dark or overexposed
- Person/objects are too small or too far away

Respond ONLY with a JSON object in this exact format:
{"is_suitable": true}
or
{"is_suitable": false}"""
