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

# Combined PPE Dataset Analysis prompt
# Checks: human presence + PPE visibility + image quality
IMAGE_ANALYSIS_PROMPT = """Analyze this image for PPE (Personal Protective Equipment) dataset training suitability.

Check the following criteria:
1. Is there at least one human/person clearly visible?
2. Is at least one PPE item visible? (helmet, safety vest, gloves, safety glasses, safety shoes, face shield, mask, or harness)
3. Is the image clear (not blurry or motion-blurred)?
4. Is the exposure good (not too dark or overexposed)?
5. Are the person/objects large enough to annotate (not too far/small)?

An image is SUITABLE if ALL criteria are met.
An image is NOT SUITABLE if ANY of these are true:
- No humans visible
- Humans visible but NO PPE items can be seen
- Image is too blurry
- Image is too dark or overexposed
- Person/objects are too small

Respond ONLY with a JSON object in this exact format:
{"is_suitable": true}
or
{"is_suitable": false}"""
