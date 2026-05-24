from google import genai
import json
import os
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

prompt = """
Generate 300 diverse and realistic user requests for a PC building recommendation system.
Each request should be a natural sentence a user might type, like:
"I travel a lot and edit videos" or "I want a powerful gaming PC under 800 dollars"

For each request, assign exactly these labels:
- use_case: one of [gaming, video_editing, machine_learning, office, 3d_rendering, music_production]
- device_type: one of [desktop, laptop]
- budget_tier: one of [low, mid, high]

Rules:
- Vary the phrasing a lot (short, long, casual, technical, implicit)
- Some requests should have implicit labels (don't mention the use case directly)
- Cover all combinations of labels
- No duplicate sentences

Return ONLY a valid JSON array, no explanation, no markdown, no code blocks. Example format:
[
  {"text": "I need something powerful for gaming", "use_case": "gaming", "device_type": "desktop", "budget_tier": "high"},
  {"text": "Cheap laptop for taking notes at university", "use_case": "office", "device_type": "laptop", "budget_tier": "low"}
]
"""

print("Generating dataset with Gemini...")
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=prompt
)

raw = response.text.strip()

if raw.startswith("```"):
    raw = raw.split("```")[1]
    if raw.startswith("json"):
        raw = raw[4:]
    raw = raw.strip()

data = json.loads(raw)
print(f"Generated {len(data)} samples")

output_path = os.path.join(os.path.dirname(__file__), "dataset.json")
with open(output_path, "w") as f:
    json.dump(data, f, indent=2)

print(f"Saved to {output_path}")