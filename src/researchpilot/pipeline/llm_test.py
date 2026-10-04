from google import genai

from researchpilot.config.settings import GEMINI_API_KEY

client = genai.Client(api_key=GEMINI_API_KEY)

for model in client.models.list():
    if "generateContent" in model.supported_actions:
        print(model.name)
