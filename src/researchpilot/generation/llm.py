from google import genai

from researchpilot.config.settings import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
)


class GeminiLLM:

    def __init__(self):
        self.client = genai.Client(
            api_key=GEMINI_API_KEY
        )

    def generate(self, prompt: str) -> str:

        response = self.client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
            },
        )

        return response.text
