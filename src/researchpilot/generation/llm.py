import logging
import time

from google import genai
from google.genai import errors

from researchpilot.config.settings import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
)

logger = logging.getLogger(__name__)

# Transient Gemini failures worth retrying (overloaded / rate limited).
RETRYABLE_STATUS_CODES = {429, 500, 503, 504}


class GeminiLLM:

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ):
        self.client = genai.Client(
            api_key=GEMINI_API_KEY
        )
        self.max_retries = max_retries
        self.base_delay = base_delay

    def generate(self, prompt: str) -> str:

        for attempt in range(self.max_retries + 1):

            try:
                response = self.client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                    },
                )

                return response.text

            except errors.APIError as exc:

                if (
                    exc.code not in RETRYABLE_STATUS_CODES
                    or attempt == self.max_retries
                ):
                    raise

                # Exponential backoff: 1s, 2s, 4s
                delay = self.base_delay * 2**attempt

                logger.warning(
                    "Gemini returned %s, retrying in %.1fs "
                    "(attempt %d/%d)",
                    exc.code,
                    delay,
                    attempt + 1,
                    self.max_retries,
                )

                time.sleep(delay)
