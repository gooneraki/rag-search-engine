import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")


class GenAIClient:
    """Client for interacting with the GenAI API."""

    def __init__(self):
        # Initialize the GenAI client here using the provided API key
        self.client = genai.Client(api_key=api_key)

    def generate_response(self, prompt: str, model="gemini-2.5-flash") -> str:
        response = self.client.models.generate_content(
            model=model,
            contents=prompt
        )
        return response.text


def prompt_for_typo(query: str) -> str:
    return f"""Fix any spelling errors in this movie search query.

Only correct obvious typos. Don't change correctly spelled words.

Query: "{query}"

If no errors, return the original query.
Corrected:"""
