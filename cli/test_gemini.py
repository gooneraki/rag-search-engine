from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
print(f"Using key {api_key[:6]}...")


client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum.")

print(response.text)

metadata = response.usage_metadata
print(f"Prompt Tokens: {metadata.prompt_token_count}")
print(f"Response Tokens: {metadata.candidates_token_count}")
