from openai import OpenAI
from typing import Optional
from dotenv import load_dotenv
import os
load_dotenv()
#but text-embedding-3-small only works for openai model.
client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)
# Get full 1536-dim embedding
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=text,
    dimensions=1536  # Full size
)

# Or get truncated 256-dim version!
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=text,
    dimensions=256  # 6x smaller, 95% of the quality!
)