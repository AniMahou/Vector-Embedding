from openai import OpenAI

client = OpenAI()

def get_embedding(text: str, model: str = "text-embedding-3-small") -> list[float]:
    """Get embedding vector for a text."""
    
    response = client.embeddings.create(
        model=model,
        input=text
    )
    
    return response.data[0].embedding

# Get embeddings
cat_embedding = get_embedding("cat")
feline_embedding = get_embedding("feline")
car_embedding = get_embedding("car")

print(f"Embedding dimension: {len(cat_embedding)}")  # 1536
print(f"First 5 values of 'cat': {cat_embedding[:5]}")