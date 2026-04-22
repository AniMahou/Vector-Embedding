from sentence_transformers import SentenceTransformer
import numpy as np

# 1. Load the model (This will use the stuff you just downloaded)
# 'all-MiniLM-L6-v2' is the gold standard for fast, local embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text: str) -> list[float]:
    """Get embedding vector locally."""
    # The model handles the cleaning and vectorization
    embedding = model.encode(text)
    return embedding.tolist()

# 2. Get embeddings
cat_vec = get_embedding("cat")
feline_vec = get_embedding("feline")
car_vec = get_embedding("car")

# 3. Check the math
print(f"Embedding dimension: {len(cat_vec)}") 
print(f"First 5 values: {cat_vec[:5]}")

# 4. Bonus: Compare them!
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

sim_cat_feline = cosine_similarity(cat_vec, feline_vec)
sim_cat_car = cosine_similarity(cat_vec, car_vec)

print(f"\nSimilarity (cat vs feline): {sim_cat_feline:.4f}")
print(f"Similarity (cat vs car): {sim_cat_car:.4f}")