class SemanticSearchEngine:
    """Simple semantic search over documents."""
    
    def __init__(self, embedding_model):
        self.model = embedding_model
        self.documents = []
        self.embeddings = []
    
    def index(self, documents: list[str]):
        """Index documents with embeddings."""
        self.documents = documents
        self.embeddings = self.model.encode(documents)
    
    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        """Search for similar documents."""
        query_emb = self.model.encode([query])[0]
        
        # Compute similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity([query_emb], self.embeddings)[0]
        
        # Get top-k
        top_indices = similarities.argsort()[-k:][::-1]
        
        return [(self.documents[i], similarities[i]) for i in top_indices]