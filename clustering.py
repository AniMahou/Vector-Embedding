import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any
import json

class ProductionEmbeddingSystem:
    """Complete embedding-based search and analysis system."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.metadata = []
        self.embeddings = None
        self.bm25 = None
    
    def index(self, documents: List[str], metadata: List[Dict] = None):
        """Index documents with embeddings and BM25."""
        self.documents = documents
        self.metadata = metadata or [{}] * len(documents)
        
        # Create embeddings
        print(f"📊 Indexing {len(documents)} documents...")
        self.embeddings = self.model.encode(documents, show_progress_bar=True)
        
        # Create BM25 index
        tokenized = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)
        
        print(f"✅ Index complete. Embedding shape: {self.embeddings.shape}")
    
    def search(
        self,
        query: str,
        k: int = 5,
        method: str = "hybrid",
        threshold: float = None
    ) -> List[Dict]:
        """
        Search with multiple methods.
        
        Args:
            query: Search query
            k: Number of results
            method: "dense", "sparse", or "hybrid"
            threshold: Minimum similarity score
        """
        if method == "dense":
            results = self._dense_search(query, k)
        elif method == "sparse":
            results = self._sparse_search(query, k)
        else:  # hybrid
            results = self._hybrid_search(query, k)
        
        if threshold:
            results = [r for r in results if r["score"] >= threshold]
        
        return results
    
    def _dense_search(self, query: str, k: int) -> List[Dict]:
        query_emb = self.model.encode([query])[0]
        similarities = cosine_similarity([query_emb], self.embeddings)[0]
        top_indices = similarities.argsort()[-k:][::-1]
        
        return [
            {
                "document": self.documents[i],
                "score": float(similarities[i]),
                "metadata": self.metadata[i],
                "index": int(i)
            }
            for i in top_indices
        ]
    
    def _sparse_search(self, query: str, k: int) -> List[Dict]:
        tokenized = query.lower().split()
        scores = self.bm25.get_scores(tokenized)
        # Normalize
        scores = scores / scores.max() if scores.max() > 0 else scores
        top_indices = scores.argsort()[-k:][::-1]
        
        return [
            {
                "document": self.documents[i],
                "score": float(scores[i]),
                "metadata": self.metadata[i],
                "index": int(i)
            }
            for i in top_indices
        ]
    
    def _hybrid_search(self, query: str, k: int) -> List[Dict]:
        # Get both scores
        query_emb = self.model.encode([query])[0]
        dense_scores = cosine_similarity([query_emb], self.embeddings)[0]
        
        tokenized = query.lower().split()
        sparse_scores = self.bm25.get_scores(tokenized)
        sparse_scores = sparse_scores / sparse_scores.max() if sparse_scores.max() > 0 else sparse_scores
        
        # Normalize dense scores
        dense_scores = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min() + 1e-8)
        
        # Combine (equal weights)
        combined = (dense_scores + sparse_scores) / 2
        top_indices = combined.argsort()[-k:][::-1]
        
        return [
            {
                "document": self.documents[i],
                "score": float(combined[i]),
                "dense_score": float(dense_scores[i]),
                "sparse_score": float(sparse_scores[i]),
                "metadata": self.metadata[i],
                "index": int(i)
            }
            for i in top_indices
        ]
    
    def find_similar(self, doc_index: int, k: int = 5) -> List[Dict]:
        """Find documents similar to a given document."""
        query_emb = self.embeddings[doc_index]
        similarities = cosine_similarity([query_emb], self.embeddings)[0]
        
        # Exclude self
        similarities[doc_index] = -1
        
        top_indices = similarities.argsort()[-k:][::-1]
        
        return [
            {
                "document": self.documents[i],
                "similarity": float(similarities[i]),
                "metadata": self.metadata[i],
                "index": int(i)
            }
            for i in top_indices
        ]
    
    def cluster(self, n_clusters: int = 5) -> Dict[int, List[int]]:
        """Cluster documents and return cluster assignments."""
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(self.embeddings)
        
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(i)
        
        return clusters
    
    def export_embeddings(self, path: str):
        """Save embeddings and documents to disk."""
        np.save(f"{path}_embeddings.npy", self.embeddings)
        with open(f"{path}_documents.json", "w") as f:
            json.dump({
                "documents": self.documents,
                "metadata": self.metadata
            }, f)
    
    @classmethod
    def load(cls, path: str, model_name: str = 'all-MiniLM-L6-v2'):
        """Load saved embeddings and documents."""
        system = cls(model_name)
        system.embeddings = np.load(f"{path}_embeddings.npy")
        with open(f"{path}_documents.json", "r") as f:
            data = json.load(f)
            system.documents = data["documents"]
            system.metadata = data["metadata"]
        
        # Recreate BM25
        tokenized = [doc.lower().split() for doc in system.documents]
        system.bm25 = BM25Okapi(tokenized)
        
        return system

# Demo
documents = [
    "How to reset your password on the login page",
    "Password reset instructions for forgotten credentials",
    "Shipping policy: orders arrive in 3-5 business days",
    "Return policy: 30-day money-back guarantee",
    "Contact customer support for account issues",
    "Billing and payment methods accepted",
    "ERROR_CODE_0x7B: Disk controller failure",
    "Troubleshooting system error 0x7B",
]

metadata = [
    {"category": "account", "type": "how-to"},
    {"category": "account", "type": "how-to"},
    {"category": "shipping", "type": "policy"},
    {"category": "returns", "type": "policy"},
    {"category": "support", "type": "contact"},
    {"category": "billing", "type": "policy"},
    {"category": "errors", "type": "troubleshooting"},
    {"category": "errors", "type": "troubleshooting"},
]

# Initialize and index
system = ProductionEmbeddingSystem()
system.index(documents, metadata)

# Test searches
print("\n" + "="*60)
print("DENSE SEARCH: 'forgot password'")
print("="*60)
for r in system.search("forgot password", method="dense", k=3):
    print(f"  Score: {r['score']:.3f} | {r['document']}")

print("\n" + "="*60)
print("SPARSE SEARCH: 'ERROR_CODE_0x7B'")
print("="*60)
for r in system.search("ERROR_CODE_0x7B", method="sparse", k=3):
    print(f"  Score: {r['score']:.3f} | {r['document']}")

print("\n" + "="*60)
print("HYBRID SEARCH: 'password reset'")
print("="*60)
for r in system.search("password reset", method="hybrid", k=3):
    print(f"  Combined: {r['score']:.3f} | Dense: {r['dense_score']:.3f} | Sparse: {r['sparse_score']:.3f}")
    print(f"  {r['document']}")

print("\n" + "="*60)
print("SIMILAR DOCUMENTS (to doc 0)")
print("="*60)
for r in system.find_similar(0, k=3):
    print(f"  Similarity: {r['similarity']:.3f} | {r['document']}")

print("\n" + "="*60)
print("CLUSTERING")
print("="*60)
clusters = system.cluster(n_clusters=3)
for cluster_id, doc_indices in clusters.items():
    print(f"\n📁 Cluster {cluster_id}:")
    for idx in doc_indices[:3]:
        print(f"  • {system.documents[idx]}")