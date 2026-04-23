import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
import time

class VectorDBLab:
    """Complete vector database hands-on lab."""
    
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./vector_db_lab")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def create_collection_with_index(
        self,
        name: str,
        hnsw_params: dict = None
    ):
        """Create collection with specific index parameters."""
        
        # Delete if exists
        try:
            self.client.delete_collection(name)
        except:
            pass
        
        metadata = {"hnsw:space": "cosine"}
        if hnsw_params:
            metadata.update(hnsw_params)
        
        collection = self.client.create_collection(
            name=name,
            metadata=metadata
        )
        
        print(f"✅ Created collection '{name}'")
        print(f"   Index params: {metadata}")
        
        return collection
    
    def index_documents(
    self,
    collection,
    documents: list[str],
    metadata: list[dict] = None
    ):
    
        embeddings = self.encoder.encode(documents).tolist()
        ids = [f"doc_{i}" for i in range(len(documents))]
    
        start = time.time()
    
        # FIX: If metadata is None, don't pass a list of empty dicts.
        # Just pass the metadata variable directly.
        collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadata,  # Removed the 'or [{}] * len(documents)'
            ids=ids
        )
        
        elapsed = time.time() - start
        print(f"📊 Indexed {len(documents)} documents in {elapsed:.2f}s")
        
    def benchmark_search(
        self,
        collection,
        queries: list[str],
        k: int = 5
    ) -> dict:
        """Benchmark search performance."""
        
        query_embeddings = self.encoder.encode(queries).tolist()
        
        times = []
        results = []
        
        for query, query_emb in zip(queries, query_embeddings):
            start = time.time()
            result = collection.query(
                query_embeddings=[query_emb],
                n_results=k
            )
            elapsed = time.time() - start
            
            times.append(elapsed)
            results.append(result)
        
        return {
            "avg_ms": np.mean(times) * 1000,
            "p95_ms": np.percentile(times, 95) * 1000,
            "p99_ms": np.percentile(times, 99) * 1000,
            "results": results
        }
    
    def demonstrate_hybrid_search(
        self,
        documents: list[str],
        query: str
    ):
        """Demonstrate hybrid search using ChromaDB + BM25."""
        
        from rank_bm25 import BM25Okapi
        
        # Create collection
        collection = self.create_collection_with_index("hybrid_demo")
        self.index_documents(collection, documents)
        
        # Dense search
        query_emb = self.encoder.encode([query]).tolist()
        dense_results = collection.query(
            query_embeddings=query_emb,
            n_results=5
        )
        
        # Sparse search (BM25)
        tokenized_docs = [doc.lower().split() for doc in documents]
        bm25 = BM25Okapi(tokenized_docs)
        sparse_scores = bm25.get_scores(query.lower().split())
        sparse_indices = sparse_scores.argsort()[-5:][::-1]
        
        # Display comparison
        print("\n" + "="*60)
        print(f"HYBRID SEARCH DEMO: '{query}'")
        print("="*60)
        
        print("\n📊 DENSE (Embedding) Results:")
        for i, (doc, dist) in enumerate(zip(dense_results['documents'][0], 
                                            dense_results['distances'][0])):
            print(f"  {i+1}. [sim={1-dist:.3f}] {doc[:80]}...")
        
        print("\n📊 SPARSE (BM25) Results:")
        for i, idx in enumerate(sparse_indices):
            print(f"  {i+1}. [bm25] {documents[idx][:80]}...")
        
        # RRF Fusion
        scores = {}
        for rank, doc in enumerate(dense_results['documents'][0], 1):
            # Find index of this document
            idx = documents.index(doc)
            scores[idx] = scores.get(idx, 0) + 1 / (60 + rank)
        
        for rank, idx in enumerate(sparse_indices, 1):
            scores[idx] = scores.get(idx, 0) + 1 / (60 + rank)
        
        sorted_indices = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        print("\n🔀 HYBRID (RRF) Results:")
        for i, (idx, score) in enumerate(sorted_indices[:5]):
            print(f"  {i+1}. [rrf={score:.4f}] {documents[idx][:80]}...")

# Run the lab
lab = VectorDBLab()

# Test documents
documents = [
    "How to reset your password on the login page. Click 'Forgot Password' and follow instructions.",
    "Password reset requires access to your registered email account.",
    "Shipping policy: Orders arrive in 3-5 business days. Free shipping over $50.",
    "Return policy: 30-day money-back guarantee. Contact support for RMA number.",
    "ERROR_CODE_0x7B: Disk controller failure. Replace hardware or update drivers.",
    "Troubleshooting: If you see error 0x7B, check your disk connections.",
    "Account security: Enable two-factor authentication for added protection.",
    "Login issues? Try clearing your browser cache or resetting your password.",
]

# Create collection with different HNSW params
collection_fast = lab.create_collection_with_index(
    "fast_search",
    {"hnsw:M": 16, "hnsw:construction_ef": 100, "hnsw:search_ef": 50}
)
lab.index_documents(collection_fast, documents)

collection_accurate = lab.create_collection_with_index(
    "accurate_search",
    {"hnsw:M": 64, "hnsw:construction_ef": 400, "hnsw:search_ef": 200}
)
lab.index_documents(collection_accurate, documents)

# Benchmark
queries = ["password reset", "shipping time", "error code"]

print("\n" + "="*60)
print("BENCHMARK: Fast vs Accurate Index")
print("="*60)

fast_results = lab.benchmark_search(collection_fast, queries)
accurate_results = lab.benchmark_search(collection_accurate, queries)

print(f"\n⚡ FAST Index (M=16, ef=50):")
print(f"   Avg: {fast_results['avg_ms']:.2f}ms")
print(f"   P95: {fast_results['p95_ms']:.2f}ms")

print(f"\n🎯 ACCURATE Index (M=64, ef=200):")
print(f"   Avg: {accurate_results['avg_ms']:.2f}ms")
print(f"   P95: {accurate_results['p95_ms']:.2f}ms")

# Hybrid search demo
lab.demonstrate_hybrid_search(documents, "error 0x7B fix")