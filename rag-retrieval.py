import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from typing import List, Tuple, Optional

class ProductionRetrievalSystem:
    """Complete retrieval system with all Class 7 techniques."""
    
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        self.encoder = SentenceTransformer(embedding_model)
        self.client = chromadb.Client()
        self.collection = None
    
    def index_documents(
        self,
        documents: List[str],
        metadata: List[dict] = None,
        collection_name: str = "default"
    ):
        """Index documents with embeddings and metadata."""
        
        # Create or get collection
        try:
            self.client.delete_collection(collection_name)
        except:
            pass
        
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Generate embeddings
        print(f"📊 Generating embeddings for {len(documents)} documents...")
        embeddings = self.encoder.encode(documents).tolist()
        
        # Add to collection
        ids = [f"doc_{i}" for i in range(len(documents))]
        metadatas = metadata or [{}] * len(documents)
        
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✅ Indexed {len(documents)} documents")
    
    def search(
        self,
        query: str,
        k: int = 5,
        threshold: Optional[float] = None,
        filters: Optional[dict] = None,
        diversity: bool = False
    ) -> List[dict]:
        """
        Unified search interface with multiple strategies.
        """
        
        # Encode query
        query_emb = self.encoder.encode([query]).tolist()
        
        # Query collection
        n_results = k * 3 if diversity else k * 2 if threshold else k
        
        results = self.collection.query(
            query_embeddings=query_emb,
            n_results=n_results,
            where=filters
        )
        
        # Process results
        documents = results['documents'][0]
        distances = results['distances'][0]
        metadatas = results['metadatas'][0]
        
        # Convert distances to similarities
        similarities = [1 - d for d in distances]
        
        # Apply threshold
        if threshold:
            filtered = []
            for doc, sim, meta in zip(documents, similarities, metadatas):
                if sim >= threshold:
                    filtered.append((doc, sim, meta))
            
            documents, similarities, metadatas = zip(*filtered) if filtered else ([], [], [])
        
        # Apply diversity (MMR)
        if diversity and len(documents) > k:
            selected = self._mmr_selection(
                query_emb[0],
                documents,
                similarities,
                metadatas,
                k
            )
            return selected
        
        # Return top-k
        results = []
        for doc, sim, meta in zip(documents[:k], similarities[:k], metadatas[:k]):
            results.append({
                "document": doc,
                "similarity": sim,
                "metadata": meta
            })
        
        return results
    
    def _mmr_selection(
        self,
        query_emb: List[float],
        documents: List[str],
        similarities: List[float],
        metadatas: List[dict],
        k: int,
        lambda_param: float = 0.5
    ) -> List[dict]:
        """Maximal Marginal Relevance for diverse results."""
        
        if len(documents) <= k:
            return [
                {"document": d, "similarity": s, "metadata": m}
                for d, s, m in zip(documents, similarities, metadatas)
            ]
        
        # Get embeddings for candidates
        candidate_embs = self.encoder.encode(documents)
        
        selected_indices = []
        
        for _ in range(k):
            best_idx = -1
            best_score = -float('inf')
            
            for i in range(len(documents)):
                if i in selected_indices:
                    continue
                
                relevance = similarities[i]
                
                if selected_indices:
                    diversity = max(
                        cosine_similarity([candidate_embs[i]], [candidate_embs[j]])[0][0]
                        for j in selected_indices
                    )
                else:
                    diversity = 0
                
                mmr = lambda_param * relevance - (1 - lambda_param) * diversity
                
                if mmr > best_score:
                    best_score = mmr
                    best_idx = i
            
            selected_indices.append(best_idx)
        
        return [
            {
                "document": documents[i],
                "similarity": similarities[i],
                "metadata": metadatas[i]
            }
            for i in selected_indices
        ]
    
    def evaluate_retrieval(
        self,
        test_queries: List[str],
        ground_truth_indices: List[List[int]],
        k: int = 5
    ) -> dict:
        """Evaluate retrieval performance."""
        
        metrics = {
            "recall@k": 0,
            "precision@k": 0,
            "mrr": 0,  # Mean Reciprocal Rank
            "queries_evaluated": len(test_queries)
        }
        
        for query, truth in zip(test_queries, ground_truth_indices):
            results = self.search(query, k=k)
            
            # Get result indices (from metadata or parse)
            result_ids = [r.get("metadata", {}).get("id", -1) for r in results]
            
            # Recall: Did we get any correct document?
            if any(idx in result_ids for idx in truth):
                metrics["recall@k"] += 1
            
            # Precision: What fraction of results are correct?
            correct = sum(1 for idx in result_ids if idx in truth)
            metrics["precision@k"] += correct / k if k > 0 else 0
            
            # MRR: 1 / rank of first correct result
            for rank, idx in enumerate(result_ids, 1):
                if idx in truth:
                    metrics["mrr"] += 1 / rank
                    break
        
        # Average
        n = len(test_queries)
        metrics["recall@k"] /= n
        metrics["precision@k"] /= n
        metrics["mrr"] /= n
        
        return metrics

# Demo
system = ProductionRetrievalSystem()

documents = [
    "Password reset: Click 'Forgot Password' on the login page.",
    "If you forgot your password, email support@company.com.",
    "Shipping takes 3-5 business days for standard delivery.",
    "Return policy: 30-day money-back guarantee.",
    "To change your password, go to Account Settings.",
    "Two-factor authentication adds extra security to your account.",
]

metadata = [
    {"category": "account", "id": 0},
    {"category": "account", "id": 1},
    {"category": "shipping", "id": 2},
    {"category": "returns", "id": 3},
    {"category": "account", "id": 4},
    {"category": "security", "id": 5},
]

system.index_documents(documents, metadata, "help_center")

# Test searches
print("\n" + "="*60)
print("SEARCH: 'forgot password'")
print("="*60)
for r in system.search("forgot password", k=3):
    print(f"  [{r['similarity']:.3f}] {r['document']}")

print("\n" + "="*60)
print("SEARCH WITH THRESHOLD: 'return an item' (threshold=0.5)")
print("="*60)
for r in system.search("return an item", k=3, threshold=0.5):
    print(f"  [{r['similarity']:.3f}] {r['document']}")

print("\n" + "="*60)
print("SEARCH WITH FILTER: 'password' (category='account')")
print("="*60)
for r in system.search("password", k=3, filters={"category": "account"}):
    print(f"  [{r['similarity']:.3f}] {r['document']}")

print("\n" + "="*60)
print("DIVERSE SEARCH: 'account security'")
print("="*60)
for r in system.search("account security", k=3, diversity=True):
    print(f"  [{r['similarity']:.3f}] {r['document']}")

# Evaluation
test_queries = ["reset password", "shipping time"]
ground_truth = [[0, 1, 4], [2]]

metrics = system.evaluate_retrieval(test_queries, ground_truth, k=3)
print("\n" + "="*60)
print("EVALUATION METRICS")
print("="*60)
for metric, value in metrics.items():
    print(f"  {metric}: {value:.3f}")