class HNSWTuner:
    """Guide for tuning HNSW parameters."""
    
    @staticmethod
    def recommend_params(
        num_vectors: int,
        dims: int,
        latency_budget_ms: int,
        recall_target: float
    ) -> dict:
        """
        Recommend HNSW parameters based on requirements.
        """
        params = {}
        
        # M: Connections per node
        if dims <= 384:
            params['M'] = 16  # Smaller dims need fewer connections
        elif dims <= 1024:
            params['M'] = 32
        else:  # 1536+
            params['M'] = 64
        
        # ef_construction: Build-time accuracy
        if recall_target >= 0.99:
            params['ef_construction'] = 500
        elif recall_target >= 0.95:
            params['ef_construction'] = 200
        else:
            params['ef_construction'] = 100
        
        # ef_search: Query-time accuracy
        if latency_budget_ms < 10:
            params['ef_search'] = 50
        elif latency_budget_ms < 50:
            params['ef_search'] = 100
        else:
            params['ef_search'] = 200
        
        # Memory estimate
        memory_per_vector = dims * 4 + params['M'] * 8  # Vector + edges
        total_memory_gb = (num_vectors * memory_per_vector) / 1e9
        
        params['estimated_memory_gb'] = total_memory_gb
        
        return params

# Example
tuner = HNSWTuner()
params = tuner.recommend_params(
    num_vectors=1_000_000,
    dims=1536,
    latency_budget_ms=20,
    recall_target=0.98
)

print("Recommended HNSW Parameters:")
for key, value in params.items():
    print(f"  {key}: {value}")