def evaluate_k_values(
    queries: list[str],
    ground_truth_docs: list[list[int]],
    similarities_matrix: np.ndarray,
    k_values: list[int] = [1, 3, 5, 10, 20]
) -> dict:
    """
    Evaluate recall@k for different K values.
    """
    results = {}
    
    for k in k_values:
        hits = 0
        total = len(queries)
        
        for i, (query, truth_indices) in enumerate(zip(queries, ground_truth_docs)):
            # Get top-k indices
            top_k_indices = np.argsort(similarities_matrix[i])[-k:][::-1]
            
            # Check if any ground truth is in top-k
            if any(idx in top_k_indices for idx in truth_indices):
                hits += 1
        
        recall = hits / total
        results[k] = {
            "recall": recall,
            "hits": hits,
            "total": total
        }
        
        print(f"K={k:2d}: Recall = {recall:.1%} ({hits}/{total})")
    
    return results

# Example usage
queries = ["password reset", "shipping time", "return policy"]
ground_truth = [[0, 1], [2], [3, 4]]  # Indices of correct docs

# Simulate similarity scores
similarities = np.random.rand(len(queries), 10)

evaluate_k_values(queries, ground_truth, similarities)