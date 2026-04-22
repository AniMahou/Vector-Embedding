def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine of the angle between vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Example
a = np.array([3, 4])      # Length = 5
b = np.array([30, 40])    # Length = 50 (same direction!)
c = np.array([-3, -4])    # Opposite direction

print(f"A vs B: {cosine_similarity(a, b):.3f}")  # 1.000 (identical direction)
print(f"A vs C: {cosine_similarity(a, c):.3f}")  # -1.000 (opposite)