def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """L1 distance - sum of absolute differences."""
    return np.sum(np.abs(a - b))

# Example
a = np.array([1, 5])
b = np.array([4, 1])
print(f"Manhattan: {manhattan_distance(a, b)}")  # |1-4| + |5-1| = 3 + 4 = 7