import numpy as np

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Straight-line distance between two vectors."""
    return np.sqrt(np.sum((a - b) ** 2))

# Example
a = np.array([3, 4])
b = np.array([0, 0])
c = np.array([2.9, 3.9])

print(f"Distance A to origin: {euclidean_distance(a, b):.2f}")  # 5.00
print(f"Distance A to C: {euclidean_distance(a, c):.2f}")       # 0.14 (very close!)