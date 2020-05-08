import numpy as np

def close_to(m, n, error=1e-6):
    return m >= n - error and m <= n + error

def normalize_vector(vector):
    if close_to(np.linalg.norm(vector), 0):
        return vector
    return vector / np.linalg.norm(vector)