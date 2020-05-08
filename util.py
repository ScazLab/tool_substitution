import numpy as np

def close_to(m, n, error=1e-6):
    return m >= n - error and m <= n + error

def normalize_vector(vector):
    if close_to(np.linalg.norm(vector), 0):
        return vector
    return vector / np.linalg.norm(vector)

def is_same_axis_matrix(mat_1, mat_2):
    """
    both mat_1 and both_2 are ordered, with the top row being the primary axis
    """
    difference = mat_1 - mat_2
    length = np.matmul(differece, difference.T)
    return close_to(length[0][0], 0) and close_to(length[1][1], 0) and close_to(length[2][2], 0)