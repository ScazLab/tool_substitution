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
    compare_1 = mat_1.copy()
    compare_2 = mat_2.copy()
    
    if compare_2[0][0] * compare_1[0][0] < 0:
        compare_2[0] *= -1.0
    if compare_2[1][0] * compare_1[1][0] < 0:
        compare_2[1] *= -1.0
    if compare_2[2][0] * compare_1[2][0] < 0:
        compare_2[2] *= -1.0    
    
    difference = compare_1 - compare_2
    
    length = np.matmul(difference, difference.T)
    return close_to(length[0][0], 0) and close_to(length[1][1], 0) and close_to(length[2][2], 0)

def get_sorted_index(array, reverse_order=False):
    """ 
    get the order of the index of a one dimensions array
    @reverse_order: True, large->small
                    False, small->large
    """
    
    dtype = [('index', 'i4'), ('value', 'f8')]
    data = np.array([(i, array[i]) for i in range(len(array))], dtype = dtype)
    data.sort(order='value')
    order_index = data['index']
    if reverse_order:
        order_index = np.flip(order_index)
    return order_index

def min_point_distance(pc1, pc2):
    """
    pc1: 2 * n
    pc2: 2 * m
    threshold: int
    """
    length_pc1 = pc1.shape[1]
    length_pc2 = pc2.shape[1]
    
    # repeat each column length_pc1 times, for example, let length_pc1 = 3
    # then the array
    # x = np.array([[1,2],
    #               [3,4]])
    # becomes
    # array([[1, 1, 1, 2, 2, 2],
    #        [3, 3, 3, 4, 4, 4]])    
    repeated_pc2 = np.repeat(pc2, length_pc1, axis=1)
    # repeat the entire marix and repeat itself length_pc2 times, for example, let length_pc2 = 3
    # then the array x becomes
    # array([[1, 2, 1, 2, 1, 2],
    #        [3, 4, 3, 4, 3, 4]])
    repeated_pc1 = np.hstack([pc1] * length_pc2)
    difference = repeated_pc2 - repeated_pc1
    all_distance = np.multiply(difference, difference)
    all_distance = np.sqrt(np.sum(all_distance, axis=0))
    all_distance = np.reshape(all_distance, (length_pc2, length_pc1))
    min_distance = np.amin(all_distance, axis=1)
    avg_min_distance = np.mean(min_distance)
    
    print "avg_min_distance = ", avg_min_distance
    
    return avg_min_distance
    
def is_2d_point_cloud_overlap(pc1, pc2, threshold):
    """
    pc1: 2 * n
    pc2: 2 * m
    threshold: int
    """
    
    test_1 = min_point_distance(pc1, pc2) < threshold
    test_2 = min_point_distance(pc2, pc1) < threshold
    return test_1 and test_2