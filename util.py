import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot
from mpl_toolkits.mplot3d import Axes3D, art3d
from sklearn.metrics.pairwise import cosine_similarity

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
    pc1: 3 * n
    pc2: 3 * m
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

def weighted_min_point_distance(src_pc, sub_pc, src_cntct_pnt, sigma=1):
    """
    Same as min_point_distance but weights sub_pc pnts by how far they are
    from src_cntct_pnt. This is because we really only care about the local
    properties of the tool near the contact pnt.
    """
    length_pc1 = src_pc.shape[1]
    length_pc2 = sub_pc.shape[1]

    weights = np.apply_along_axis(lambda col: 1.0 - rbf(src_cntct_pnt, col, sigma) ,
                                  arr=sub_pc, axis=0)

    print("weights")

    # repeat each column length_pc1 times, for example, let length_pc1 = 3
    # then the array
    # x = np.array([[1,2],
    #               [3,4]])
    # becomes
    # array([[1, 1, 1, 2, 2, 2],
    #        [3, 3, 3, 4, 4, 4]])
    repeated_pc2 = np.repeat(sub_pc, length_pc1, axis=1)
    # repeat the entire marix and repeat itself length_pc2 times, for example, let length_pc2 = 3
    # then the array x becomes
    # array([[1, 2, 1, 2, 1, 2],
    #        [3, 4, 3, 4, 3, 4]])
    repeated_pc1 = np.hstack([src_pc] * length_pc2)
    difference = repeated_pc2 - repeated_pc1
    all_distance = np.multiply(difference, difference)
    all_distance = np.sqrt(np.sum(all_distance, axis=0))
    all_distance = np.reshape(all_distance, (length_pc2, length_pc1))
    min_distance = np.amin(all_distance, axis=1)
    min_distance *= weights
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



def rbf(x1, x2, sigma = 1):
    """
    Radial basis function. Returns vals btwn [0-1].
    1 is returned if x1 == x2.
    """
    diff = x1 - x2
    return np.exp( -(np.dot(diff, diff)) / (2. * (sigma ** 2)))


def rotation_matrix_from_box_rots(sub_dir,src_dir):
    sub_dir  = sub_dir / np.linalg.norm(sub_dir)
    src_dir = src_dir / np.linalg.norm(src_dir)
    # print "INITIAL ORIENTATION"
    # visualize_vectors(np.vstack([  src_dir, sub_dir]))

    print sub_dir
    print src_dir
    v = np.cross(sub_dir, src_dir)
    c = np.dot(sub_dir, src_dir)
    # cos_sim = cosine_similarity(sub)
    theta = np.arccos(c)

    eps = .15
    is_parallel =  np.isclose(np.linalg.norm(v), np.zeros(1), atol=eps).item()
    is_same_dir  = np.isclose(theta,  np.zeros(1), atol=eps).item()
    print "SAME DIR: ", is_same_dir
    print "IS PARALLEL: ", is_parallel

    # rots = [np.pi /2 , np.pi, 3 * np.pi / 2, 2 * np.pi]
    rots = np.arange(np.pi /2, 2*np.pi, step=.10)
    scores = []

    if is_parallel and is_same_dir:
        print "PC already pointing in right dir. No rotation required."
        return np.identity(3)
    elif not is_parallel:
        rot_vec = v

    else:
        a = src_dir
        rot_vec = np.array([a[1], -a[0], 0.]) if a[2] < a[0] else np.array([0, -a[2], a[1]])

    for rot in rots:
        rot_vec = rot_vec * rot
        R = Rot.from_rotvec(rot_vec)
        R = R.as_dcm()
        rot_sub_dir = R.dot(sub_dir.T)
        # visualize_vectors(np.vstack([  src_dir, rot_sub_dir, rot_vec  ]))
        # cosine_similarity(src_dir.reshape(1,-1),
        #                                     rot_sub_dir.reshape(1,-1)).item()))

        dir_score = np.arccos(np.dot(src_dir, rot_sub_dir))
        scores.append((R, dir_score)) 

    best_R, fitness = min(scores, key=lambda score: score[1])
    print "BEST ROTATION COSINE SIM ", fitness
    visualize_vectors(np.vstack([  src_dir, best_R.dot(sub_dir.T)]))

    return best_R


def rotation_matrix_from_vectors(a, b):
    """ Find the rotation matrix that aligns a to b
    :param a: A 3d "source" vector
    :param b: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to a, aligns it with b.
    """
    a, b = (a / np.linalg.norm(a)).reshape(3), (b / np.linalg.norm(b)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    theta = np.arccos(c)

    eps = .15
    is_parallel =  np.isclose(np.linalg.norm(v), np.zeros(1), atol=eps).item()
    is_opp_dir  = np.isclose(theta, np.pi, atol=eps).item()
    is_same_dir  = np.isclose(theta,  np.zeros(1), atol=eps).item()

    print "V ", v
    print "IS PARALLEL {}: {} ".format(is_parallel, np.linalg.norm(v))
    print "THETA ", theta
    print "SAME DIR: ", is_same_dir
    print "OPP DIR: ", is_opp_dir


    if is_parallel:
        if is_same_dir:
            return np.identity(3)
        elif is_opp_dir:
            v = np.array([a[1], -a[0], 0.]) if a[2] < a[0] else np.array([0, -a[2], a[1]])
            v *= np.pi
            # if not a[2] == 0.:
            #     v = np.array([1.,1., -(a[0] + a[1]) / a[2]])
            # elif not a[1] == 0.:
            #     v = np.array([1.,-(a[0] + a[2]) / a[1], 1.])
            # else:
            #     v = np.array([-(a[1] + a[2]) / a[0], 1., 1.])
            # v = v /np.linalg.norm(v)

    s = np.linalg.norm(v)
    v = v /s
    print "NEW V ", v
    # if is_parallel:
        # rot_vec = np.array([a[1], -a[0], 0.]) if a[2] < a[0] else np.array([0, -a[2], a[1]])
        # rot_vec *= np.pi
        # print "ROT VEC ", rot_vec
        # R = Rot.from_rotvec(rot_vec)
        # R = R.as_dcm()
        # # R[:, 0] = -1 * R[:, 0]

    # else:
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))


    # if R.dot(src_c_side_dir)
    return R



def visualize_two_pcs(pnts1, pnts2, s1=None, s2=None):
    """
    Just plots two pointclouds for easy comparison.
    """


    faces = {'a':[1,2,6,7],
             'b':[2,7,3,8],
             'c':[5,8, 6,7],
             'd':[1,6,5,0],
             'e':[5,8,3,2],
             'f':[0,1,2,3]}

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    print "Pnts 1 mean: {} Pnts 2 mean: {}".format(pnts1.mean(axis=0),
                                                   pnts2.mean(axis=0))


    if not s1 is None:
        bb, s = s1
        face = [bb[faces[s], :]]
        side = art3d.Poly3DCollection(face)
        side.set_color('r')
        ax.add_collection3d(side)

    if not s2 is None:
        bb, s = s2
        face = [bb[faces[s], :]]
        side = art3d.Poly3DCollection(face)
        side.set_color('b')
        ax.add_collection3d(side)


    ax.axis('equal')
    ax.scatter(xs=pnts1[:, 0], ys=pnts1[:, 1], zs=pnts1[:, 2], c='r')
    ax.scatter(xs=pnts2[:, 0], ys=pnts2[:, 1], zs=pnts2[:, 2], c='b')



    plt.show()
