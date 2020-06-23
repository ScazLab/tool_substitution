import numpy as np
from numpy import cross
from numpy.linalg import norm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
from matplotlib.patches import Rectangle

from tool_pointcloud import ToolPointCloud
from sample_pointcloud import GeneratePointcloud
from util import (min_point_distance, rotation_matrix_from_vectors,
                  weighted_min_point_distance, visualize_two_pcs,
                  rotation_matrix_from_box_rots, visualize_vectors)

from scipy.spatial.transform import Rotation as Rot

from get_target_tool_pose import get_T_from_R_p, get_pnts_world_frame, get_aruco_world_frame


def shrink_pc(pc):
    shrink_ratio = .3
    pnts = pc.get_pc_bb_axis_frame_centered()
    x_min = pnts.min(axis=0)[0]
    x_max = pnts.max(axis=0)[0]
    x_len = x_max - x_min
    
    pnts = pnts[np.where(pnts[:,0] > x_min + (x_len * shrink_ratio)), :][0]
    print pnts

    return ToolPointCloud(pnts)


class ArucoStuff(object):
    def __init__(self, pc):
        "docstring"
        self.pc = pc


    def get_aruco_intial_T(self):
        pnts     = self.pc.get_unnormalized_pc()
        centroid = pnts.mean(axis =0)
        # R        = self.pc.get_axis()

        return get_T_from_R_p(centroid)

    def percieve_aruco_T(self):
        p = np.random.unform(3)
        R = np.array([
            [0,  1, 0],
            [-1, 0, 0],
            [0,  0, -1]
        ])

        return get_T_from_R_p(p, R)

    def get_src_tool_T(self):
        p = np.random.unform(3)
        R = np.array([
            [0,  -1, 0],
            [1, 0, 0],
            [0,  0, 1]
        ])



def visualize_candidate_sides(pnts, pc_bb, cand_sides, centroids, proj, c_point=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    bb = pc_bb.bb
    faces = {'a': bb[[2,7,6,1], :],
             'b': bb[[3,8,7,2], :],
             'c': bb[[7,8,5,6], :],
             'd': bb[[1,6,5,0], :],
             'e': bb[[0,5,8,3], :],
             'f': bb[[3,2,1,0], :]}

    for face in faces.keys():
        c = 'g' if face in cand_sides else 'r'
        side = art3d.Poly3DCollection([faces[face]], alpha=.3)
        side.set_color(c)
        ax.add_collection3d(side)
    # Tool pointcloud
    ax.scatter(xs=pnts[:,0], ys=pnts[:, 1], zs=pnts[:, 2], c='b')

    # Centroids
    for c in centroids:
        ax.scatter(xs=c[0], ys=c[1], zs=c[ 2], c='y', s=450)


    for p in proj:
        ax.scatter(xs=p[0], ys=p[1], zs=p[ 2], c='black', s=350)

    if not c_point is None:
        ax.scatter(xs=c_point[0], ys=c_point[1], zs=c_point[ 2], c='r', s=350)


    plt.show()

def visualize_bb_pnt_proj(bb, pnt, proj):
    """Visualize points and bounding box"""

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    centroid = np.vstack([pnt, bb.mean(axis=0)])
    ax.axis('equal')

    ax.scatter(xs=pnt[0], ys=pnt[1], zs=pnt[2], c='r', s=15)
    ax.scatter(xs=proj[0], ys=proj[1], zs=proj[2], c='g', s=30)

    ax.plot(centroid[:,0], centroid[:,1], centroid[:, 2], c='r')

    ax.plot(bb.T[0], bb.T[1], bb.T[2], c='r')
    ax.plot(bb[(1, 6), :].T[0], bb[(1, 6), :].T[1], bb[(1, 6), :].T[2], c='r')
    ax.plot(bb[(2, 7), :].T[0], bb[(2, 7), :].T[1], bb[(2, 7), :].T[2], c='r')
    ax.plot(bb[(3, 8), :].T[0], bb[(3, 8), :].T[1], bb[(3, 8), :].T[2], c='r')

    plt.show()


def calc_seg_centroids(segs, pc):
    centroids = []

    for s in segs:
        pnts = pc.get_pnts_in_segment(s)
        centroids.append(pnts.mean(axis=0))

    return centroids



def visualize(pnts, cp=None, segments=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if segments is None:
        c = 'b'
    else:
        c = segments

    ax.axis('equal')
    ax.scatter(xs=pnts[:, 0], ys=pnts[:, 1], zs=pnts[:, 2], c=c)
    if not cp is None:
        ax.scatter(xs=cp[0], ys=cp[1], zs=cp[2], c='r', s=550)
    plt.show()


def project_pnt_on_bb_sides(pnt, bb):
    bb_sides = bb_to_planes(bb)
    scores = []
    projs = []

    for k, v in bb_sides.items():
        p, _, c, n = v
        n = n / norm(n) # normalize
        q = proj_on_plane(pnt, (c, n))
        projs.append(q)
        print "{} proj: {}".format(k, q)
        scores.append((k, norm(q - pnt)))

    print(scores)
    side = min(scores, key=lambda t: t[1])

    return side[0], bb_sides, projs

def get_closest_bb_side_to_pnt(pnt, bb):
    """
    Determines which bb side pnt is closest to pnt
    by comparing point against side centroids.
    returns: the closest side: str
             A list of bb side info: tuple
             The centroids of each bb side: ndarray
    """
    bb_sides = bb_to_planes(bb)
    scores = []
    centroids = []

    for k, v in bb_sides.items():
        p, _, c, n = v
        centroids.append(c)
        scores.append((k, norm(c - pnt)))

    for k, s in scores:
        print "{}: {}".format(k, s)
    side = min(scores, key=lambda t: t[1])

    return side[0], bb_sides, centroids


def proj_on_plane(q, plane):
    """
    Orthogonal projection of pnt q on plane
    """
    p, norm = plane

    return q  - np.dot(q - p, norm ) * norm



def bb_to_planes(bb, use_areas=True):
    """
    Given a bb (10x3 ndarray) caluclate planes for each face of the box.
    Returns dict where keys are [a-f; str] faces of box and vals are tuples.
    Tuples contain: a pnt on the face (ndarray) the relative size of face (int, 0-2),
                    the centroid (ndarray), and the norm of the plane (ndarray).
    """
    bb_pnts = bb.bb
    # axes = bb.get_normalized_axis()
    axes = bb.get_unnormalized_unordered_axis()
    ax1  = axes[:, 0]
    ax2  = axes[:, 1]
    ax3  = axes[:, 2]

    bb_areas = [norm(bb.norms[0]) * norm(bb.norms[1]),
                norm(bb.norms[0]) * norm(bb.norms[2]),
                norm(bb.norms[1]) * norm(bb.norms[2])]

    bb_ratios = [min(abs(bb.norms[0]), abs(bb.norms[1])) / max(abs(bb.norms[0]),
                                                               abs(bb.norms[1])),
                min(abs(bb.norms[0]), abs(bb.norms[2])) / max(abs(bb.norms[0]),
                                                               abs(bb.norms[2])),
                min(abs(bb.norms[1]), abs(bb.norms[2])) / max(abs(bb.norms[1]),
                                                               abs(bb.norms[2]))]


    faces = {'a': bb_pnts[[2,7,6,1], :],
             'b': bb_pnts[[3,8,7,2], :],
             'c': bb_pnts[[7,8,5,6], :],
             'd': bb_pnts[[1,6,5,0], :],
             'e': bb_pnts[[0,5,8,3], :],
             'f': bb_pnts[[3,2,1,0], :]}

    # bb side areas from lowest to highest
    if use_areas:
        side_ranks = bb_areas
        sorted_ranks  = sorted(bb_areas)
    else:
        side_ranks = bb_ratios
        sorted_ranks  = sorted(bb_ratios)

    # print("BB AREAS: {}".format(bb_areas))

    # Get normal by by taking cross product of of bb side face
    n1 = ax1
    # Calculate area of face
    a1 = side_ranks[2]
    a1 = np.argmin(np.abs(sorted_ranks - a1))
    # Centroid of phase
    c1 = faces['a'].mean(axis=0)
    s1 = (bb_pnts[7,:], a1, c1, n1)

    n2 = ax2
    a2 = side_ranks[1]
    a2 = np.argmin(np.abs(sorted_ranks - a2))
    c2 = faces['b'].mean(axis=0)
    s2 = (bb_pnts[8,:], a2, c2, n2)

    n3 = ax3
    a3 = side_ranks[0]
    a3 = np.argmin(np.abs(sorted_ranks - a3))
    c3 = faces['c'].mean(axis=0)
    s3 = (bb_pnts[8,:], a3, c3, n3)


    n4 = ax2 * -1.
    a4 = side_ranks[1]
    a4 = np.argmin(np.abs(sorted_ranks - a4))
    c4 = faces['d'].mean(axis=0)
    s4 = (bb_pnts[6,:], a4, c4, n4)


    n5 = ax1 * -1.
    a5 = side_ranks[2]
    a5 = np.argmin(np.abs(sorted_ranks - a5))
    c5 = faces['e'].mean(axis=0)
    s5 = (bb_pnts[5,:], a5, c5,  n5)

    n6 = ax3 * -1.
    a6 = side_ranks[0]
    a6 = np.argmin(np.abs(sorted_ranks - a6))
    c6 = faces['f'].mean(axis=0)
    s6 = (bb_pnts[2,:], a6, c6, n6)

    return {'a':s1, 'b':s2, 'c':s3, 'd':s4, 'e':s5, 'f':s6}


class ToolSubstitution(object):
    def __init__(self, src_tool_pc, sub_tool_pc, visualize=False):
        "docstring"
        self.src_tool = src_tool_pc
        self.sub_tool = sub_tool_pc
        self.sigma = 1.0 # variance of radial basis kernel
        self.visualize = visualize
        # self.alignment_thresh = .33
        self.alignment_thresh = .0001

    def _center_and_align_pnts(self, pc):
        """
        Creates a centered and aligned ToolPointCloud from unaligned ToolPointCloud
        """
        pnts = pc.get_pc_bb_axis_frame_centered()
        # Add the segment labels back in.
        pnts = np.vstack([pnts.T, pc.segments]).T

        return ToolPointCloud(pnts)

    def _calc_best_orientation(self, src_pnts, sub_pnts, Rs, c_pnt):
        """
        Given a list of rotation matrices, R, determine f
        """
        if not Rs:
            Rs.append(np.identity(3))

        scores = []
        aligned_pnts = []
        for R in Rs:
            sub_pnts_rot = R.dot(sub_pnts.T)
            visualize_two_pcs(src_pnts, sub_pnts_rot.T)
            # visualize_two_pcs(sub_pnts, sub_pnts_rot.T)

            # Average scores due to slight asymmetry in distance metric.
            # score1 = min_point_distance(src_pnts.T, sub_pnts_rot)
            # score2 = min_point_distance(sub_pnts_rot, src_pnts.T)
            score1 = weighted_min_point_distance(src_pnts.T, sub_pnts_rot, c_pnt, self.sigma)
            score2 = weighted_min_point_distance(sub_pnts_rot, src_pnts.T, c_pnt, self.sigma)

            score = (score1 + score2) / 2.0
            scores.append(score)
            aligned_pnts.append(sub_pnts_rot.T)

        i = np.argmin(scores)

        print "rotation {} is best".format(i + 1)
        return Rs[i], aligned_pnts[i], scores[i]


    def _get_closest_pnt(self, pnt, pntcloud):
        """
        returns the point in pntcloud closest to pnt.
        """
        print "SHAPE ", pntcloud.shape
        diffs = np.apply_along_axis(lambda row: np.linalg.norm(pnt - row),
                                    axis=1, arr=pntcloud)

        idx = np.argmin(diffs).item()
        print("{}, type:{}".format(idx, type(idx)))
        return idx

    def _align_pnts(self, src_pc, sub_pc):
        """
        Scale sub_pc to and then detemine most similar orientation to src_pc
        Returns ndarray of sub_pc pnts in best orientation.
        """


        scaled_sub_pnts, _ = sub_pc.scale_pnts_to_target(src_pc)
        scaled_sub_pc = ToolPointCloud(scaled_sub_pnts)

        # aligned_sub_pnts = scaled_sub_pc.get_pc_bb_axis_frame_centered()
        # aligned_src_pnts = src_pc.get_pc_bb_axis_frame_centered()
        aligned_sub_pnts = scaled_sub_pc.pnts
        aligned_src_pnts = src_pc.pnts

        # Test current orientation
        R1 = np.identity(3)
        # Test orientation rotated 180 degrees along first axis
        R2 = np.identity(3)
        R2[:, 0] =  -1. * R1[:,0]

        src_cntct_pnt = self.centered_src_pc.get_pnt(self.src_tool.contact_pnt_idx)

        return self._calc_best_orientation(src_pc.pnts,
                                           scaled_sub_pc.pnts,
                                           [R1, R2],
                                           src_cntct_pnt)



    def _get_sub_tool_action_part(self):

        self.centered_src_pc = self._center_and_align_pnts(self.src_tool)
        self.centered_sub_pc = self._center_and_align_pnts(self.sub_tool)

        # Find the best alignment of the sub tool based on 3d haming distance.
        R, aligned_sub_pnts, score = self._align_pnts(self.centered_src_pc,
                                                  self.centered_sub_pc)



        # visualize_two_pcs(src_pnts, aligned_sub_pnts)
        src_contact_pnt = self.centered_src_pc.get_pnt(self.src_tool.contact_pnt_idx)
        sub_action_pnt = self._get_closest_pnt(src_contact_pnt, aligned_sub_pnts)

        return (self.sub_tool.get_segment_from_point(sub_action_pnt),
                aligned_sub_pnts, R, score)

    def _score_segment(self, sub_pc, src_action_pc, sub_action_seg, src_cntc_info):
        """
        Aligns a segment of sub tool to src tool contact surface and calculates
        contact point and alignment score for segment.
        """

        # The relative size of src contact surface, the surface normal and
        # a bool signifying whether the contact surface is also a surface
        # adjacent to another segment (This might happen for a pulling obj).
        rel_size, src_c_side_norm, src_c_side_centorid, cntct_on_adj_side = src_cntc_info
        sub_other_segs = [s for s in sub_pc.segment_list \
                          if not s == sub_action_seg]

        print("sub other segs: {}".format(sub_other_segs))

        # Get centroids of bb sides in order to help determine which sides
        # Should be considered as contact surfaces.

        sub_seg_pnts = sub_pc.get_pnts_in_segment(sub_action_seg)
        # sub_seg_pnts = np.vstack([sub_seg_pnts.T, self.sub_tool.segments]).T
        sub_seg_pc = ToolPointCloud(sub_seg_pnts, normalize=False)
        sub_seg_bb = sub_seg_pc.bb

        sub_seg_centroids = calc_seg_centroids(sub_other_segs,
                                               sub_pc)
        # Scale sub action part to src action part size for better comparison.
        scaled_sub_action_pnts, scale_f = sub_seg_pc.scale_pnts_to_target(src_action_pc)
        scaled_sub_seg_pc = ToolPointCloud(scaled_sub_action_pnts, normalize=False)

        sub_action_bb = sub_seg_pc.bb
        # visualize_two_pcs(scaled_sub_seg_pc.pnts, sub_seg_pc.pnts)

        sub_seg_centroids = calc_seg_centroids(sub_other_segs,
                                               sub_pc)

        # visualize_two_pcs(sub_seg_pc.pnts, self.centered_sub_pc.pnts)
        src_cntct_pnt = self.centered_src_pc.get_pnt(self.src_tool.contact_pnt_idx)

        sub_adj_sides = []
        sub_all_sides = {}
        projs = []

        for c in sub_seg_centroids:
            s, sub_all_sides, projs = get_closest_bb_side_to_pnt(c ,sub_seg_bb)
            sub_adj_sides.append(s)

        # Remove duplicates
        sub_adj_sides = list(set(sub_adj_sides))
        cand_sides = [s for s in sub_all_sides.keys() if not s in sub_adj_sides]


        if cntct_on_adj_side and sub_adj_sides:
            # cand_sides = [s for s in sub_all_sides.keys() if s in sub_adj_sides]
            cand_sides = sub_adj_sides
        # Otherwise we consider non-adj sides11
        else:
            print "REL SIZE OF SRC CNTCT SURFACE: ", rel_size
            cand_sides = [s for s in sub_all_sides.keys() if not s in sub_adj_sides]
            # sub_same_size_sides = [k for k in cand_sides if sub_all_sides[k][1] == rel_size]
            # cand_sides = sub_same_size_sides if sub_same_size_sides else cand_sides
            # print "SUB SAME SIDES: ", sub_same_size_sides
            print "CANDIDATE SIDES: ", cand_sides



        if self.visualize:
            visualize_candidate_sides(sub_seg_pc.pnts,
                                        sub_seg_bb,
                                        cand_sides,
                                        sub_seg_centroids,
                                        projs)


        Rs = [] # Stores all candidate rotation martices
        aligned_Rs = []
        # First test sides of sub tool that are same relative size
        # to contact side of src tool.
        for s in cand_sides:
            sub_dir = sub_all_sides[s][3]
            sub_centroid = sub_all_sides[s][2]

            # If norms are already parallel, then rotating
            # them gives wrong answer for some reason.
            print "CURR DIR ", sub_dir
            print "GOAL DIR ", src_c_side_norm
            R = rotation_matrix_from_box_rots(sub_dir, src_c_side_norm)
            print "RESULT ", R.dot(sub_dir)
            new_seg_bb = R.dot(sub_seg_bb.bb.T).T



            aligned_Rs.append(( R, src_c_side_centorid, sub_centroid ))
            Rs.append(R)

        # Determine the contact point of src tool in action part segment point cloud
        score = self._calc_best_orientation(src_action_pc.get_normalized_pc(),
                                            scaled_sub_seg_pc.get_normalized_pc(),
                                            Rs,
                                            src_cntct_pnt)

        return score


    def get_contact_pnt(self):
        # Get segment of contact point of src tool.
        src_action_seg = self.src_tool.get_segment_from_point(self.src_tool.contact_pnt_idx)
        # Determine the corresponding segment based on shape in sub tool.
        sub_action_seg, scaled_sub_pnts, init_R, score = self._get_sub_tool_action_part()

        # If a good alignment cannot be found using the entire sub tool,
        # Then we test all segments of the sub tool against the action segment.
        alignment_thresh = self.centered_src_pc.bb.dim_lens[0] * self.alignment_thresh
        print "ALIGNMENT THRESH ", alignment_thresh
        print "SCORE ", score
        if score > alignment_thresh:
            print "TESTING ALL SEGMENTS"
            sub_segments =  self.sub_tool.segment_list

        # Otherwise we use the estimated action segment found above.
        else:
            print "TESTING SEGMENT ", sub_action_seg
            sub_segments = [sub_action_seg]

        scaled_sub_pnts = np.vstack([scaled_sub_pnts.T, self.sub_tool.segments]).T
        scaled_sub_pc = ToolPointCloud(scaled_sub_pnts, normalize=False)

        print("src action seg: {}".format(src_action_seg))
        print("sub action seg: {}".format(sub_action_seg))

        # Segments of tools that aren't the action segments
        src_other_segs = [s for s in self.centered_src_pc.segment_list \
                          if not s == src_action_seg]

        print("src other segs: {}".format(src_other_segs))

        # Get centroids of bb sides in order to help determine which sides
        # Should be considered as contact surfaces.
        src_seg_centroids = calc_seg_centroids(src_other_segs, self.centered_src_pc)
        # sub_seg_centroids = calc_seg_centroids(sub_other_segs, scaled_sub_pc)

        # Get pointclouds corresponding to action parts of each tool.
        src_action_pnts = self.centered_src_pc.get_pnts_in_segment(src_action_seg)
        src_action_pc = ToolPointCloud(src_action_pnts, normalize=False)
        src_action_pnts = src_action_pc.pnts
        src_action_bb = src_action_pc.bb

        src_cntct_idx = self.centered_src_pc.idx_to_segment_idx(self.src_tool.contact_pnt_idx)
        src_cntct_pnt = src_action_pc.get_pnt(src_cntct_idx)

        src_adj_sides = [] # stores sides of action part bb that conenct to other segments

        projs = []
        for c in src_seg_centroids:
            s,src_all_sides, _ = get_closest_bb_side_to_pnt(c ,src_action_bb)
            src_adj_sides.append(s)

        # src_c_side, src_all_sides, _= get_closest_bb_side_to_pnt(src_cntct_pnt,
        #                                                           src_action_bb)
        # Project contact point on bb sides to determine closest bb side.
        src_c_side, src_all_sides, projs= project_pnt_on_bb_sides(src_cntct_pnt,
                                                              src_action_bb)

        print "SOURCE CONTACT SIDE: ", src_c_side

        if self.visualize:
            visualize_candidate_sides(src_action_pnts,
                                      src_action_bb,
                                      [src_c_side],
                                      src_seg_centroids,
                                      projs,
                                      src_cntct_pnt)

        _, rel_size, src_c_centroid, n = src_all_sides[src_c_side]

        # If the contacting side is on a side that connects to another segment,
        # Then the tool is likely being used for pulling, and similar sides
        # ought to be considered on the sub tool.
        cntct_on_adj_side = src_c_side in src_adj_sides

        src_cntct_info = (rel_size, n, src_c_centroid, cntct_on_adj_side)

        scores = []
        for seg in sub_segments:
            # score = self._score_segment(scaled_sub_pc, src_action_pc,
            #                             seg, src_cntct_info)
            score = self._score_segment(self.centered_sub_pc, src_action_pc,
                                        seg, src_cntct_info)
            scores.append((score[0], score[1], score[2], seg))

        best_R, best_sub_pnts, _,  seg  = min(scores, key=lambda s: s[2])


        # src_cntct_idx = self.centered_src_pc.idx_to_segment_idx(self.src_tool.contact_pnt_idx)
        src_cntct_idx = self.src_tool.idx_to_segment_idx(self.src_tool.contact_pnt_idx)
        # src_action_contact_pnt = src_action_pc.get_pnt(src_cntct_idx)
        src_action_contact_pnt = src_action_pc.get_normalized_pc()[src_cntct_idx, :]
        src_orig_contact_pnt = self.src_tool.get_normalized_pc()[src_cntct_idx, :]
        # src_orig_contact_pnt = self.centered_src_pc.get_normalized_pc()[src_cntct_idx, :]
        sub_action_c_pnt_idx = self._get_closest_pnt(src_action_contact_pnt,
                                                    best_sub_pnts)
        # sub_action_c_pnt_idx = self._get_closest_pnt(src_orig_contact_pnt,
        #                                             best_sub_pnts)
        # Get pnts corresponding to these segments
        # sub_cntct_pnt = centered_sub_pc.get_pnt(sub_contact_pnt_idx)
        sub_contact_pnt_idx = self.sub_tool.segment_idx_to_idx(seg,
                                                                 sub_action_c_pnt_idx)
        sub_cntct_pnt = self.sub_tool.get_pnt(sub_contact_pnt_idx)



        if self.visualize:
            visualize(self.centered_src_pc.pnts, src_cntct_pnt,
                    self.centered_src_pc.segments)
            visualize(self.sub_tool.get_normalized_pc(),
                    sub_cntct_pnt, self.sub_tool.segments)


        ret_R = self.centered_sub_pc.get_axis().dot(best_R)

        return ret_R, sub_cntct_pnt



    def get_random_contact_pnt(self):
        """
        Get a random contact point and rotation matrix for substitute tool.
        """

        # src_action_seg = self.src_tool.get_segment_from_point(self.src_tool.contact_pnt_idx)
        scaled_sub_pnts, _ = self.sub_tool.scale_pnts_to_target(self.src_tool)
        R = Rot.random(num=1).as_dcm() # Generate a radom rotation matrix.
        rot_sub_pnts = R.dot(scaled_sub_pnts.T).T

        src_cntct_pnt = self.src_tool.get_normalized_pc()[self.src_tool.contact_pnt_idx, :]
        sub_contact_pnt_idx = self._get_closest_pnt(src_cntct_pnt,
                                                    rot_sub_pnts)

        sub_contact_pnt = self.sub_tool.get_pnt(sub_contact_pnt_idx)

        if self.visualize:
            visualize_two_pcs(self.sub_tool.pnts, rot_sub_pnts)
            visualize(self.sub_tool.pnts, sub_contact_pnt, self.sub_tool.segments)

        return R, sub_contact_pnt

    def calc_sub_tool_aruco(self, R):
        aruco = ArucoStuff(self.sub_tool)
        # First, Get the initial pose of the sub tool
        T_aruco_sub = aruco.get_aruco_intial_T()
        # Get the desired use pose of src tool
        T_aruco_src = aruco.get_src_tool_T()
        # Get pose of actual tool via perception.
        T_world_aruco_sub = aruco.percieve_aruco_T()

        Ps_pnts_sub = self.sub_tool.get_unnormalized_pc()
        # Calculate location of points in world frame by aligning model with percpetion.
        Ps_pnts_world = get_pnts_world_frame(T_world_aruco_sub,
                                             T_aruco_sub,
                                             Ps_pnts_sub)


        T_world_aruco_sub_rot = get_aruco_world_frame(T_aruco_sub,
                                                      Ps_pnts_sub,
                                                      Ps_pnts_world,
                                                      R)
    def main(self):
        # TODO: Make sure contact point can be transformed properly and recovered
        # self._align_action_parts()
        # cntct_pnt, R  = self._calc_sub_contact_pnt()
        self.get_contact_pnt()
        # self.get_random_contact_pnt()
        # c_point, R = self._find_best_segment()



if __name__ == '__main__':
    gp = GeneratePointcloud()
    n = 4000
    get_color = True

    pnts1 = gp.get_random_ply(n, get_color)
    pnts2 = gp.get_random_ply(n, get_color)
    # pnts1 = gp.mesh_to_pointcloud('ranch/3/ranch_out_6_20_fused.ply',n,  get_color)
    # pnts2 = gp.mesh_to_pointcloud('hammer/2/hammer_out_3_10_fused.ply',n , get_color)
    # pnts1 = gp.mesh_to_pointcloud('screwdriver_right/3/screwdriver_right_out_7_40_fused.ply',n,  get_color)
    # pnts1 = gp.mesh_to_pointcloud('clamp_left/2/clamp_left_out_2_10_fused.ply',n,  get_color)
    # pnts2 = gp.mesh_to_pointcloud('screwdriver_right/2/screwdriver_right_out_2_20_fused.ply',n , get_color)
    # pnts2 = gp.mesh_to_pointcloud('clamp_right/2/clamp_right_out_3_10_fused.ply', n, get_color)

    "./tool_files/data_demo_segmented_numbered/screwdriver_right/2/screwdriver_right_out_2_20_fused.ply"
    src = ToolPointCloud(pnts1, contact_pnt_idx=None)

    # src.visualize_bb()
    sub = ToolPointCloud(pnts2)
    # src = sub
    # sub = shrink_pc(sub)

    cntc_pnt = src.get_pc_bb_axis_frame_centered().argmax(axis=0)[0]
    # cntc_pnt = np.random.randint(0, src.pnts.shape[0], size = 1).item()
    src.contact_pnt_idx = cntc_pnt
    # src.contact_pnt_idx = cntc_pnt

    hammer_pnt = sub.pnts.argmax(axis=0)[0]
    print("SRC TOOL")
    # src.visualize()
    visualize(src.pnts, src.get_pnt(cntc_pnt))
    print("SUB TOOL")
    # sub.visualize_bb()
    visualize(sub.pnts, sub.get_pnt(hammer_pnt), segments=sub.segments)

    ts = ToolSubstitution(src, sub, visualize=True)
    ts.main()
