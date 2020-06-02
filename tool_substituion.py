import numpy as np
from numpy import cross
from numpy.linalg import norm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
from matplotlib.patches import Rectangle

from tool_pointcloud import ToolPointCloud
from sample_pointcloud import GeneratePointcloud
from util import min_point_distance, r_y, rotation_matrix_from_vectors, visualize_two_pcs, rotation_matrix



def visualize_candidate_sides(pnts, pc_bb, cand_sides, centroids, proj):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    bb = pc_bb.bb
    faces = {'a': bb[[0,3,2,1], :],
             'b': bb[[2,7,3,8], :],
             'c': bb[[5,0,3,8], :],
             'd': bb[[1,6,5,0], :],
             'e': bb[[5,8,7,6], :],
             'f': bb[[1,6,2,7], :]}

    for face in faces.keys():
        c = 'g' if face in cand_sides else 'r'
        # c = 'y' if face == 'a' else c
        side = art3d.Poly3DCollection([faces[face]])
        side.set_color(c)
        side.set_alpha(0.4) # opacity
        ax.add_collection3d(side)
    # Tool pointcloud
    ax.scatter(xs=pnts[:,0], ys=pnts[:, 1], zs=pnts[:, 2], c='b')

    # Centroids
    for c in centroids:
        ax.scatter(xs=c[0], ys=c[1], zs=c[ 2], c='y', s=150)


    for p in proj:
        ax.scatter(xs=p[0], ys=p[1], zs=p[ 2], c='black', s=150)


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



def visualize(pnts, cp, segments=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if segments is None:
        c = 'b'
    else:
        c = segments

    ax.axis('equal')
    ax.scatter(xs=pnts[:, 0], ys=pnts[:, 1], zs=pnts[:, 2], c=c)
    ax.scatter(xs=cp[0], ys=cp[1], zs=cp[2], c='r', s=200)
    plt.show()




def project_pnt_on_bb_sides(pnt, bb):
    bb_sides = bb_to_planes(bb)
    scores = []
    projs = []

    for k, v in bb_sides.items():
        p, _, c, n = v
        q = proj_on_plane(pnt, (p,n))
        projs.append(q)
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

    print(scores)
    side = min(scores, key=lambda t: t[1])

    return side[0], bb_sides, centroids


def proj_on_plane(q, plane):
    """
    Orthogonal projection of pnt q on plane
    """
    p, norm = plane

    return q  - np.dot(q - p, norm ) * norm



def bb_to_planes(bb):
    """
    Given a bb (10x3 ndarray) caluclate planes for each face of the box.
    Returns dict where keys are [a-f; str] faces of box and vals are tuples.
    Tuples contain: a pnt on the face (ndarray) the relative size of face (int, 0-2),
                    the centroid (ndarray), and the norm of the plane (ndarray).
    """
    bb_pnts = bb.bb
    bb_areas = [norm(bb.norms[0]) * norm(bb.norms[1]),
                norm(bb.norms[0]) * norm(bb.norms[2]),
                norm(bb.norms[1]) * norm(bb.norms[2])]


    faces = {'a': bb_pnts[[0,3,2,1], :],
             'b': bb_pnts[[2,7,3,8], :],
             'c': bb_pnts[[5,0,3,8], :],
             'd': bb_pnts[[1,6,5,0], :],
             'e': bb_pnts[[5,8,7,6], :],
             'f': bb_pnts[[1,6,2,7], :]}

    # bb side areas from lowest to highest
    bb_areas  = sorted(bb_areas)
    print("BB AREAS: {}".format(bb_areas))

    n1 = cross(bb_pnts[2, :] - bb_pnts[3,:], bb_pnts[0, :] - bb_pnts[3, :])
    n1 = n1 / norm(n1)
    a1 = norm(bb_pnts[2, :] - bb_pnts[3,:]) * norm(bb_pnts[0, :] - bb_pnts[3, :])
    a1 = np.argmin(np.abs(bb_areas - a1))
    c1 = faces['a'].mean(axis=0)
    s1 = (bb_pnts[3,:], a1, c1, n1)

    n2 = cross(bb_pnts[7, :] - bb_pnts[8,:], bb_pnts[3, :] - bb_pnts[8, :])
    n2 = n2 / norm(n2)
    a2 = norm(bb_pnts[7, :] - bb_pnts[8,:]) * norm( bb_pnts[3, :] - bb_pnts[8, :])
    a2 = np.argmin(np.abs(bb_areas - a2))
    c2 = faces['b'].mean(axis=0)
    s2 = (bb_pnts[8,:], a2, c2, n2)

    n3 = cross(bb_pnts[0, :] - bb_pnts[3,:], bb_pnts[8, :] - bb_pnts[3, :])
    n3 = n3 / norm(n3)
    a3 = norm(bb_pnts[0, :] - bb_pnts[3,:]) * norm(bb_pnts[8, :] - bb_pnts[3, :])
    a3 = np.argmin(np.abs(bb_areas - a3))
    c3 = faces['c'].mean(axis=0)
    s3 = (bb_pnts[3,:], a3, c3, n3)


    n4 = cross(bb_pnts[5, :] - bb_pnts[0,:], bb_pnts[1, :] - bb_pnts[0, :])
    n4 = n4 / norm(n4)
    a4 = norm(bb_pnts[5, :] - bb_pnts[0,:]) *  norm(bb_pnts[1, :] - bb_pnts[0, :])
    a4 = np.argmin(np.abs(bb_areas - a4))
    c4 = faces['d'].mean(axis=0)
    s4 = (bb_pnts[0,:], a4, c4, n4)


    n5 = cross(bb_pnts[6, :] - bb_pnts[5,:], bb_pnts[8, :] - bb_pnts[5, :])
    n5 = n5 / norm(n5)
    a5 = norm(bb_pnts[6, :] - bb_pnts[5,:]) * norm( bb_pnts[8, :] - bb_pnts[5, :])
    a5 = np.argmin(np.abs(bb_areas - a5))
    c5 = faces['e'].mean(axis=0)
    s5 = (bb_pnts[5,:], a5, c5,  n5)

    n6 = cross(bb_pnts[7, :] - bb_pnts[2,:], bb_pnts[1, :] - bb_pnts[2, :])
    n6 = n6 / norm(n6)
    a6 = norm(bb_pnts[7, :] - bb_pnts[2,:]) * norm(bb_pnts[1, :] - bb_pnts[2, :])
    a6 = np.argmin(np.abs(bb_areas - a6))
    c6 = faces['f'].mean(axis=0)
    s6 = (bb_pnts[2,:], a6, c6, n6)

    return {'a':s1, 'b':s2, 'c':s3, 'd':s4, 'e':s5, 'f':s6}


class ToolSubstitution(object):
    def __init__(self, src_tool_pc, sub_tool_pc ):
        "docstring"
        self.src_tool = src_tool_pc
        self.sub_tool = sub_tool_pc

    def _calc_best_orientation(self, src_pnts, sub_pnts, Rs):
        """
        Given a list of rotation matrices, R, determine f
        """
        if not Rs:
            Rs.append(np.identity(3))

        scores = []
        aligned_pnts = []
        for R in Rs:
            sub_pnts_rot = R.dot(sub_pnts.T)
            # visualize_two_pcs(src_pnts, sub_pnts_rot.T)

            # Average scores due to slight asymmetry in distance metric.
            score1 = min_point_distance(src_pnts.T, sub_pnts_rot)
            score2 = min_point_distance(sub_pnts_rot, src_pnts.T)

            score = (score1 + score2) / 2.0
            scores.append(score)
            aligned_pnts.append(sub_pnts_rot.T)

        i = np.argmin(scores)

        print "rotation {} is best".format(i + 1)
        return Rs[i], aligned_pnts[i]


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


        scaled_sub_pnts = sub_pc.scale_pnts_to_target(src_pc)
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

        return self._calc_best_orientation(src_pc.pnts,
                                           scaled_sub_pc.pnts,
                                           [R1, R2])



    def _get_sub_tool_action_part(self):
        # Center pointclouds
        src_pnts = self.src_tool.get_pc_bb_axis_frame_centered()
        sub_pnts = self.sub_tool.get_pc_bb_axis_frame_centered()

        # Add the segment labels back in.
        src_pnts = np.vstack([src_pnts.T, self.src_tool.segments]).T
        sub_pnts = np.vstack([sub_pnts.T, self.sub_tool.segments]).T

        self.centered_src_pc = ToolPointCloud(src_pnts)
        self.centered_sub_pc = ToolPointCloud(sub_pnts)

        # Find the best alignment of the sub tool based on 3d haming distance.
        _, aligned_sub_pnts = self._align_pnts(self.centered_src_pc, self.centered_sub_pc)



        visualize_two_pcs(src_pnts, aligned_sub_pnts)
        src_contact_pnt = self.centered_src_pc.get_pnt(self.src_tool.contact_pnt_idx)
        sub_action_pnt = self._get_closest_pnt(src_contact_pnt, aligned_sub_pnts)

        return self.sub_tool.get_segment_from_point(sub_action_pnt), aligned_sub_pnts

    def _calc_sub_contact_pnt(self):
        """
        Returns estimated contact point and rotation matrix for sub tool based on src tool.

        """
        # Get segment of contact point of src tool.
        src_action_seg = self.src_tool.get_segment_from_point(self.src_tool.contact_pnt_idx)
        # Determine the corresponding segment based on shape in sub tool.
        sub_action_seg, scaled_sub_pnts = self._get_sub_tool_action_part()

        scaled_sub_pnts = np.vstack([scaled_sub_pnts.T, self.sub_tool.segments]).T
        scaled_sub_pc = ToolPointCloud(scaled_sub_pnts, normalize=False)

        print("src action seg: {}".format(src_action_seg))
        print("sub action seg: {}".format(sub_action_seg))

        # Segments of tools that aren't the action segments
        src_other_segs = [s for s in self.centered_src_pc.segment_list \
                          if not s == src_action_seg]
        sub_other_segs = [s for s in scaled_sub_pc.segment_list \
                          if not s == sub_action_seg]

        print("src other segs: {}".format(src_other_segs))
        print("sub other segs: {}".format(sub_other_segs))

        # Get centroids of bb sides in order to help determine which sides
        # Should be considered as contact surfaces.
        src_seg_centroids = calc_seg_centroids(src_other_segs, self.centered_src_pc)
        sub_seg_centroids = calc_seg_centroids(sub_other_segs, scaled_sub_pc)

        # Get pointclouds corresponding to action parts of each tool.
        src_action_pnts = self.centered_src_pc.get_pnts_in_segment(src_action_seg)
        src_action_pc = ToolPointCloud(src_action_pnts, normalize=False)

        sub_action_pnts = scaled_sub_pc.get_pnts_in_segment(sub_action_seg)
        sub_action_pc = ToolPointCloud(sub_action_pnts, normalize=False)

        # Scale sub action part to src action part size for better comparison.
        scaled_sub_action_pnts = sub_action_pc.scale_pnts_to_target(src_action_pc)
        scaled_sub_action_pc = ToolPointCloud(scaled_sub_action_pnts, normalize=False)

        sub_action_bb = sub_action_pc.bb
        src_action_bb = src_action_pc.bb

        src_cntct_pnt = self.centered_src_pc.get_pnt(self.src_tool.contact_pnt_idx)

        src_adj_sides = [] # stores sides of action part bb that conenct to other segments

        for c in src_seg_centroids:
            s,src_all_bb_sides, projs = get_closest_bb_side_to_pnt(c ,src_action_bb)
            src_adj_sides.append(s)

        src_c_side, src_all_sides, _= get_closest_bb_side_to_pnt(src_cntct_pnt,
                                                                  src_action_bb)

        # If the contacting side is on a side that connects to another segment,
        # Then the tool is likely being used for pulling, and similar sides
        # ought to be considered on the sub tool.
        cntct_on_adj_side = src_c_side in src_adj_sides

        sub_adj_sides = []
        sub_all_sides = {}
        projs = []
        # Determine sides of bounding box adjacent to other segments
        for c in sub_seg_centroids:
            s, sub_all_sides, projs = get_closest_bb_side_to_pnt(c ,sub_action_bb)
            sub_adj_sides.append(s)

        # Remove duplicates
        sub_adj_sides = list(set(sub_adj_sides))
        # Remove sides adjacent to other segments.
        print "Sub all sides ", sub_all_sides

        # If the contact point on src tool is on adj side, then we consider those
        # sides on sub tool.
        if cntct_on_adj_side and sub_adj_sides:
            # cand_sides = [s for s in sub_all_sides.keys() if s in sub_adj_sides]
            cand_sides = sub_adj_sides
        # Otherwise we consider non-adj sides11
        else:
            cand_sides = [s for s in sub_all_sides.keys() if not s in sub_adj_sides]
            # cand_sides = list(sub_adj_sides.keys())

        print("Sub connecting sides: {}".format(sub_adj_sides))
        print("Sub candidate sides: {}".format(cand_sides))

        sub_action_pc.visualize_bb(),
        visualize_candidate_sides(scaled_sub_pc.pnts,
                                  sub_action_bb,
                                  cand_sides,
                                  sub_seg_centroids,
                                  projs)

        # print("sub all sides: {}".format(sub_all_sides))

        print("Contacting side: {}".format(src_c_side))
        _, rel_size, _, n = src_all_sides[src_c_side]

        sub_same_size_sides = [k for k in cand_sides if sub_all_sides[k][1] == rel_size]
        print("Rel size of src contact surface: ", rel_size)
        print("Sub sides of same rel size: ", sub_same_size_sides)
        
        visualize_candidate_sides(scaled_sub_pc.pnts,
                                  sub_action_bb,
                                  sub_same_size_sides,
                                  sub_seg_centroids,
                                  projs)



        src_c_side_dir = n # Norm of side containing src contact pnt
        Rs = [] # Stores all candidate rotation martices
        # First test sides of sub tool that are same relative size
        # to contact side of src tool.
        if sub_same_size_sides:
            for s in sub_same_size_sides:
                sub_dir = sub_all_sides[s][3]
                print "Parallel score : ",  np.dot(n, sub_dir.T)

                # If norms are already parallel, then rotating
                # them gives wrong answer for some reason.
                norms_are_parallel = np.isclose(np.abs(np.dot(src_c_side_dir, sub_dir.T)),
                                                np.ones(1), atol=0.05)

                print "Are parallel? ", norms_are_parallel
                if not norms_are_parallel:
                    print "NORMALS ARE NOT PARALLEL"
                    R = rotation_matrix_from_vectors(sub_dir, src_c_side_dir)
                else:
                    print "NORMALS ARE PARALLEL"
                    R = np.identity(3)

                Rs.append(R)

        # If no  available sides with same relative sides,
        # just try all candidate sides.
        elif cand_sides:
            for s in cand_sides:
                sub_dir = sub_all_sides[s][3]
                R = rotation_matrix_from_vectors(sub_dir, src_c_side_dir)
                Rs.append(R)


        _, aligned_sub_pnts= self._calc_best_orientation(src_action_pc.get_normalized_pc(),
                                                         scaled_sub_action_pc.get_normalized_pc(),
                                                         Rs)
        print "BEST ORIENTATION"
        visualize_two_pcs(src_action_pc.get_normalized_pc(),
                          scaled_sub_action_pc.get_normalized_pc(),
                          )


        # Determine the contact point of src tool in action part segment point cloud
        # and find corresponding point on the sub action part segment pointcloud
        src_cntct_idx = self.centered_src_pc.idx_to_segment_idx(self.src_tool.contact_pnt_idx)
        src_action_contact_pnt = src_action_pc.get_pnt(src_cntct_idx)

        sub_contact_pnt_idx = self._get_closest_pnt(src_action_contact_pnt,
                                                    aligned_sub_pnts)

        # Get pnts corresponding to these segments
        sub_contact_pnt_idx = scaled_sub_pc.segment_idx_to_idx(sub_action_seg,
                                                               sub_contact_pnt_idx)
        sub_cntct_pnt = scaled_sub_pc.get_pnt(sub_contact_pnt_idx)
        print("SRC CONTACT PNT {}".format(src_cntct_pnt))
        visualize(self.centered_src_pc.pnts, src_cntct_pnt, self.centered_src_pc.segments)
        print("sub CONTACT PNT {}".format(sub_cntct_pnt))
        visualize(scaled_sub_pc.pnts, sub_cntct_pnt, self.centered_sub_pc.segments)

        return sub_cntct_pnt



    def _align_action_parts(self):

        # Get segment of contact point of src tool.
        src_action_seg = self.src_tool.get_segment_from_point(self.src_tool.contact_pnt_idx)
        # Determine the corresponding segment based on shape in sub tool.
        sub_action_seg = self._get_sub_tool_action_part()

        # Get pnts corresponding to these segments
        src_action_pnts = self.centered_src_pc.get_pnts_in_segment(src_action_seg)
        sub_action_pnts = self.centered_sub_pc.get_pnts_in_segment(sub_action_seg)


        src_action_pc = ToolPointCloud(src_action_pnts)
        sub_action_pc = ToolPointCloud(sub_action_pnts)


        # align action segment of sub tool to src tool
        _, aligned_sub_pnts = self._align_pnts(src_action_pc, sub_action_pc)

        # # src_action_contact_pnt = src_action_pc.get_pnt(self.src_tool.contact_pnt_idx)


        # Determine the contact point of src tool in action part segment point cloud
        # and find corresponding point on the sub action part segment pointcloud
        src_cntct_idx = self.centered_src_pc.idx_to_segment_idx(self.src_tool.contact_pnt_idx)
        src_action_contact_pnt = src_action_pc.get_pnt(src_cntct_idx)
        # src_cntct_pnt = self.centered_src_pc.get_pnt(self.src_tool.contact_pnt_idx)
        # src_action_contact_pnt = src_action_pc.transform(src_cntct_pnt)
        sub_contact_pnt_idx = self._get_closest_pnt(src_action_contact_pnt,
                                                    aligned_sub_pnts)

        # Get calculated sub tool contact point on original pointcloud.
        sub_contact_pnt_idx = self.centered_sub_pc.segment_idx_to_idx(sub_action_seg,
                                                                      sub_contact_pnt_idx)
        sub_cntct_pnt = self.centered_sub_pc.get_pnt(sub_contact_pnt_idx)
        src_cntct_pnt = self.centered_src_pc.get_pnt(self.src_tool.contact_pnt_idx)

        print("SRC CONTACT PNT {}".format(src_cntct_pnt))
        visualize(self.centered_src_pc.pnts, src_cntct_pnt, self.centered_src_pc.segments)
        print("sub CONTACT PNT {}".format(sub_cntct_pnt))
        visualize(self.centered_sub_pc.pnts, sub_cntct_pnt, self.centered_sub_pc.segments)
        # visualize(self.centered_sub_pc, sub_cntct_pnt)

        visualize(src_action_pc.pnts, src_cntct_pnt, src_action_pc.segments)
        print("sub CONTACT PNT {}".format(sub_cntct_pnt))
        sub_action_pc = ToolPointCloud(aligned_sub_pnts)
        visualize(sub_action_pc.pnts, sub_cntct_pnt, sub_action_pc.segments)

        # R =
        return sub_cntct_pnt

    def main(self):
        # TODO: Make sure contact point can be transformed properly and recovered
        # self._align_action_parts()
        self._calc_sub_contact_pnt()



if __name__ == '__main__':
    gp = GeneratePointcloud()
    n = 3000
    get_color = True

    # pnts1 = gp.get_random_ply(n, get_color)
    pnts2 = gp.get_random_ply(n, get_color)
    pnts1 = gp.ply_to_pointcloud(n, 'screwdriver_right/3/screwdriver_right_out_6_60_fused.ply', get_color)
    # pnts2 = gp.ply_to_pointcloud(n, 'clamp_left/2/clamp_left_out_8_50_fused.ply',get_color)

    src = ToolPointCloud(pnts1, contact_pnt_idx=None)

    # src.visualize_bb()
    sub = ToolPointCloud(pnts2)

    cntc_pnt = src.get_pc_bb_axis_frame_centered().argmax(axis=0)[0]
    src.contact_pnt_idx = cntc_pnt

    print("SRC TOOL")
    # src.visualize()
    visualize(src.pnts, src.get_pnt(cntc_pnt))
    print("SUB TOOL")
    sub.visualize()


    # bb = src.bb.bb
    # # pnt = bb[0, :] +
    # proj, _ = get_closest_bb_side_to_pnt(pnt, bb)
    # visualize_bb_pnt_proj(bb, pnt, proj)

    ts = ToolSubstitution(src, sub)
    ts.main()
