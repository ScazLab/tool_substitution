#!/usr/bin/env python
import numpy as np
from copy import deepcopy
from numpy import cross
from numpy.linalg import norm

from itertools import permutations

from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree, KDTree

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
from matplotlib.patches import Rectangle

import open3d as o3d

from tool_pointcloud import ToolPointCloud
from sample_pointcloud import GeneratePointcloud
from util import (min_point_distance, rotation_matrix_from_vectors,
                  weighted_min_point_distance, visualize_two_pcs,
                  rotation_matrix_from_box_rots, visualize_vectors,
                  r_x, r_y, r_z, visualize_contact_area)

from scipy.spatial.transform import Rotation as Rot

from get_target_tool_pose import get_T_from_R_p, T_inv, get_scaling_T
from pointcloud_registration import prepare_dataset, draw_registration_result


def gen_contact_surface(pc, pnt_idx):
    """
    Generare a contact surface on a pointcloud around a desired point.
    
    """
    tree = cKDTree(pc)
    i =  tree.query_ball_point(pc[pnt_idx,:], .01)

    return i

def visualize(pnts, cp=None, segments=None):
    """
    Util function for visualizing tool point cloud with contact point and segments.
    """
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


class ToolSubstitution(object):
    """
    Class for aligning a substitute tool pointcloud to a source_action tool pointcloud.
    """
    def __init__(self, src_tool_pc, sub_tool_pc, voxel_size=0.02, visualize=False):
        """
        Class for aligning substitute tool to source_action tool based on a given contact surface
        of the source_action tool.
        """
        # ToolPointClouds for src and sub tools
        self.src_tool = src_tool_pc
        self.sub_tool = sub_tool_pc

        # Open3d pointcloud of src and sub tool.
        self.src_pcd = self._tpc_to_o3d(self.src_tool)
        self.sub_pcd = self._tpc_to_o3d(self.sub_tool)
        # Same as above but we will apply all transformations to these
        self.T_src_pcd = deepcopy(self.src_pcd)
        self.T_sub_pcd = deepcopy(self.sub_pcd)

        # Params for ICP
        self.voxel_size = voxel_size
        self.correspondence_thresh = voxel_size * .5
        # self.correspondence_thresh = voxel_size * .1
        # Acceptable amount of alignment loss after applying ICP.
        # Often the quantitatively alignment will drop, but qualitatively it gets better.
        self.fit_ratio_thresh = .75
        # See https://en.wikipedia.org/wiki/Mahalanobis_distance
        self.mahalanobis_thresh = 1.

        self.visualize = visualize
        self.Ts =[] # Tracks all transformations of sub tool.
        self.scale_Ts = [] # Tracks all scalings of sub tool.


    def _icp_wrapper(self, src,sub, src_fpfh, sub_fpfh, correspondence_thresh, n_iter=5):
        """
        Wrapper function for configuring and running icp registration using feature mapping.
        """
        checker = [
            o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(correspondence_thresh)
            # o3d.registration.CorrespondenceCheckerBasedOnNormal(np.pi/2)
                   ]


        RANSAC = o3d.registration.registration_ransac_based_on_feature_matching

        est_ptp = o3d.registration.TransformationEstimationPointToPoint()
        est_ptpln = o3d.registration.TransformationEstimationPointToPlane()

        criteria = o3d.registration.RANSACConvergenceCriteria(max_iteration=400000,
                                                              max_validation=1000)

        results = []

        # Apply icp n_iter times and get best result
        for i in range(n_iter):
            result1 = RANSAC(sub, src, sub_fpfh, src_fpfh,
                            max_correspondence_distance=correspondence_thresh,
                            estimation_method=est_ptp,
                            ransac_n=4,
                            checkers=checker,
                            criteria=criteria)

            result2 = o3d.registration.registration_icp(sub, src,
                                                        self.voxel_size,
                                                        result1.transformation,
                                                        est_ptpln)
            results.append(result2)
            print "ICP FITNESS: ", result2.fitness 
            if result2.fitness == 1.0: break # 1.0 is best possible score.

        return max(results, key=lambda i:i.fitness)

    
    def _get_sub_pnts(self, get_segments=True):
        """
        Get ndarray of points from o3d pointcloud.
        """
        pnts = np.asarray(self.T_sub_pcd.points)
        if get_segments:
            pnts = np.vstack([pnts.T, self.sub_tool.segments]).T

        return pnts


    def _get_src_pnts(self, get_segments=True):
        """
        Get ndarray of points from o3d pointcloud.
        """
        pnts = np.asarray(self.T_src_pcd.points)
        if get_segments:
            pnts =  np.vstack([pnts.T, self.src_tool.segments]).T

        return pnts

    @staticmethod
    def _tpc_to_o3d(tpc):
        """
        Get o3d pc from ToolPointcloud object.
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(tpc.pnts)

        return pcd

    def _calc_center_and_align_T(self, pc):
        """
        Creates a centered and aligned ToolPointCloud from unaligned ToolPointCloud
        """
        T_align  = get_T_from_R_p(R=pc.get_axis())
        T_center = get_T_from_R_p(p=pc.get_bb_centroid())

        return np.matmul(T_align, T_center)

    def _calc_best_orientation(self, sub_pcd, Rs):
        """
        @src_pnts: (nx3) ndarray of points of src tool.
        @sub_pnts: (mx3) ndarray of points of sub tool.
        @Rs:       list of (3x3) ndarray rotations.
        Returns (R, (nx3) array of rotated points, score) representing
        The rotation of sub_pnts that best aligns with src_pnts.
        """
        if not Rs:
            Rs.append(np.identity(3))

        scores = []
        aligned_pnts = []
        for R in Rs:
            T = get_T_from_R_p(R=R)
            rot_sub_pcd = deepcopy(sub_pcd)
            rot_sub_pcd = sub_pcd.transform(T)
            dist = o3d.registration.evaluate_registration(rot_sub_pcd,
                                                          self.T_src_pcd,
                                                          self.correspondence_thresh)
            # o3d.visualization.draw_geometries([rot_sub_pcd, self.Tsrc_pcd], "Aligned")
            score = dist.fitness

            print "ALIGNMENT SCORE: ", score

            scores.append(score)

        i = np.argmax(scores) # Higher the score the better.

        print "rotation {} is best".format(i + 1)
        return get_T_from_R_p(R=Rs[i]), scores[i]

    def _get_contact_surface(self, src_cps, sub_pnts, src_pnts):
        """
        @src_cps: (nk3) ndarray contact surface of src tool (subset of src_pnts).
        @sub_pnts: (mx3) ndarray points in sub tool pc.
        @src_pnts: (nx3) ndarray points of src tool.

        Returns idx of pnts in sub_pnts estimated to be its contact surface.
        """

        if len(src_cps.shape) > 1: # If there are multiple contact points
            cov = np.cov(src_cps.T) # Then get the mean and cov from these points
            cp_mean = src_cps.mean(axis=0)
        else: # If only one contact point...
            # Then get more by finding 20 nearest neighbors around it.
            tree = KDTree(src_pnts)
            _, i = tree.query(src_cps, k=20)
            est_src_surface = src_pnts[i, :]

            cov = np.cov(est_src_surface.T)
            cp_mean = src_cps


        est_sub_cp, _ = self._get_closest_pnt(cp_mean, sub_pnts)
        # Get points arounnd est_sub_cp with similar distribution as src_cps.
        mdist = cdist(sub_pnts, [est_sub_cp], metric='mahalanobis', V=cov)[:,0]


        return mdist < self.mahalanobis_thresh



    def _get_closest_pnt(self, pnt, pntcloud):
        """
        returns the point in pntcloud closest to pnt.
        """
        tree = cKDTree(pntcloud)
        _, i = tree.query(pnt)

        return pntcloud[i,:], i

    def _align_pnts(self):
        """
        Scale sub_pc to and then detemine most similar orientation to src_pc
        Returns ndarray of sub_pc pnts in best orientation.
        """

        # Test current orientation
        R1 = np.identity(3)
        R2 = r_x(np.pi)
        R3 = r_y(np.pi)
        R4 = r_z(np.pi)

        scores = []
        # Ger perms of all indx
        # _, scale_f = self.sub_tool.scale_pnts_to_target(self.scaled_src_pc)
        src_tool_norms = ToolPointCloud(self._get_src_pnts()).bb.norms
        sub_tool_norms = ToolPointCloud(self._get_sub_pnts()).bb.norms


        for p in permutations([0,1,2]):
            scaled_sub_pcd = deepcopy(self.T_src_pcd)
            # permed_scale_f = scale_f[list(p)]
            permed_scale_f = (src_tool_norms / sub_tool_norms)[list(p)]
            T_sub_action_part_scale = get_T_from_R_p(R=np.identity(3)*permed_scale_f)
            # scaled_sub_pcd = self.T_src_pcd.transform(T_sub_action_part_scale)
            scaled_sub_pcd.transform(T_sub_action_part_scale)


            T_rot, score = self._calc_best_orientation(scaled_sub_pcd,
                                                       [R1, R2, R3, R4])

            scores.append((T_rot, T_sub_action_part_scale, score))

        T_rot, T_sub_action_part_scale, fit =  max(scores, key=lambda s: s[2])
        

        self.T_sub_pcd.transform(T_sub_action_part_scale)
        self.T_sub_pcd.transform(T_rot)


        self.Ts.append(T_rot)
        self.scale_Ts.append(T_sub_action_part_scale) # Append scaling matrix

        # o3d.visualization.draw_geometries([self.T_src_pcd, self.T_sub_pcd], "Aligned")
        return fit


    def get_random_contact_pnt(self):
        """
        Get a random contact point and rotation matrix for substitute tool.
        """

        # src_action_seg = self.src_tool.get_segment_from_point(self.src_tool.contact_pnt_idx)
        scaled_sub_pnts, _ = self.sub_tool.scale_pnts_to_target(self.src_tool)
        R = Rot.random(num=1).as_dcm() # Generate a radom rotation matrix.
        T = get_T_from_R_p(p=np.zeros((1,3)), R=R )
        rot_sub_pnts = R.dot(scaled_sub_pnts.T).T

        src_cntct_pnt = self.src_tool.get_normalized_pc()[self.src_tool.contact_pnt_idx, :]
        _, sub_contact_pnt_idx = self._get_closest_pnt(src_cntct_pnt,
                                                    rot_sub_pnts)

        sub_contact_pnt = self.sub_tool.get_pnt(sub_contact_pnt_idx)

        T = get_T_from_R_p(p=np.zeros((1,3)), R=R )
        if self.visualize:
            visualize_two_pcs(self.sub_tool.pnts, rot_sub_pnts)
            visualize(self.sub_tool.pnts, sub_contact_pnt, self.sub_tool.segments)

        return T, sub_contact_pnt



    def _scale_pcs(self):
        """
        Ensures that all pcs are of a consistent scale (~1m ) so that ICP default params will work.
        """

        T_src = self._calc_center_and_align_T(self.src_tool)
        T_sub = self._calc_center_and_align_T(self.sub_tool)

        self.T_src_pcd.transform(T_src)
        self.T_sub_pcd.transform(T_sub)
        self.Ts.append(T_sub) # To account for alignment of sub tool along bb axis.
        self.Ts.append(T_inv(T_src)) # To account for alignment of src tool along bb axis.

        # Get the lengths of the bounding box sides for each pc
        src_norm = norm(self.T_src_pcd.get_max_bound() - self.T_src_pcd.get_min_bound())
        sub_norm = norm(self.T_sub_pcd.get_max_bound() - self.T_sub_pcd.get_min_bound())

        # Scale by 1 / longest_side to ensure longest side is no longer than 1m.
        largest_span = src_norm if src_norm > sub_norm else sub_norm
        T_sub_action_part_scale = get_T_from_R_p(R=np.identity(3)/largest_span)
        

        self.T_src_pcd.transform(T_sub_action_part_scale)
        self.T_sub_pcd.transform(T_sub_action_part_scale)

        self.scale_Ts.append(T_sub_action_part_scale) # Store scaling factor

        # Sub tool AND src tool have been transformed, so make sure to account for both
        # for final sub rotation.



    def get_tool_action_parts(self):
        """
        Get the idx associated with the action parts of the src and sub tools.
        """

        self._src_action_segment = self.src_tool.get_action_segment()
        print "SRC ACTION SEGMENT: ", self._src_action_segment

        scaled_src_pc = ToolPointCloud(self._get_src_pnts())
        scaled_sub_pc = ToolPointCloud(self._get_sub_pnts())

        print "SCALED SUB SEGMENTS: ", scaled_sub_pc.segment_list
        # Get points in segment of src tool containing the contact area
        src_action_part = scaled_src_pc.get_pnts_in_segment(self._src_action_segment )
        # First, scale both tools to the same size, in order to determine which part of
        # sub tool is the action part.
        _, scale_f = scaled_sub_pc.scale_pnts_to_target(scaled_src_pc)
        T_sub_scale = get_T_from_R_p(R=np.identity(3)*scale_f)
        self.T_sub_pcd.transform(T_sub_scale)
        self.scale_Ts.append(T_sub_scale)
        # o3d.visualization.draw_geometries([self.T_sub_pcd, self.T_src_pcd], "Aligned")
        scaled_sub_pc = ToolPointCloud(self._get_sub_pnts())
        _, sub_action_pnt_idx = self._get_closest_pnt(src_action_part.mean(axis=0),
                                                      scaled_sub_pc.pnts)


        self._sub_action_segment = scaled_sub_pc.get_segment_from_point(sub_action_pnt_idx)

        sub_action_part = scaled_sub_pc.get_pnts_in_segment(self._sub_action_segment)

        # Scale action parts to same size for better comparison.
        print "SCALED SUB SEGMENTS: ", scaled_sub_pc.segment_list
        _, scale_f = ToolPointCloud(sub_action_part).scale_pnts_to_target(ToolPointCloud(src_action_part))
        T_sub_action_part_scale = get_T_from_R_p(R=np.identity(3)*scale_f)

        self.T_sub_pcd.transform(T_sub_action_part_scale)
        self.scale_Ts.append(T_sub_action_part_scale)
        # sub_action_part = self._get_sub_pnts()

        # o3d.visualization.draw_geometries([self.T_src_pcd, self.T_sub_pcd], "Aligned")
        return src_action_part, sub_action_part


    def icp_alignment(self,src_pnts,sub_pnts, correspondence_thresh, n_iter=10):
        """
        Algin sub tool to src tool using ICP.

         """
        # Scale points to be ~1 meter so that self.voxel_size can
        # be consistent for all pointclouds.

        # o3d.visualization.draw_geometries([self.T_src_pcd, self.T_sub_pcd], "Aligned")
        # src_action_pnts = self.scaled_src_pc.pnt

        source_action, substitute, source_down, substitute_down, source_fpfh, substitute_fpfh = \
            prepare_dataset(src_pnts, sub_pnts, self.voxel_size)

        # Apply icp n_iter times and get best result
        best_icp = self._icp_wrapper(source_down, substitute_down,
                                     source_fpfh, substitute_fpfh,
                                     correspondence_thresh,
                                     n_iter)



        icp_fit = best_icp.fitness # Fit score (between [0.0,1.0])
        icp_trans = best_icp.transformation # T matrix

        return icp_trans, icp_fit, source_action, substitute

    def refine_registration(self, init_trans, init_fit, source, substitute, voxel_size):


        pose_graph = o3d.registration.PoseGraph()
        accum_pose = np.identity(4)
        pose_graph.nodes.append(o3d.registration.PoseGraphNode(accum_pose))

        pcds = [substitute, source]
        src_idx, sub_idx = (1, 0)
        n_pcds = len(pcds)
        for source_id in range(n_pcds):
            for target_id in range(source_id + 1, n_pcds):
                sub = pcds[source_id]
                src = pcds[target_id]

                GTG_mat = o3d.registration.get_information_matrix_from_point_clouds(sub, src,
                                                                                    voxel_size,
                                                                                    init_trans)

                if target_id == source_id + 1:
                    accum_pose = np.matmul(init_trans, accum_pose)

                pose_graph.nodes.append(o3d.registration.PoseGraphNode(np.linalg.inv(accum_pose)))

                pose_graph.edges.append(o3d.registration.PoseGraphEdge(source_id,
                                                                target_id,
                                                                init_trans,
                                                                GTG_mat,
                                                                uncertain=True))

        solver = o3d.registration.GlobalOptimizationLevenbergMarquardt()
        criteria = o3d.registration.GlobalOptimizationConvergenceCriteria()
        option = o3d.registration.GlobalOptimizationOption(max_correspondence_distance=voxel_size / 10,
                                                           edge_prune_threshold=voxel_size / 10,
                                                           reference_node=0)

        o3d.registration.global_optimization(pose_graph,
                                            method=solver,
                                            criteria=criteria,
                                            option=option)

        trans_pcds = [pcd for pcd in pcds] # deep copy
        fitnesses  = [] #Store fitnes score to determine whether to use transformations.
        # Apply calculated transformations to pcds
        for pcd_id in range(n_pcds):
            trans = pose_graph.nodes[pcd_id].pose
            trans_pcds[pcd_id].transform(trans)
            fit = o3d.registration.evaluate_registration(trans_pcds[sub_idx],
                                                         trans_pcds[src_idx],
                                                         self.correspondence_thresh).fitness
            fitnesses.append(fit)

        # If these new transformations lower fit score, dont apply them.

        return trans_pcds[src_idx], trans_pcds[sub_idx], pose_graph.nodes[sub_idx].pose
        print fitnesses
        if fitnesses[sub_idx] > init_fit:
            aligned_sub = np.asarray(trans_pcds[sub_idx].points)
            trans = pose_graph.nodes[sub_idx].pose
            self.Ts.append(trans)
            self.T_sub_pcd.transform(trans)
        else:
            print "USING INITIAL ICP RESULTS: "
            aligned_sub = np.asarray(pcds[sub_idx].points)
            trans = init_trans

        return trans_pcds[src_idx], trans_pcds[sub_idx], trans

    def get_T_cp(self, n_iter=10):
        """
        Refine Initial ICP alignment and return final T rotation matrix and sub contact pnts.
        """

        # Apply initial icp alignment

        self._scale_pcs()
        # Find best initial alignment.
        align_fit = self._align_pnts()
        # src_action_part, sub_action_part = self.get_tool_action_parts(sub_pnts)
        src_action_pnts, sub_action_pnts = self.get_tool_action_parts()

        icp_trans, fit, src_action_pcd, sub_action_pcd = self.icp_alignment(src_action_pnts,
                                                                            sub_action_pnts,
                                                                            self.correspondence_thresh,
                                                                            n_iter)

        print "ORGINAL ALIGN FITNESS: ", align_fit
        print "INIT ICP FITNESS:      ", fit
        fit_ratio = fit / align_fit

        if fit_ratio > self.fit_ratio_thresh:
            print "USING INIT. ICP RESULTS.."
            trans = icp_trans
        else:
            print "SKIPPING  INIT. ICP RESULTS, USING INIT. ALIGNMENT."
            trans = get_T_from_R_p()


        aligned_src_pcd, aligned_sub_pcd, T_align = self.refine_registration(trans,
                                                                             fit,
                                                                             src_action_pcd,
                                                                             sub_action_pcd,
                                                                             self.voxel_size)

        # o3d.visualization.draw_geometries([aligned_src_pcd, aligned_sub_pcd])
        self.T_sub_pcd.transform(T_align)

        aligned_src = np.asarray(aligned_src_pcd.points)
        aligned_sub = np.asarray(aligned_sub_pcd.points)

        # visualize_two_pcs(aligned_sub, aligned_src)
        src_action_part_cp_idx = self.src_tool.idx_to_segment_idx(self.src_tool.contact_pnt_idx)
        # src_contact_pnt = aligned_src[self.src_tool.contact_pnt_idx, :]
        src_contact_pnt = aligned_src[src_action_part_cp_idx, :]
        # src_contact_pnt = self.src_tool.pnts[src_action_part_cp_idx, :]
        # Reshape point for easy computation
        src_contact_pnt = src_contact_pnt.reshape(1,-1) \
            if len(src_contact_pnt.shape) == 1 else src_contact_pnt

        # Estimate contact surface on the sub tool
        sub_action_part_cp_idx = self._get_contact_surface(src_contact_pnt,
                                                    aligned_sub,
                                                    aligned_src)

        mean_sub_action_part_cp = np.mean(aligned_sub[sub_action_part_cp_idx, :])
        scale_T = T_inv(np.linalg.multi_dot(self.scale_Ts)) # Get inverted scaling T
        scale_f = np.diag(scale_T)[:3] # get first 3 diag entries.
        final_scale_T = get_scaling_T(scale_f, mean_sub_action_part_cp)

        visualize_contact_area(aligned_sub, sub_action_part_cp_idx)

        descaled_aligned_sub = deepcopy(self.T_sub_pcd).transform(final_scale_T)
        o3d.visualization.draw_geometries([aligned_src_pcd, descaled_aligned_sub])
        # o3d.visualization.draw_geometries([self.T_src_pcd, self.src_pcd ])

        descaled_sub_pnts = np.asarray(descaled_aligned_sub.points)
        original_sub_pnts = np.asarray(self.sub_pcd.points)

        new_vox_size = self.voxel_size 
        new_corr_thresh = self.correspondence_thresh

        icp_final_trans, fit, _, _= self.icp_alignment(original_sub_pnts,
                                                       descaled_sub_pnts,
                                                       new_corr_thresh,
                                                       n_iter)

        # o3d.visualization.draw_geometries([descaled_aligned_sub, self.sub_pcd, aligned_src_pcd])
        # o3d.visualization.draw_geometries([descaled_aligned_sub,
        #                                    deepcopy(self.sub_pcd).transform(icp_final_trans)])
        _, aligned_sub_pcd, final_trans = self.refine_registration(icp_final_trans,
                                                               fit,
                                                               self.sub_pcd,
                                                               descaled_aligned_sub,
                                                               new_vox_size)

        # o3d.visualization.draw_geometries([descaled_aligned_sub, aligned_sub_pcd, aligned_src_pcd])
        # o3d.visualization.draw_geometries([_, descaled_aligned_sub])




        # Idx of sub contact surface for original sub pointcloud.
        sub_cp_idx = self.sub_tool.segment_idx_to_idx([self._sub_action_segment],
                                                      sub_action_part_cp_idx)

        if self.visualize:
            final_pcds = [self.src_pcd, deepcopy(self.sub_pcd).transform(final_trans)]
            print "FINAL RESULT"
            o3d.visualization.draw_geometries(final_pcds, "Aligned")


        # Retrun final T and contact surface on original sub tool model
        return final_trans, self.sub_tool.get_pnt(sub_cp_idx)

    def test_scaling(self):
        print "TESTING SCALING..."
        if len(self.scale_Ts) > 1:
            scale_T = T_inv(np.linalg.multi_dot(self.scale_Ts)) # Get inverted scaling T
        else:
            scale_T = T_inv(self.scale_Ts[0])
        descaled_sub = deepcopy(self.T_sub_pcd).transform(scale_T)

        o3d.visualization.draw_geometries([self.sub_pcd, descaled_sub])




    def main(self):
        R, cp = self.get_T_cp(n_iter=5)




if __name__ == '__main__':
    gp = GeneratePointcloud()
    n = 8000
    get_color = True

    # pnts1 = gp.get_random_pointcloud(n)
    # pnts2 = gp.get_random_pointcloud(n)
    # pnts2 = gp.load_pointcloud("../../tool_files/rake.ply", get_segments=True)
    # pnts2 = gp.load_pointcloud("../../tool_files/rake.ply", get_segments=False)
    pnts1 = gp.load_pointcloud("../../tool_files/point_clouds/b_wildo_bowl.ply")
    pnts2 = gp.load_pointcloud("../../tool_files/point_clouds/a_bowl.ply", get_segments=True)
    #pnts1 = np.random.uniform(0, 1, size=(n, 3))
    #pnts2 = np.random.uniform(1.4, 2, size=(n, 3))
    # 
    # pnts1 = gp.load_pointcloud("../../tool_files/point_clouds/a_knifekitchen2.ply",get_segments=True)
    # pnts2 = gp.mesh_to_pointcloud("a_knifekitchen3/2/a_knifekitchen3_out_8_60_fused.ply", n)
    # pnts2 = gp.load_pointcloud("./tool_files/point_clouds/a_bowlchild.ply", None)
    # pnts2 = gp.mesh_to_pointcloud("/rake_remove_box/2/rake_remove_box_out_2_40_fused.ply", n)
    # pnts1 = gp.load_pointcloud('./tool_files/point_clouds/a_bowl.ply', None)
    # pnts1 = gp.mesh_to_pointcloud('a_knifekitchen2/2/a_knifekitchen2_out_4_60_fused.ply',n , get_color)

    src = ToolPointCloud(pnts1, contact_pnt_idx=None)

    # src.visualize_bb()
    sub = ToolPointCloud(pnts2)
    # src = sub
    # sub = shrink_pc(sub)

    cntct_pnt = src.get_pc_bb_axis_frame_centered().argmax(axis=0)[1]
    src.contact_pnt_idx = gen_contact_surface(src.pnts, cntct_pnt)


    print("SRC TOOL")
    # src.visualize()
    # visualize(src.pnts, src.get_pnt(cntct_pnt), src.segments)
    # visualize(src.pnts, src.get_pnt(cntct_pnt), src.segments)
    visualize_contact_area(src.pnts, src.contact_pnt_idx)
    print("SUB TOOL")
    # sub.visualize_bb()
    # sub_seg = [1 if sub.pnts[i,0] > -0.024 else 0 for i in range(sub.pnts.shape[0])]
    # sub.segments=sub_seg
    # visualize(sub.pnts, sub.get_pnt(cntct_pnt2), segments=sub.segments)
    # sub = ToolPointCloud(np.vstack([pnts2.T, sub_seg]).T)

    ts = ToolSubstitution(src, sub, voxel_size=0.01, visualize=True)
    ts.main()
