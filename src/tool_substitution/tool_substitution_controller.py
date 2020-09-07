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
from probreg import bcpd, cpd

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


def visualize_tool(pcd, cp_idx=None, segment=None):

    p = deepcopy(pcd)
    p.paint_uniform_color([0, 0, 1]) # Blue result

    colors = np.asarray(p.colors)

    if not segment is None:
        colors[segment==0, :] = np.array([0,1,0])
        colors[segment==1, :] = np.array([0,.5,.5])
    if not cp_idx is None:
        colors[cp_idx, : ] = np.array([1,0,0])

    p.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([p])


def visualize_reg(src, target, result, result_cp_idx=None, target_cp_idx=None):

    s = deepcopy(src)
    t = deepcopy(target)
    r = deepcopy(result)

    s.paint_uniform_color([1, 0, 0])    # Red src
    t.paint_uniform_color([0, 1, 0]) # Green target
    r.paint_uniform_color([0, 0, 1]) # Blue result

    if not result_cp_idx is None:
        colors = np.asarray(r.colors)
        colors[result_cp_idx, :] = np.array([.5,.5,0])
        r.colors = o3d.utility.Vector3dVector(colors)

    if not target_cp_idx is None:
        colors = np.asarray(t.colors)
        colors[target_cp_idx, :] = np.array([.9,.1,0])
        t.colors = o3d.utility.Vector3dVector(colors)


    o3d.visualization.draw_geometries([s, t, r])


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
        self.src_pcd = self._np_to_o3d(np.asarray(self.src_tool.pnts))
        self.sub_pcd = self._np_to_o3d(np.asarray(self.sub_tool.pnts))
        # Same as above but we will apply all transformations to these
        self.T_src_pcd = deepcopy(self.src_pcd)
        self.T_sub_pcd = deepcopy(self.sub_pcd)

        # Params for ICP
        self.voxel_size = voxel_size
        self.correspondence_thresh = voxel_size * 2
        # self.correspondence_thresh = voxel_size * .1
        # Acceptable amount of alignment loss after applying ICP.
        # Often the quantitatively alignment will drop, but qualitatively it gets better.
        self.fit_ratio_thresh = .75
        # See https://en.wikipedia.org/wiki/Mahalanobis_distance
        self.mahalanobis_thresh = 1.

        self.visualize = visualize
        self.temp_src_T = np.identity(4)  # This stores temporary transformations we will later undo.
        self.scale_Ts = [] # Tracks all scalings of sub tool.

        visualize_tool(self.src_pcd, self.src_tool.contact_pnt_idx, self.src_tool.segments)


    def nonrigid_registration(self, src, sub):
        # tf_param = bcpd.registration_bcpd(sub, src)
        tf_param,a,b = cpd.registration_cpd(sub, src, tf_type_name='nonrigid',maxiter=50000,
                                            w=.1,tol=.00001)
        result_nonrigid = deepcopy(sub)
        result_rigid = deepcopy(sub)

        result_nonrigid.points = tf_param.transform(result_nonrigid.points)
        # result_rigid.points = tf_param.rigid_trans.transform(result_rigid.points)

        sub.paint_uniform_color([1, 0, 0])
        src.paint_uniform_color([0, 1, 0])
        # result_rigid.paint_uniform_color([0, 1, 0])
        result_nonrigid.paint_uniform_color([0, 0, 1])
        # print "trans ", tf_param.rigid_trans

        # o3d.visualization.draw_geometries([sub,src, result_rigid, result_nonrigid])
        o3d.visualization.draw_geometries([sub,src, result_nonrigid])



    def _icp_wrapper(self, sub, src, sub_fpfh, src_fpfh, corr_thresh, n_iter=5):
        """
        Wrapper function for configuring and running icp registration using feature mapping.
        """
        checker = [
            o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(corr_thresh)
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
                            max_correspondence_distance=corr_thresh,
                            estimation_method=est_ptp,
                            ransac_n=4,
                            checkers=checker,
                            criteria=criteria)

            result2 = o3d.registration.registration_icp(sub, src,
                                                        self.voxel_size,
                                                        result1.transformation,
                                                        est_ptpln)

            # visualize_reg(sub, src, deepcopy(sub.transform(result2.transformation)))
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
    def _np_to_o3d(pnts):
        """
        Get o3d pc from ToolPointcloud object.
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pnts)

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
        @ source : (nx3) ndarray of points of src tool.
        @sub_pnts: (mx3) ndarray of points of sub tool.
        @Rs:       list of (3x3) ndarray rotations.
        Returns (R, (nx3) array of rotated points, score) representing
        The rotation of sub_pnts that best aligns with  source .
        """
        if not Rs:
            Rs.append(np.identity(3))

        scores = []
        aligned_pnts = []
        for R in Rs:
            T = get_T_from_R_p(R=R)
            rot_sub_pcd = deepcopy(sub_pcd)
            rot_sub_pcd.transform(T)
            dist = o3d.registration.evaluate_registration(rot_sub_pcd,
                                                          self.T_src_pcd,
                                                          self.correspondence_thresh)
            score = (T, rot_sub_pcd, dist.fitness)
            print "ALIGNMENT SCORE: ", score[2]

            scores.append(score)

        T, rot_sub, fit =  max(scores, key=lambda s:s[2])
        # o3d.visualization.draw_geometries([rot_sub_pcd, self.T_src_pcd], "Aligned")
        return T, fit

    def _get_contact_surface(self, src_cps, sub_pnts,  source ):
        """
        @src_cps: (nk3) ndarray contact surface of src tool (subset of  source ).
        @sub_pnts: (mx3) ndarray points in sub tool pc.
        @ source : (nx3) ndarray points of src tool.

        Returns idx of pnts in sub_pnts estimated to be its contact surface.
        """

        if len(src_cps.shape) > 1: # If there are multiple contact points
            cov = np.cov(src_cps.T) # Then get the mean and cov from these points
            cp_mean = src_cps.mean(axis=0)
        else: # If only one contact point...
            # Then get more by finding 20 nearest neighbors around it.
            tree = KDTree( source )
            _, i = tree.query(src_cps, k=20)
            est_src_surface =  source [i, :]

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
            scaled_sub_pcd = deepcopy(self.T_sub_pcd)
            # permed_scale_f = scale_f[list(p)]
            permed_scale_f = (src_tool_norms / sub_tool_norms)[list(p)]
            T_sub_action_part_scale = get_scaling_T(scale=permed_scale_f)
            # T_sub_action_part_scale = get_T_from_R_p()
            # scaled_sub_pcd = self.T_src_pcd.transform(T_sub_action_part_scale)
            scaled_sub_pcd.transform(T_sub_action_part_scale)


            T_rot, score = self._calc_best_orientation(scaled_sub_pcd,
                                                       [R1, R2, R3, R4])

            scores.append((T_rot, T_sub_action_part_scale, score))

        T_rot, T_sub_action_part_scale, fit =  max(scores, key=lambda s: s[2])

        self.T_sub_pcd.transform(T_sub_action_part_scale)
        self.T_sub_pcd.transform(T_rot)


        self.scale_Ts.append(T_sub_action_part_scale) # Append scaling matrix

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

        # o3d.visualization.draw_geometries([self.T_src_pcd, self.T_sub_pcd])
        self.T_src_pcd.transform(T_src)
        self.T_sub_pcd.transform(T_sub)

        # o3d.visualization.draw_geometries([self.T_src_pcd, self.T_sub_pcd])

        self.temp_src_T = T_inv(T_src) # To account for alignment of src tool along bb axis.

        # Get the lengths of the bounding box sides for each pc
        src_norm = norm(self.T_src_pcd.get_max_bound() - self.T_src_pcd.get_min_bound())
        sub_norm = norm(self.T_sub_pcd.get_max_bound() - self.T_sub_pcd.get_min_bound())

        # Scale by 1 / longest_side to ensure longest side is no longer than 1m.
        largest_span = src_norm if src_norm > sub_norm else sub_norm
        scale_center = np.mean(self._get_src_pnts(get_segments=False)[self.src_tool.contact_pnt_idx, :],
                               axis=0)
        # T_sub_action_part_scale = get_scaling_T(scale=largest_span, center=scale_center)
        T_sub_action_part_scale = get_scaling_T(scale=largest_span)

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

        scaled_src_pc = ToolPointCloud(self._get_src_pnts(), normalize=False)
        scaled_sub_pc = ToolPointCloud(self._get_sub_pnts(), normalize=False)

        print "SCALED SUB SEGMENTS: ", scaled_sub_pc.segment_list
        # Get points in segment of src tool containing the contact area
        src_action_part = scaled_src_pc.get_pnts_in_segment(self._src_action_segment )
        # First, scale both tools to the same size, in order to determine which part of
        # sub tool is the action part.
        _, scale_f = scaled_sub_pc.scale_pnts_to_target(scaled_src_pc)
        T_sub_scale = get_scaling_T(scale=scale_f)
        self.T_sub_pcd.transform(T_sub_scale)
        self.scale_Ts.append(T_sub_scale)
        # o3d.visualization.draw_geometries([self.T_sub_pcd, self.T_src_pcd], "Aligned")
        scaled_sub_pc = ToolPointCloud(self._get_sub_pnts(), normalize=False)
        _, sub_action_pnt_idx = self._get_closest_pnt(src_action_part.mean(axis=0),
                                                      scaled_sub_pc.pnts)


        self._sub_action_segment = scaled_sub_pc.get_segment_from_point(sub_action_pnt_idx)

        sub_action_part = scaled_sub_pc.get_pnts_in_segment(self._sub_action_segment)

        # Scale action parts to same size for better comparison.
        print "SCALED SUB SEGMENTS: ", scaled_sub_pc.segment_list
        _, scale_f = ToolPointCloud(sub_action_part,
                                    normalize=False).scale_pnts_to_target(ToolPointCloud(src_action_part,
                                                                                         normalize=False))
        T_sub_action_part_scale = get_scaling_T(scale=scale_f)

        self.T_sub_pcd.transform(T_sub_action_part_scale)
        self.scale_Ts.append(T_sub_action_part_scale)

        # o3d.visualization.draw_geometries([self.T_src_pcd, self.T_sub_pcd], "Aligned")
        return src_action_part, sub_action_part


    def icp_alignment(self, source, target, correspondence_thresh, n_iter=10):
        """
        Algin sub tool to src tool using ICP.

         """
        # Scale points to be ~1 meter so that self.voxel_size can
        # be consistent for all pointclouds.

        # o3d.visualization.draw_geometries([self.T_src_pcd, self.T_sub_pcd], "Aligned")
        # src_action_pnts = self.scaled_src_pc.pnt

        source, target, source_down, target_down, source_fpfh, target_fpfh = \
            prepare_dataset(source , target, self.voxel_size)

        # Apply icp n_iter times and get best result
        best_icp = self._icp_wrapper(source_down, target_down,
                                     source_fpfh, target_fpfh,
                                     correspondence_thresh,
                                     n_iter)
        # best_icp = self.nonrigid_registration(source_down, target_down)



        icp_fit = best_icp.fitness # Fit score (between [0.0,1.0])
        icp_trans = best_icp.transformation # T matrix

        print "ICP RESULTS"
        visualize_reg(source, target, deepcopy(source).transform(icp_trans))
        return icp_trans, icp_fit, source, target

    def refine_registration(self, init_trans, source, target, voxel_size):


        pose_graph = o3d.registration.PoseGraph()
        accum_pose = np.identity(4)
        pose_graph.nodes.append(o3d.registration.PoseGraphNode(accum_pose))

        pcds = [deepcopy(source), deepcopy(target)]
        sub_idx, src_idx = (0, 1)

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

        # Apply calculated transformations to pcds
        for pcd_id in range(n_pcds):
            trans = pose_graph.nodes[pcd_id].pose
            print "TRANS ", trans
            pcds[pcd_id].transform(trans)

        fit = o3d.registration.evaluate_registration(pcds[sub_idx],
                                                     pcds[src_idx],
                                                     self.correspondence_thresh).fitness


        # If these new transformations lower fit score, dont apply them.

        # print "SRC REG"
        # visualize_reg(source, target, pcds[src_idx])
        # print "SUB REG"
        # visualize_reg(source, target, pcds[sub_idx])
        print "INV SRC REG"
        sub_T, src_T = pose_graph.nodes[sub_idx].pose, pose_graph.nodes[src_idx].pose
        # This alg rotates both the src and sub tools. We dont want the src tool to rotate 
        # at all so we apply its inverse T onto the sub tool to the same effect.
        final_T = np.matmul(T_inv(src_T), sub_T)
        aligned_source = deepcopy(source).transform(final_T)
        visualize_reg(source, target, aligned_source)

        return pcds[src_idx], aligned_source, final_T, fit


    def get_T_cp(self, n_iter=10):
        """
        Refine Initial ICP alignment and return final T rotation matrix and sub contact pnts.
        """

        # Scale and orient tools based on their principal axes
        self._scale_pcs()
        # Find best initial alignment via rotations over these axes.
        align_fit = self._align_pnts()
        # Get the points in the action segments of both tools
        src_action_pnts, sub_action_pnts = self.get_tool_action_parts()


        icp_trans, fit, sub_action_pcd, src_action_pcd = self.icp_alignment(sub_action_pnts,
                                                                            src_action_pnts,
                                                                            self.correspondence_thresh,
                                                                            n_iter)

        print "ORGINAL ALIGN FITNESS: ", align_fit
        print "INIT ICP FITNESS:      ", fit
        fit_ratio = fit / align_fit

        if fit_ratio > self.fit_ratio_thresh:
            print "USING INIT. ICP RESULTS.."
            T_align = icp_trans
        else:
            print "SKIPPING  INIT. ICP RESULTS, USING INIT. ALIGNMENT."
            T_align = get_T_from_R_p()


        _, aligned_sub_pcd, T_align,_ = self.refine_registration(T_align,
                                                               sub_action_pcd,
                                                               src_action_pcd,
                                                               self.voxel_size)

        print "TEST ALIGNMENT"
        o3d.visualization.draw_geometries([src_action_pcd, src_action_pcd, self.T_src_pcd])

        self.T_sub_pcd.transform(T_align)
        # aligned_sub_pcd = sub_action_pcd.transform(T_align)

        print "sub action part and full tool alignment"
        o3d.visualization.draw_geometries([self.T_sub_pcd, aligned_sub_pcd])

        print "src action part and full tool alignment"
        o3d.visualization.draw_geometries([src_action_pcd, self.T_src_pcd])
        aligned_src = np.asarray(src_action_pcd.points)
        aligned_sub = np.asarray(aligned_sub_pcd.points)

        # visualize_two_pcs(aligned_sub, aligned_src)
        src_action_part_cp_idx = self.src_tool.idx_to_segment_idx(self.src_tool.contact_pnt_idx)
        src_contact_pnt = aligned_src[src_action_part_cp_idx, :]
        # src_contact_pnt = self._get_src_pnts(get_segments=False)[src_action_part_cp_idx, :]
        # src_contact_pnt = self.src_tool.pnts[src_action_part_cp_idx, :]
        # Reshape point for easy computation
        src_contact_pnt = src_contact_pnt.reshape(1,-1) \
            if len(src_contact_pnt.shape) == 1 else src_contact_pnt

        # Estimate contact surface on the sub tool
        sub_action_part_cp_idx = self._get_contact_surface(src_contact_pnt,
                                                    aligned_sub,
                                                    aligned_src)


        # Corresponding idx of sub contact surface for original sub pointcloud.
        sub_cp_idx = self.sub_tool.segment_idx_to_idx([self._sub_action_segment],
                                                      sub_action_part_cp_idx)

        print "SUB CP IDX ", sub_cp_idx

        mean_sub_action_part_cp = np.mean(aligned_sub[sub_action_part_cp_idx, :],axis=0)
        # mean_sub_action_part_cp = np.mean(src_contact_pnt)
        scale_T = T_inv(np.linalg.multi_dot(self.scale_Ts)) # Get inverted scaling T
        scale_f = np.diag(scale_T)[:3] # get first 3 diag entries.
        final_scale_T = get_scaling_T(scale_f, mean_sub_action_part_cp)

        visualize_tool(aligned_sub_pcd, sub_action_part_cp_idx)
        visualize_tool(self.T_sub_pcd, sub_cp_idx)


        print "DESCALING"
        descaled_aligned_sub = deepcopy(self.T_sub_pcd).transform(final_scale_T)
        # descaled_aligned_sub = deepcopy(aligned_sub_pcd).transform(scale_T)
        # descaled_sub_action = deepcopy(aligned_sub_pcd).transform(final_scale_T)
        visualize_reg(aligned_sub_pcd, self.sub_pcd, descaled_aligned_sub)
        self.T_sub_pcd.transform(final_scale_T)
        # descaled_sub_mean_cp = np.mean(np.asarray(descaled_aligned_sub.points)[sub_cp_idx, :], axis=0)
        # descaled_sub_mean_cp = np.mean(np.asarray(descaled_aligned_sub.points)[sub_action_part_cp_idx, :], axis=0)
        print "VISUALIZING DESCALE BEFORE TRANS"
        visualize_reg(aligned_sub_pcd, src_action_pcd, descaled_aligned_sub)

        # T_translate = get_T_from_R_p(p=np.mean(src_contact_pnt,axis=0)-descaled_sub_mean_cp)
        # descaled_aligned_sub.transform(T_translate)
        # self.T_sub_pcd.transform(T_translate)

        # print "VISUALIZING DESCALE AFTER TRANS"
        # visualize_reg(aligned_sub_pcd, src_action_pcd, descaled_aligned_sub)


        print "HELOO"
        visualize_reg(descaled_aligned_sub, self.src_pcd,
                      deepcopy(descaled_aligned_sub).transform(self.temp_src_T))
        descaled_aligned_sub.transform(self.temp_src_T)
        # This transformation accounts for the alignment of the src tool to its principal axes
        self.T_sub_pcd.transform(self.temp_src_T)

        descaled_sub_pnts = np.asarray(self._get_sub_pnts(get_segments=False))
        original_sub_pnts = np.asarray(self.sub_pcd.points)

        new_vox_size = self.voxel_size
        new_corr_thresh = self.correspondence_thresh

        icp_T, icp_fit, _, _= self.icp_alignment(original_sub_pnts,
                                                 descaled_sub_pnts,
                                                 new_corr_thresh,
                                                   n_iter)


        aligned_sub_pcd,_, refine_T, refine_fit = self.refine_registration(icp_T,
                                                                           self.sub_pcd,
                                                                           # descaled_aligned_sub,
                                                                           self.T_sub_pcd ,
                                                                           new_vox_size)


        print "ICP fit: ", icp_fit
        print "Refine fit: ", refine_fit

        final_trans = icp_T if icp_fit > refine_fit else refine_T
        # self.T_sub_pcd.transform(final_trans)

        # Algin the centroids of contact areas of the tools.
        src_mean_cp = np.mean(np.asarray(self.src_pcd.points)[self.src_tool.contact_pnt_idx,:], axis=0)
        sub_mean_cp = np.mean(np.asarray(self.T_sub_pcd.points)[sub_cp_idx, :], axis=0)
        T_translate = get_T_from_R_p(p=(src_mean_cp-sub_mean_cp))
        final_trans = np.matmul(T_translate, final_trans)


        if self.visualize:
            print "FINAL RESULT"
            visualize_reg(self.sub_pcd,
                          self.src_pcd,
                          deepcopy(self.sub_pcd).transform(final_trans),
                          result_cp_idx=sub_cp_idx,
                          target_cp_idx=self.src_tool.contact_pnt_idx)


        # Must account for the fact that the src and sub tool pcs have been centered at
        # 0, so we create one final translation transformation
        orig_sub_tool = self._np_to_o3d(self.sub_tool.get_unnormalized_pc())
        orig_src_tool = self._np_to_o3d(self.src_tool.get_unnormalized_pc())
        orig_sub_tool.transform(final_trans)

        src_mean_cp = np.mean(np.asarray(orig_src_tool.points)[self.src_tool.contact_pnt_idx,:], axis=0)
        sub_mean_cp = np.mean(np.asarray(orig_sub_tool.points)[sub_cp_idx, :], axis=0)
        T_translate = get_T_from_R_p(p=(src_mean_cp-sub_mean_cp))
        final_trans = np.matmul(T_translate, final_trans)

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
        return self.get_T_cp(n_iter=3)




if __name__ == '__main__':
    gp = GeneratePointcloud()
    n = 8000
    get_color = True

    # pnts1 = gp.get_random_pointcloud(n)
    # pnts2 = gp.get_random_pointcloud(n)
    # pnts2 = gp.load_pointcloud("../../tool_files/rake.ply", get_segments=False)
    pnts1 = gp.load_pointcloud("../../tool_files/rake.ply", get_segments=True)
    # pnts1 = gp.load_pointcloud("../../tool_files/point_clouds/b_wildo_bowl.ply", get_segments=False)
    # pnts2 = gp.load_pointcloud("../../tool_files/point_clouds/a_bowl.ply", get_segments=False)
    #pnts1 = np.random.uniform(0, 1, size=(n, 3))
    #pnts2 = np.random.uniform(1.4, 2, size=(n, 3))
    # 
    pnts2 = gp.load_pointcloud("../../tool_files/point_clouds/a_knifekitchen2.ply",get_segments=True)
    # pnts1 = gp.load_pointcloud("../../tool_files/point_clouds/a_chineseknife.ply", get_segments=False)
    # pnts2 = gp.load_pointcloud("./tool_files/point_clouds/a_bowlchild.ply", None)
    # pnts2 = gp.mesh_to_pointcloud("/rake_remove_box/2/rake_remove_box_out_2_40_fused.ply", n)
    # pnts1 = gp.load_pointcloud('./tool_files/point_clouds/a_bowl.ply', None)
    # pnts1 = gp.mesh_to_pointcloud('a_knifekitchen2/2/a_knifekitchen2_out_4_60_fused.ply',n , get_color)

    src = ToolPointCloud(pnts1, contact_pnt_idx=None, normalize=True)

    # src.visualize_bb()
    sub = ToolPointCloud(pnts2, normalize=True)
    # src = sub
    # sub = shrink_pc(sub)

    cntct_pnt = src.get_pc_bb_axis_frame_centered().argmax(axis=0)[1]
    src.contact_pnt_idx = gen_contact_surface(src.pnts, cntct_pnt)


    print("SRC TOOL")
    # src.visualize()
    # visualize(src.pnts, src.get_pnt(cntct_pnt), src.segments)
    # visualize(src.pnts, src.get_pnt(cntct_pnt), src.segments)
    # visualize_contact_area(src.pnts, src.contact_pnt_idx)
    print("SUB TOOL")
    # sub.visualize_bb()
    # sub_seg = [1 if sub.pnts[i,0] > -0.024 else 0 for i in range(sub.pnts.shape[0])]
    # sub.segments=sub_seg
    # visualize(sub.pnts, sub.get_pnt(cntct_pnt2), segments=sub.segments)
    # sub = ToolPointCloud(np.vstack([pnts2.T, sub_seg]).T)

    ts = ToolSubstitution(src, sub, voxel_size=0.005, visualize=True)
    T, cp = ts.main()

    src_pcd = o3d.geometry.PointCloud()
    sub_pcd = o3d.geometry.PointCloud()

    src_pcd.points = o3d.utility.Vector3dVector(pnts1[:, :3])
    sub_pcd.points = o3d.utility.Vector3dVector(pnts2[:, :3])

    visualize_reg(sub_pcd, src_pcd, sub_pcd.transform(T))
