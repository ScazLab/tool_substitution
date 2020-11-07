#!/usr/bin/env python
import numpy as np
from copy import deepcopy
from numpy.linalg import norm
import random

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
                  r_x, r_y, r_z, visualize_contact_area,
                  visualize_reg, visualize_tool, visualize_multiple_cps,
                  align_pcd_select_size, get_homogenous_transformation_matrix_inverse)

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

def get_np_pc_distance(pc1, pc2):
    pc1_o3d = o3d.geometry.PointCloud()
    pc1_o3d.points = o3d.utility.Vector3dVector(pc1)
    
    pc2_o3d = o3d.geometry.PointCloud()
    pc2_o3d.points = o3d.utility.Vector3dVector(pc2)
    
    return get_o3d_pc_distance(pc1_o3d, pc2_o3d)
    
def get_o3d_pc_distance(pc1, pc2):
    distance = pc1.compute_point_cloud_distance(pc2)
    #print "min distance: ", np.min(distance)
    #print "max distance: ", np.max(distance)
    #print "average distance: ", np.average(distance)
    return np.average(distance)

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
        
        self.src_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=5))
        self.sub_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=5))
        
        # Same as above but we will apply all transformations to these
        self.T_src_pcd = deepcopy(self.src_pcd)
        self.T_sub_pcd = deepcopy(self.sub_pcd)

        # Params for ICP
        self.voxel_size = voxel_size
        # self.correspondence_thresh = voxel_size * 2
        self.correspondence_thresh = voxel_size *.5
        # self.correspondence_thresh = voxel_size * .1
        # Acceptable amount of alignment loss after applying ICP.
        # Often the quantitatively alignment will drop, but qualitatively it gets better.
        self.fit_ratio_thresh = .75
        # See https://en.wikipedia.org/wiki/Mahalanobis_distance
        self.mahalanobis_thresh = 1.

        self.visualize = visualize
        self.temp_src_T = np.identity(4)  # This stores temporary transformations we will later undo.
        self.scale_Ts = [] # Tracks all scalings of sub tool.

        if self.visualize:
            visualize_tool(self.src_pcd, self.src_tool.contact_pnt_idx, 
                           self.src_tool.segments, "Src tool with contact surface")


    def nonrigid_registration(self, src, sub):
        # tf_param = bcpd.registration_bcpd(sub, src)
        print "PERFORMING NON RIGID REG"
        tf_param,a,b = cpd.registration_cpd(sub, src, tf_type_name='nonrigid',
                                            maxiter=500000, w=.1, tol=.00001)
        result_nonrigid = deepcopy(sub)
        result_rigid = deepcopy(sub)

        result_nonrigid.points = tf_param.transform(result_nonrigid.points)
        # result_rigid.points = tf_param.rigid_trans.transform(result_rigid.points)
        # result_rigid.points = tf_param.v + result_rigid.points

        sub.paint_uniform_color([1, 0, 0])
        src.paint_uniform_color([0, 1, 0])
        # result_rigid.paint_uniform_color([0, 1, 0])
        result_nonrigid.paint_uniform_color([0, 0, 1])
        # print "trans ", tf_param.rigid_trans

        # print "rigid result"
        # visualize_reg(sub, src, result_rigid)
        print "nonrigid result"
        if self.visualize:
            visualize_reg(sub,src, result_nonrigid, name="nonrigid result")

        return result_nonrigid


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
        icp_criteria = o3d.registration.ICPConvergenceCriteria(max_iteration=300)


        results = []
        distances = []
        
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
                                                        est_ptpln,
                                                        criteria=icp_criteria)

            # visualize_reg(sub, src, deepcopy(sub.transform(result2.transformation)))
            results.append(result2)
            distance = get_o3d_pc_distance(src, sub)
            distances.append(distance)
            #print "ICP FITNESS: ", result2.fitness
            print "Distance: ", distance
            #if result2.fitness == 1.0: break # 1.0 is best possible score.

        index_min = np.argmin(distances)
        
        #return max(results, key=lambda i:i.fitness)
        return results[index_min]

    def _get_sub_pnts(self, T_sub_pcd = None, get_segments=True, segments=None):
        """
        Get ndarray of points from o3d pointcloud.
        """
        if T_sub_pcd is None:
            T_sub_pcd = self.T_sub_pcd
        pnts = deepcopy(np.asarray(T_sub_pcd.points))
        #print "in _get_sub_pnts"
        #print "pnts before"
        #print pnts
        if get_segments:
            if segments is None:
                segments = deepcopy(self.sub_tool.segments)
            pnts = np.vstack([pnts.T, deepcopy(segments)]).T
        
        #print "pnts after"
        #print pnts
        return pnts

    def _get_src_pnts(self, T_src_pcd = None, get_segments=True, segments=None):
        """
        Get ndarray of points from o3d pointcloud.
        """
        if T_src_pcd is None:
            T_src_pcd = self.T_src_pcd
        pnts = deepcopy(np.asarray(T_src_pcd.points))
        if get_segments:
            if segments is None:
                segments = deepcopy(self.src_tool.segments)
            pnts = np.vstack([pnts.T, deepcopy(segments)]).T

        return pnts

    @staticmethod
    def _np_to_o3d(pnts):
        """
        Get o3d pc from ToolPointcloud object. ?? from numpy object actually???
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(deepcopy(pnts[:, :3])) # just in case there is a 4th column with the segments

        return pcd

    def _calc_center_and_align_T(self, pc):
        """
        Creates a centered and aligned ToolPointCloud from unaligned ToolPointCloud
        """
        centroid = pc.get_bb_centroid() * -1 # subtract bb mean to center points
        R = np.linalg.inv(pc.get_axis()) # align points to bb axes.
        
        return get_T_from_R_p(R=R, p=centroid)


    def _calc_best_orientation(self, T_src_pcd, sub_pcd, Rs):
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
            
            # use fitness
            #dist = o3d.registration.evaluate_registration(rot_sub_pcd,
                                                          #self.T_src_pcd,
                                                          #self.correspondence_thresh)
            #score = (T, rot_sub_pcd, dist.fitness)
            
            # use distance
            distance = get_o3d_pc_distance(T_src_pcd, rot_sub_pcd)
            print "R"
            print R
            print "distance"
            print distance
            #o3d.visualization.draw_geometries([self.T_src_pcd, rot_sub_pcd], "aligning pikachu")
            score = (T, rot_sub_pcd, distance)
            
            print "ALIGNMENT SCORE: ", score[2]
            #visualize_reg(sub_pcd, self.T_src_pcd, rot_sub_pcd)

            scores.append(score)
        
        # use fitness
        #T, rot_sub, fit = max(scores, key=lambda s:s[2])
        
        # use distance
        T, rot_sub, fit = min(scores, key=lambda s:s[2])
        #o3d.visualization.draw_geometries([self.T_src_pcd, rot_sub], "aligning pikachu")       
        
        #o3d.visualization.draw_geometries([rot_sub, self.T_src_pcd], "Best orientation (score: {})".format(fit))
        return T, fit

    def _get_contact_surface(self, src_cps, sub_pnts, source):
        """
        @src_cps: (nk3) ndarray contact surface of src tool (subset of source).
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
    
    def _find_sub_action_part(self, src_action_pnts, sub_pnts):
        print "src_action_pnts"
        print src_action_pnts
        
        print "sub_pnts"
        print sub_pnts
        
        sub_segments = np.unique(sub_pnts[:, -1]).tolist()
        
        sub_action_segment = -1
        sub_action_segment_indices = None
        sub_action_segment_pnts = None
        sub_action_segment_distance = 100.
        
        for segment in sub_segments:
            print "segment: ", segment
            indices = np.where(sub_pnts[:, -1] == segment)
            action_pnts = sub_pnts[indices][:, :-1]
            distance = get_np_pc_distance(src_action_pnts, action_pnts)
            print "distance: ", distance
            if distance < sub_action_segment_distance:
                sub_action_segment_distance = distance
                sub_action_segment = segment
                sub_action_segment_indices = deepcopy(indices)
                sub_action_segment_pnts = deepcopy(action_pnts)
        
        print "sub_action_segment: ", sub_action_segment
        
        return sub_action_segment, sub_action_segment_pnts, sub_action_segment_indices

    def _rotate_np_with_segments(self, np_with_segments, p):
        # np_with_segments: n * 4
        
        R = []
        for index in list(p):
            if index == 0:
                R.append([1., 0., 0.])
            elif index == 1:
                R.append([0., 1., 0.])
            elif index == 2:
                R.append([0., 0., 1.])
        
        R = np.array(R).T
        
        segments = deepcopy(np_with_segments[:, -1])
        
        pnts = deepcopy(np_with_segments[:, :-1]).T # 3 by n
        
        pnts = np.matmul(R, pnts).T # n by 3
        
        pnts = np.hstack((pnts, np.array([segments]).T))
        
        #print "R"
        #print R
        T = np.vstack((R, np.array([[0., 0., 0.]])))
        T = np.hstack((T, np.array([[0., 0., 0., 1.]]).T))
        
        return T, pnts

    #def _add_segments_to_pnts(self, pnts, segments):
        ## pnts: n by 3 np array
        ## segments: an np array with len n (n by 1)
        
        ## return n by 4 np array
        #return np.hstack((pnts, np.array([segments]).T))
    
    #def _get_segments_from_pnts(self, pnts):
        ## pnts: n by 4 np array
        
        ## return an np array with len n (n by 1)
        #return pnts[:, -1]
    
    #def _generate_np_segments(self, pnts_with_segments, pnts_pc):
        ## pnts_with_segments: n by 4 np array
        ## pnts_pc: open3d pc with correct transformation
        
        #segments = self._get_segments_from_pnts(pnts_with_segments)
        #pnts_correct_transformation = np.asarray(pnts_pc.points)
        
        #pnts_correct_transformation_with_segment = self._add_segments_to_pnts(pnts_correct_transformation, segments)
        
        #return segments

    def _align_pnts(self, src_np_pnts, sub_np_pnts, keep_proportion=True):
        """
        Scale sub_pc to and then detemine most similar orientation to src_pc
        Returns ndarray of sub_pc pnts in best orientation.
        src_np_pnts: n by 4 matrix (including segments)
        sub_np_pnts: n by 4 matrix (including segments)
        """
        copy_src_np_pnts = deepcopy(src_np_pnts)
        copy_sub_np_pnts = deepcopy(sub_np_pnts)
        
        T_src_pcd, T_sub_pcd, temp_src_T, T_src_to_return, T_sub_to_return = self._scale_pcs(src_np_pnts=copy_src_np_pnts, sub_np_pnts=copy_sub_np_pnts)
        
        #o3d.visualization.draw_geometries([T_src_pcd, T_sub_pcd], "initial normalize")
        
        copy_src_np_pnts = self._get_src_pnts(T_src_pcd=T_src_pcd, segments=copy_src_np_pnts[:, -1])
        copy_sub_np_pnts = self._get_sub_pnts(T_sub_pcd=T_sub_pcd, segments=copy_sub_np_pnts[:, -1])

        # Test current orientation
        R1 = np.identity(3)
        R2 = r_x(np.pi)
        R3 = r_y(np.pi)
        R4 = r_z(np.pi)

        scores = []
        # Ger perms of all indx
        # _, scale_f = self.sub_tool.scale_pnts_to_target(self.scaled_src_pc)
        #src_tool_norms = ToolPointCloud(self._get_src_pnts()).bb.norms
        #sub_tool_norms = ToolPointCloud(self._get_sub_pnts()).bb.norms
        
        print "****************************************src normalized axis"
        src_bb = ToolPointCloud(copy_src_np_pnts, normalize=False).bb
        src_bb._calculate_axis()
        print "axis is ", src_bb.get_normalized_axis()
        print "norm is ", src_bb.norms
        print "****************************************sub normalized axis"
        sub_bb = ToolPointCloud(copy_sub_np_pnts, normalize=False).bb
        sub_bb._calculate_axis()
        print "axis is ", sub_bb.get_normalized_axis() 
        print "norm is ", sub_bb.norms
        
        max_length = np.max(src_bb.norms)

        #for p in permutations([0, 1, 2]):
        for p in [[0, 1, 2], [1, 0, 2], [2, 1, 0]]:
            print "........................................................."
            print "p"
            print p
            T, sub_pnts = self._rotate_np_with_segments(copy_sub_np_pnts, p)
            src_tool_norm = src_bb.norms
            sub_tool_norm = deepcopy(sub_bb.norms)[list(p)]
            print "src_tool_norm: ", src_tool_norm
            print "sub_tool_norm: ", sub_tool_norm
            #sub_tool_norms = sub_tool_norms[list(p)]
            scaled_sub_pcd = self._np_to_o3d(copy_sub_np_pnts)
            scaled_sub_pcd.transform(T)
            # permed_scale_f = scale_f[list(p)]
            #permed_scale_f = (src_tool_norms / sub_tool_norms)[list(p)]
            permed_scale_f = np.array([1., 1., 1.])
            if not keep_proportion:
                permed_scale_f = src_tool_norm / sub_tool_norm
            else:
                scale = np.max(src_tool_norm) / np.max(sub_tool_norm)
                permed_scale_f = np.array([scale, scale, scale])
            #print "src_tool_norms"
            #print src_tool_norms
            #print "sub_tool_norms"
            #print sub_tool_norms
            print "permed_scale_f"
            print permed_scale_f
            T_sub_action_part_scale = get_scaling_T(scale=permed_scale_f)
            # T_sub_action_part_scale = get_T_from_R_p()
            # scaled_sub_pcd = self.T_src_pcd.transform(T_sub_action_part_scale)
            scaled_sub_pcd.transform(T_sub_action_part_scale)

            T_rot, score = self._calc_best_orientation(T_src_pcd, scaled_sub_pcd,
                                                       [R1, R2, R3, R4])
            
            #copy_scaled_sub_pcd = deepcopy(self.T_sub_pcd)
            #copy_scaled_sub_pcd.transform(np.matmul(T_rot, T))
            #copy_scaled_sub_pcd.transform(T_sub_action_part_scale)
            #o3d.visualization.draw_geometries([scaled_sub_pcd, copy_scaled_sub_pcd], "scaled sub pcd")
            
            T_rot = np.matmul(T_rot, T)
            #T_sub_action_part_scale = np.matmul(T_sub_action_part_scale, T)
            scores.append((T_rot, T_sub_action_part_scale, score))
            
            #copy_scaled_sub_pcd = deepcopy(self.T_sub_pcd)
            #copy_scaled_sub_pcd.transform(T_rot)
            #copy_scaled_sub_pcd.transform(T_sub_action_part_scale)
            #o3d.visualization.draw_geometries([scaled_sub_pcd, copy_scaled_sub_pcd], "scaled sub pcd")            
            
            #print "saved is: T_rot"
            #print T_rot
            #print "saved is: T_sub_action_part_scale"
            #print T_sub_action_part_scale

        # use fitness
        #T_rot, T_sub_action_part_scale, fit =  max(scores, key=lambda s: s[2])
        
        # use distance
        T_rot, T_sub_action_part_scale, distance = min(scores, key=lambda s: s[2])

        print "selected is T_rot"
        print T_rot
        print "selected is T_sub_action_part_scale"
        print T_sub_action_part_scale

        T_sub_pcd = self._np_to_o3d(copy_sub_np_pnts)
        T_sub_pcd.transform(T_rot)
        T_sub_to_return = np.matmul(T_rot, T_sub_to_return)
        #self.T_sub_pcd.transform(T_rot)
        #o3d.visualization.draw_geometries([self.T_sub_pcd, self.T_src_pcd], "after rotate")
        T_sub_pcd.transform(T_sub_action_part_scale)
        T_sub_to_return = np.matmul(T_sub_action_part_scale, T_sub_to_return)
        #self.T_sub_pcd.transform(T_sub_action_part_scale)
        #o3d.visualization.draw_geometries([self.T_sub_pcd, self.T_src_pcd], "after scaling")

        #scale_Ts = [T_sub_action_part_scale]

        #self.scale_Ts.append(T_sub_action_part_scale) # Append scaling matrix

        return distance, distance / max_length, T_sub_action_part_scale, T_src_pcd, T_sub_pcd, temp_src_T, T_src_to_return, T_sub_to_return

    #def _align_pnts(self):
        #"""
        #Scale sub_pc to and then detemine most similar orientation to src_pc
        #Returns ndarray of sub_pc pnts in best orientation.
        #"""

        ## Test current orientation
        #R1 = np.identity(3)
        #R2 = r_x(np.pi)
        #R3 = r_y(np.pi)
        #R4 = r_z(np.pi)

        #scores = []
        ## Ger perms of all indx
        ## _, scale_f = self.sub_tool.scale_pnts_to_target(self.scaled_src_pc)
        ##src_tool_norms = ToolPointCloud(self._get_src_pnts()).bb.norms
        ##sub_tool_norms = ToolPointCloud(self._get_sub_pnts()).bb.norms
        
        #print "****************************************src normalized axis"
        #src_bb = ToolPointCloud(self._get_src_pnts()).bb
        #src_bb._calculate_axis()
        #print "axis is ", src_bb.get_normalized_axis()
        #print "norm is ", src_bb.norms
        #print "****************************************sub normalized axis"
        #sub_bb = ToolPointCloud(self._get_sub_pnts()).bb
        #sub_bb._calculate_axis()
        #print "axis is ", sub_bb.get_normalized_axis() 
        #print "norm is ", sub_bb.norms
        
        #print "!!!!!!!!!!!!!!!self._get_src_pnts()"
        #print self._get_src_pnts()

        ##for p in permutations([0, 1, 2]):
        #for p in [[0, 1, 2], [1, 0, 2], [2, 1, 0]]:
            #print "........................................................."
            #print "p"
            #print p
            #T, sub_pnts = self._rotate_np_with_segments(self._get_sub_pnts(), p)
            #src_tool_norm = src_bb.norms
            #sub_tool_norm = deepcopy(sub_bb.norms)[list(p)]
            #print "src_tool_norm: ", src_tool_norm
            #print "sub_tool_norm: ", sub_tool_norm
            ##sub_tool_norms = sub_tool_norms[list(p)]
            #scaled_sub_pcd = deepcopy(self.T_sub_pcd)
            #scaled_sub_pcd.transform(T)
            ## permed_scale_f = scale_f[list(p)]
            ##permed_scale_f = (src_tool_norms / sub_tool_norms)[list(p)]
            #permed_scale_f = src_tool_norm / sub_tool_norm
            ##print "src_tool_norms"
            ##print src_tool_norms
            ##print "sub_tool_norms"
            ##print sub_tool_norms
            #print "src_tool_norm / sub_tool_norm"
            #print src_tool_norm / sub_tool_norm
            #T_sub_action_part_scale = get_scaling_T(scale=permed_scale_f)
            ## T_sub_action_part_scale = get_T_from_R_p()
            ## scaled_sub_pcd = self.T_src_pcd.transform(T_sub_action_part_scale)
            #scaled_sub_pcd.transform(T_sub_action_part_scale)

            #T_rot, score = self._calc_best_orientation(scaled_sub_pcd,
                                                       #[R1, R2, R3, R4])
            
            ##copy_scaled_sub_pcd = deepcopy(self.T_sub_pcd)
            ##copy_scaled_sub_pcd.transform(np.matmul(T_rot, T))
            ##copy_scaled_sub_pcd.transform(T_sub_action_part_scale)
            ##o3d.visualization.draw_geometries([scaled_sub_pcd, copy_scaled_sub_pcd], "scaled sub pcd")
            
            #T_rot = np.matmul(T_rot, T)
            ##T_sub_action_part_scale = np.matmul(T_sub_action_part_scale, T)
            #scores.append((T_rot, T_sub_action_part_scale, score))
            
            ##copy_scaled_sub_pcd = deepcopy(self.T_sub_pcd)
            ##copy_scaled_sub_pcd.transform(T_rot)
            ##copy_scaled_sub_pcd.transform(T_sub_action_part_scale)
            ##o3d.visualization.draw_geometries([scaled_sub_pcd, copy_scaled_sub_pcd], "scaled sub pcd")            
            
            ##print "saved is: T_rot"
            ##print T_rot
            ##print "saved is: T_sub_action_part_scale"
            ##print T_sub_action_part_scale

        ## use fitness
        ##T_rot, T_sub_action_part_scale, fit =  max(scores, key=lambda s: s[2])
        
        ## use distance
        #T_rot, T_sub_action_part_scale, fit = min(scores, key=lambda s: s[2])

        #print "selected is T_rot"
        #print T_rot
        #print "selected is T_sub_action_part_scale"
        #print T_sub_action_part_scale

        #self.T_sub_pcd.transform(T_rot)
        ##o3d.visualization.draw_geometries([self.T_sub_pcd, self.T_src_pcd], "after rotate")
        #self.T_sub_pcd.transform(T_sub_action_part_scale)
        ##o3d.visualization.draw_geometries([self.T_sub_pcd, self.T_src_pcd], "after scaling")

        #self.scale_Ts.append(T_sub_action_part_scale) # Append scaling matrix

        #return fit

    def get_random_contact_pnt(self):
        """
        Get a random contact point and rotation matrix for substitute tool.
        """

        src = self._np_to_o3d(self.src_tool.get_unnormalized_pc())
        sub = self._np_to_o3d(self.sub_tool.get_unnormalized_pc())
        sub_init = deepcopy(sub)

        R = Rot.random(num=1).as_dcm()[0] # Generate a radom rotation matrix.
        T = get_T_from_R_p(R=R)
        sub.transform(T)

        src_cp = np.mean(np.asarray(src.points)[self.src_tool.contact_pnt_idx, :],
                         axis=0)
        _, sub_contact_pnt_idx = self._get_closest_pnt(src_cp,
                                                       np.asarray(sub.points))

        # Translate the the sub tool such that the contact points are overlapping
        sub_cp = np.asarray(sub.points)[sub_contact_pnt_idx, :]
        trans_T = get_T_from_R_p(p=src_cp-sub_cp)
        final_T = np.matmul(trans_T, T)

        sub.transform(trans_T)

        # Generate a contact surface around the closest point.
        sub_contact_pnt = gen_contact_surface(np.asarray(sub.points),
                                              sub_contact_pnt_idx)
        # sub_contact_pnt = self.sub_tool.get_pnt(sub_contact_pnt_idx)

        # T = get_T_from_R_p(p=np.zeros((1,3)), R=R )
        if self.visualize:
            visualize_reg(sub_init, src, sub,
                          result_cp_idx=sub_contact_pnt,
                          target_cp_idx=self.src_tool.contact_pnt_idx,
                          name="get random contact pnts")

        return final_T, sub_contact_pnt

    def _scale_pcs(self, src_np_pnts=None, sub_np_pnts=None):
        """
        Ensures that all pcs are of a consistent scale (~1m ) so that ICP default params will work.
        Current: did not scale it, just center it.
        """
        T_src_to_return = np.identity(4)
        T_sub_to_return = np.identity(4)
        
        if src_np_pnts is None:
            src_np_pnts = self._get_src_pnts()
        src_tool_tpc = ToolPointCloud(src_np_pnts, normalize=False)

        if sub_np_pnts is None:
            src_np_pnts = self._get_sub_pnts()
        sub_tool_tpc = ToolPointCloud(sub_np_pnts, normalize=False)

        T_src = self._calc_center_and_align_T(src_tool_tpc)
        T_sub = self._calc_center_and_align_T(sub_tool_tpc)

        # o3d.visualization.draw_geometries([self.T_src_pcd, self.T_sub_pcd])
        T_src_pcd = self._np_to_o3d(src_np_pnts)
        T_sub_pcd = self._np_to_o3d(sub_np_pnts)
        
        T_src_pcd.transform(T_src)
        T_sub_pcd.transform(T_sub)
        T_src_to_return = np.matmul(T_src, T_src_to_return)
        T_sub_to_return = np.matmul(T_sub, T_sub_to_return)

        #o3d.visualization.draw_geometries([self.T_src_pcd, self.T_sub_pcd], "INITIAL ALIGNMENT")

        temp_src_T = T_inv(T_src) # To account for alignment of src tool along bb axis.

        ## Get the lengths of the bounding box sides for each pc
        #src_norm = norm(T_src_pcd.get_max_bound() - T_src_pcd.get_min_bound())
        #sub_norm = norm(T_sub_pcd.get_max_bound() - T_sub_pcd.get_min_bound())

        ## Scale by 1 / longest_side to ensure longest side is no longer than 1m.
        #largest_span = src_norm if src_norm > sub_norm else sub_norm
        #scale_center = np.mean(self._get_src_pnts(get_segments=False)[self.src_tool.contact_pnt_idx, :],
                               #axis=0)
        ## T_sub_action_part_scale = get_scaling_T(scale=largest_span, center=scale_center)
        #T_sub_action_part_scale = get_scaling_T(scale=largest_span)

        #T_src_pcd.transform(T_sub_action_part_scale)
        #T_sub_pcd.transform(T_sub_action_part_scale)

        #self.scale_Ts.append(T_sub_action_part_scale) # Store scaling factor

        print "SCALING TOOLS"
        # o3d.visualization.draw_geometries([self.T_src_pcd, self.T_sub_pcd])
        # Sub tool AND src tool have been transformed, so make sure to account for both
        # for final sub rotation.
        
        #return T_src_pcd, T_sub_pcd, temp_src_T, T_sub_action_part_scale
        return T_src_pcd, T_sub_pcd, temp_src_T, T_src_to_return, T_sub_to_return

    #def _get_average_distance(pc1, pc2):
        #distance = pc1.compute_point_cloud_distance(pc2)
        #return np.average(distance)    

    #def _get_registration_evaluation(pc1, pc2, T):
        #pc1_copy = deepcopy(pc1)
        #pc2_copy = deepcopy(pc2)
        
        #pc2_copy.transform(T)
        
        #return _get_average_distance(pc1_copy, pc2_copy)

    def get_tool_action_parts(self, T_src_pcd, T_sub_pcd):
        """
        Get the idx associated with the action parts of the src and sub tools.
        """
        self._src_action_segment = self.src_tool.get_action_segment()
        print "SRC ACTION SEGMENT: ", self._src_action_segment

        scaled_src_pc = ToolPointCloud(self._get_src_pnts(T_src_pcd=T_src_pcd), normalize=False)
        scaled_sub_pc = ToolPointCloud(self._get_sub_pnts(T_sub_pcd=T_sub_pcd), normalize=False)

        print "SCALED SUB ALL SEGMENTS: ", scaled_sub_pc.segment_list
        # Get points in segment of src tool containing the contact area
        src_action_part, src_action_indices = scaled_src_pc.get_pnts_in_segment(self._src_action_segment)
        # First, scale both tools to the same size, in order to determine which part of
        # sub tool is the action part.
        # May not need to do this as it has been aligned and scaled?
        #_, scale_f = scaled_sub_pc.scale_pnts_to_target(scaled_src_pc)
        #T_sub_scale = get_scaling_T(scale=scale_f)
        #self.T_sub_pcd.transform(T_sub_scale)
        #self.scale_Ts.append(T_sub_scale)
        #o3d.visualization.draw_geometries([self.T_sub_pcd, self.T_src_pcd], "Aligned")
        #scaled_sub_pc = ToolPointCloud(self._get_sub_pnts(), normalize=False)
        #_, sub_action_pnt_idx = self._get_closest_pnt(src_action_part.mean(axis=0),
                                                      #scaled_sub_pc.pnts)
        self._sub_action_segment, sub_action_part, sub_action_indices = self._find_sub_action_part(src_action_part, self._get_sub_pnts(T_sub_pcd=T_sub_pcd))

        #self._sub_action_segment = scaled_sub_pc.get_segment_from_point(sub_action_pnt_idx)

        #sub_action_part = scaled_sub_pc.get_pnts_in_segment(self._sub_action_segment)

        print "sub_action_part"
        print sub_action_part

        #o3d.visualization.draw_geometries([self.T_src_pcd, self.T_sub_pcd], "Aligned")
        return src_action_part, sub_action_part, src_action_indices, sub_action_indices

    def icp_alignment(self, source, target, correspondence_thresh, n_iter=10, name="ICP", size=None):
        """
        Algin sub tool to src tool using ICP.

         """
        # Scale points to be ~1 meter so that self.voxel_size can
        # be consistent for all pointclouds.

        # o3d.visualization.draw_geometries([self.T_src_pcd, self.T_sub_pcd], "Aligned")
        # src_action_pnts = self.scaled_src_pc.pnt

        if size is None:
            size = self.voxel_size

        source, target, source_down, target_down, source_fpfh, target_fpfh = \
            prepare_dataset(source, target, size)

        # Apply icp n_iter times and get best result
        best_icp = self._icp_wrapper(source_down, target_down,
                                     source_fpfh, target_fpfh,
                                     correspondence_thresh,
                                     n_iter)
        # best_icp = self.nonrigid_registration(source_down, target_down)



        icp_fit = best_icp.fitness # Fit score (between [0.0,1.0])
        icp_trans = best_icp.transformation # T matrix

        # if self.visualize:
        #     print "ICP RESULTS"
        #     visualize_reg(source, target, deepcopy(source).transform(icp_trans), name)

        return icp_trans, icp_fit, source, target

    #def icp_alignment_select_size(self, source, target, correspondence_thresh, n_iter=10, name="ICP"):
        #best_icp_trans = np.identity(4)
        #best_icp_fitness = 0.
        ##best_icp_distance = 100.
        #best_source = deepcopy(source)
        #best_target = deepcopy(target)
        #best_size = 0.
        
        #size = 0.0025
        #for i in range(5):
            #size += 0.0025
            #source_copy = deepcopy(source)
            #target_copy = deepcopy(target)
            #icp_trans, icp_fit, updated_source, updated_target = self.icp_alignment(source_copy,
                                                                                    #target_copy,
                                                                                    #correspondence_thresh,
                                                                                    #n_iter=n_iter,
                                                                                    #name=name,
                                                                                    #size=size)
            ##distance = self._get_registration_evaluation(source_copy, target_copy, icp_trans)
            ##if distance < best_icp_distance:
            #if icp_fit > best_icp_fitness:
                ##best_icp_distance = distance
                #best_icp_trans = deepcopy(icp_trans)
                #best_icp_fitness = icp_fit
                #best_source = deepcopy(updated_source)
                #best_target = deepcopy(updated_target)
                #best_size = size   
        
        #print "[tool_substitution_controller][icp_alignment_select_size] size selected: ", best_size
        
        #return best_icp_trans, best_icp_fitness, best_source, best_target
    
    def icp_alignment_select_size(self, source, target, correspondence_thresh, n_iter=10, name="ICP"):
        best_icp_trans = np.identity(4)
        best_icp_fitness = 0.
        best_icp_distance = 100.
        best_source = deepcopy(source)
        best_target = deepcopy(target)
        best_size = 0.
        
        size = 0.0025
        for i in range(5):
            size += 0.0025
            source_copy = deepcopy(source)
            target_copy = deepcopy(target)
            icp_trans, icp_fit, updated_source, updated_target = self.icp_alignment(source_copy,
                                                                                    target_copy,
                                                                                    correspondence_thresh,
                                                                                    n_iter=n_iter,
                                                                                    name=name,
                                                                                    size=size)
            distance = get_o3d_pc_distance(updated_target, updated_source)
            if distance < best_icp_distance:
                best_icp_distance = distance
                best_icp_trans = deepcopy(icp_trans)
                best_icp_fitness = icp_fit
                best_source = deepcopy(updated_source)
                best_target = deepcopy(updated_target)
                best_size = size   
        
        print "[tool_substitution_controller][icp_alignment_select_size] size selected: ", best_size
        
        return best_icp_trans, best_icp_fitness, best_source, best_target, best_icp_distance   

    def refine_registration(self, init_trans, source, target, voxel_size, name="Reg result"):
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
        sub_T, src_T = pose_graph.nodes[sub_idx].pose, pose_graph.nodes[src_idx].pose
        # This alg rotates both the src and sub tools. We dont want the src tool to rotate 
        # at all so we apply its inverse T onto the sub tool to the same effect.
        final_T = np.matmul(T_inv(src_T), sub_T)
        aligned_source = deepcopy(source).transform(final_T)

        if self.visualize:
            print "INV SRC REG"
            visualize_reg(source, target, aligned_source,
                name=name)      

        return pcds[src_idx], aligned_source, final_T, fit

    #def refine_registration_select_size(self, init_trans, source, target, name="Reg result"):
        #best_pcds = deepcopy(source)
        #best_aligned_source = deepcopy(source)
        #best_final_T = deepcopy(init_trans)
        #best_fit = 0.
        #best_size = 0.
        
        #size = 0.0025
        #for i in range(5):
            #size += 0.0025
            #source_copy = deepcopy(source)
            #target_copy = deepcopy(target)
            #pcds, aligned_source, final_T, fit = self.refine_registration(init_trans,
                                                                          #source_copy,
                                                                          #target_copy,
                                                                          #size,
                                                                          #name=name)
            #if fit > best_fit:
                #best_pcds = deepcopy(pcds)
                #best_aligned_source = deepcopy(aligned_source)
                #best_final_T = deepcopy(final_T)
                #best_fit = fit
                #best_size = size   
        
        #print "[tool_substitution_controller][refine_registration_select_size] size selected: ", best_size
        
        #target = deepcopy(target)
        #target.paint_uniform_color(np.array([1., 0., 0.]))
        #aligned_source = deepcopy(best_aligned_source)
        #aligned_source.paint_uniform_color(np.array([0., 1., 0.]))
        #source = deepcopy(source)
        #source.paint_uniform_color(np.array([0., 1., 0.]))
        #o3d.visualization.draw_geometries([target, source], "initial alignment")
        #o3d.visualization.draw_geometries([target, best_aligned_source], "aliiiiiiiiiiigning!!!, best_fit: " + str(best_fit)) # to delete  
        
        #return best_pcds, best_aligned_source, best_final_T, best_fit
        
    def refine_registration_select_size(self, init_trans, source, target, name="Reg result"):
        best_pcds = deepcopy(source)
        best_icp_distance = 100.
        best_aligned_source = deepcopy(source)
        best_final_T = deepcopy(init_trans)
        best_fit = 0.
        best_size = 0.
        
        size = 0.0025
        for i in range(5):
            size += 0.0025
            source_copy = deepcopy(source)
            target_copy = deepcopy(target)
            pcds, aligned_source, final_T, fit = self.refine_registration(init_trans,
                                                                          source_copy,
                                                                          target_copy,
                                                                          size,
                                                                          name=name)
            distance = get_o3d_pc_distance(target, aligned_source)
            if distance < best_icp_distance:
                best_pcds = deepcopy(pcds)
                best_aligned_source = deepcopy(aligned_source)
                best_final_T = deepcopy(final_T)
                best_fit = fit
                best_size = size 
                best_icp_distance = distance
        
        print "[tool_substitution_controller][refine_registration_select_size] size selected: ", best_size
        
        #TODO: to delete
        #target = deepcopy(target)
        #target.paint_uniform_color(np.array([1., 0., 0.]))
        #aligned_source = deepcopy(best_aligned_source)
        #aligned_source.paint_uniform_color(np.array([0., 1., 0.]))
        #source = deepcopy(source)
        #source.paint_uniform_color(np.array([0., 1., 0.]))
        #o3d.visualization.draw_geometries([target, source], "initial alignment")
        #o3d.visualization.draw_geometries([target, best_aligned_source], "aliiiiiiiiiiigning!!!, best_fit: " + str(best_fit)) # to delete  
        
        return best_pcds, best_aligned_source, best_final_T, best_fit, best_icp_distance 
    
    def _get_sub_contact_indices(self, src_pnts, sub_pnts, src_contact_indices, src_action_indices, sub_action_indices):
        sub_action_part_cp_idx = self._get_contact_surface(src_pnts[src_contact_indices],
                                                           sub_pnts[sub_action_indices],
                                                           src_pnts[src_action_indices])
    
        # Corresponding idx of sub contact surface for original sub pointcloud.
        sub_cp_idx = self.sub_tool.segment_idx_to_idx([self._sub_action_segment],
                                                      sub_action_part_cp_idx)
        
        return sub_cp_idx
    
    def step_0_initial_alignment(self):
        # Scale and orient tools based on their principal axes
        #self._scale_pcs()
        # Find best initial alignment via rotations over these axes.
        # STEP 1: align the source and sub tools
        initial_distance, initial_distance_percentage, T_sub_scale, T_src_pcd, T_sub_pcd, temp_src_T, T_src, T_sub = self._align_pnts(self._get_src_pnts(), self._get_sub_pnts(), keep_proportion=False)
        if self.visualize:
            o3d.visualization.draw_geometries([T_src_pcd, T_sub_pcd], "step 0 initial alignment") 
        
        #total_distance = initial_distance / initial_distance_percentage
        #distance = T_src_pcd.compute_point_cloud_distance(T_sub_pcd)
        #print "min distance: ", np.min(distance)
        #print "min distance(%): ", np.min(distance) / total_distance
        #print "max distance: ", np.max(distance)
        #print "max distance(%): ", np.max(distance) / total_distance
        
        #print "initial_distance: ", initial_distance
        #print "initial_distance_percentage,", initial_distance_percentage
        
        return initial_distance, initial_distance_percentage, T_sub_scale, T_src_pcd, T_sub_pcd, temp_src_T, T_src, T_sub
    
    def step_1_find_action_part(self, src_pcd, sub_pcd):
        # Get the points in the action segments of both tools
        scaled_src_action_pnts, scaled_sub_action_pnts, src_action_indices, sub_action_indices = self.get_tool_action_parts(src_pcd, sub_pcd)
        
        #initial_distance = get_np_pc_distance(src_action_pnts, sub_action_pnts)
        
        if self.visualize:
            source_action_pc = o3d.geometry.PointCloud()
            source_action_pc.points = o3d.utility.Vector3dVector(scaled_src_action_pnts)
            source_action_pc.paint_uniform_color(np.array([0., 1., 0.]))

            sub_action_pc = o3d.geometry.PointCloud()
            sub_action_pc.points = o3d.utility.Vector3dVector(scaled_sub_action_pnts)
            sub_action_pc.paint_uniform_color(np.array([1., 0., 0.]))
            
            o3d.visualization.draw_geometries([source_action_pc, sub_action_pc], "action parts chosen")
        
        return scaled_src_action_pnts, scaled_sub_action_pnts, src_action_indices, sub_action_indices
    
    def step_2_get_initial_alignment_contact_area(self, src_pcd, sub_pcd, src_action_indices, sub_action_indices, T_src, T_sub, T_sub_scale):
        sub_contact_point_idx = self._get_sub_contact_indices(np.asarray(src_pcd.points), np.asarray(sub_pcd.points), np.asarray(self.src_tool.contact_pnt_idx), src_action_indices, sub_action_indices)
        
        return (sub_contact_point_idx, deepcopy(T_src), deepcopy(T_sub), deepcopy(T_sub_scale), deepcopy(sub_pcd)) # when revert this, first unscale, and then unrotate        
    
    # scale the sub tool based on the relative size between the source action part and the sub action part
    def step_3_scale_sub_tool(self, src_pcd, sub_pcd, src_action_index, sub_action_index):
        src_action_part = deepcopy(np.array(src_pcd.points)[src_action_index])
        sub_action_part = deepcopy(np.array(sub_pcd.points)[sub_action_index])

        src_action_part_bb = ToolPointCloud(src_action_part, normalize=False).bb
        src_action_part_bb._calculate_axis()
        src_action_part_norm = src_action_part_bb.norms
        
        sub_action_part_bb = ToolPointCloud(sub_action_part, normalize=False).bb
        sub_action_part_bb._calculate_axis()
        sub_action_part_norm = sub_action_part_bb.norms
        
        scale = np.max(src_action_part_norm) / np.max(sub_action_part_norm)
        permed_scale_f = np.array([scale, scale, scale])
        T_sub_action_part_scale = get_scaling_T(scale=permed_scale_f)
        
        scaled_sub_pcd = deepcopy(sub_pcd)
        scaled_sub_pcd.transform(T_sub_action_part_scale)
        
        return scaled_sub_pcd, T_sub_action_part_scale
    
    def step_4_register_action_parts(self, src_pcd, sub_pcd):
        copy_src_pcd = deepcopy(src_pcd)
        copy_sub_pcd = deepcopy(sub_pcd)
        
        aligned_set, min_transformations, min_threshold = align_pcd_select_size([copy_src_pcd, copy_sub_pcd])
        
        if self.visualize:
            o3d.visualization.draw_geometries(aligned_set, "align action parts")
        
        return aligned_set[1], min_transformations[1], min_threshold
        
        #icp_trans, fit, sub_action_pcd, src_action_pcd = self.icp_alignment(sub_action_pnts,
                                            #src_action_pnts,
                                            #self.correspondence_thresh,
                                            #n_iter)                                                                                        
    
    
        #print "ORGINAL ALIGN FITNESS: ", align_fit
        #print "INIT ICP FITNESS:      ", fit
        #fit_ratio = fit / align_fit
    
        #if fit_ratio > self.fit_ratio_thresh:
            #print "USING INIT. ICP RESULTS.."
            #T_align = icp_trans
        #else:
            #print "SKIPPING  INIT. ICP RESULTS, USING INIT. ALIGNMENT."
            #T_align = get_T_from_R_p()
    
        #T_align = icp_trans
        #initial_T_align = get_T_from_R_p()
        #if initial_distance < distance:
            #T_align = initial_T_align
    
        #print "Refining registration..."
        ## Refine initial icp alignment.
        #_, aligned_sub_pcd, T_align, best_fit, best_distance = self.refine_registration_select_size(T_align,
                                                                                                        #sub_action_pcd,
    
                                                                                      #src_action_pcd,
                                                                                      #name="Aligned action parts")  
    
    def step_5_get_aligned_contact_area(self, src_pcd, sub_pcd, src_action_indices, sub_action_indices, T_src, T_sub, T_sub_scale):
        sub_contact_point_idx = self._get_sub_contact_indices(np.asarray(src_pcd.points), np.asarray(sub_pcd.points), np.asarray(self.src_tool.contact_pnt_idx), src_action_indices, sub_action_indices)
        
        return (sub_contact_point_idx, deepcopy(T_src), deepcopy(T_sub), deepcopy(T_sub_scale), deepcopy(sub_pcd))        

    def step_6_choose_contact_area(self, contact_area_1, contact_area_2, sub_action_indices):
        sub_pcd = deepcopy(self.sub_pcd)
        
        contact_area_1_pcd = deepcopy(self._np_to_o3d(np.asarray(sub_pcd.points)[contact_area_1[0]]))
        contact_area_2_pcd = deepcopy(self._np_to_o3d(np.asarray(sub_pcd.points)[contact_area_2[0]]))
        
        aligned_set, min_transformations, min_threshold = align_pcd_select_size([contact_area_1_pcd, contact_area_2_pcd])
        
        sub_action_part = deepcopy(np.array(self.sub_pcd.points)[sub_action_indices])
        sub_action_part_bb = ToolPointCloud(sub_action_part, normalize=False).bb
        sub_action_part_bb._calculate_axis()
        sub_action_part_norm = sub_action_part_bb.norms 
        
        dislikeness = min_threshold / max(sub_action_part_norm) # the high the value, the more dislike the two contact areas are
        
        contact_area = None
        if dislikeness > 0.05: # tune this value
            contact_area = contact_area_2
        else:
            contact_area = contact_area_1
        
        return contact_area

    def step_7_align_tools(self, contact_area):
        sub_contact_point_idx = deepcopy(contact_area[0])
        T_src                 = deepcopy(contact_area[1])
        T_sub                 = deepcopy(contact_area[2])
        T_sub_scale           = deepcopy(contact_area[3])
        sub_pcd               = deepcopy(contact_area[4])
        
        src_contact_area = np.asarray(self.src_pcd.points)[self.src_tool.self.src_tool.contact_pnt_idx]
        src_contact_area_center = np.mean(src_contact_area, axis=0)
        
        inv_T_sub_scale = T_inv(T_sub_scale)
        sub_pcd.transform(inv_T_sub_scale)
        
        # align the contact areas
        sub_contact_area = np.asarray(sub_pcd)[sub_contact_point_idx]
        sub_contact_area_center = np.mean(sub_contact_area, axis=0)
        sub_pcd.translate(src_contact_area_center - sub_contact_area_center)
        
        aligned_set, min_transformations, min_threshold = align_pcd_select_size([sub_pcd, self.sub_pcd])
        
        T_sub = min_transformations[1]
        
        Tsrc_sub = get_homogenous_transformation_matrix_inverse(T_src, T_sub)
        
        return Tsrc_sub

    def get_T_cp(self, n_iter=10):
        """
        Refine Initial ICP alignment and return final T rotation matrix and sub contact pnts.
        """
        step_0_results = self.step_0_initial_alignment()
        
        step_0_initial_distance            = step_0_results[0]
        step_0_initial_distance_percentage = step_0_results[1]
        step_0_T_sub_scale                 = step_0_results[2]
        step_0_src_pcd                     = step_0_results[3]
        step_0_sub_pcd                     = step_0_results[4]
        step_0_temp_src_T                  = step_0_results[5]
        step_0_T_src                       = step_0_results[6] # centered
        step_0_T_sub                       = step_0_results[7] # scaled and re-oriented
        # the previous scaling or transformations can be ignored, as the whole was to find the action segment

        step_1_results = self.step_1_find_action_part(step_0_src_pcd, step_0_sub_pcd)
        
        step_1_scaled_src_action_pnts = step_1_results[0]
        step_1_scaled_sub_action_pnts = step_1_results[1]
        step_1_src_action_indices     = step_1_results[3]
        step_1_sub_action_indices     = step_1_results(4)
        #src action segment: self._src_action_segment
        #sub action segment: self._sub_action_segment
        #src contact area index: self.src_tool.contact_pnt_idx
        
        # the contact area of based on the initial alignment
        step_2_results = self.step_2_get_initial_alignment_contact_area(step_0_src_pcd,
                                                                        step_0_sub_pcd, 
                                                                        step_1_src_action_indices, 
                                                                        step_1_sub_action_indices, 
                                                                        step_0_T_src, 
                                                                        step_0_T_sub, 
                                                                        step_0_T_sub_scale)
        contact_area_1 = step_2_results # when revert this, first unscale, and then unrotate  
        
        # scale the two tools based on the action part size
        step_3_results = self.step_3_scale_sub_tool(self.src_pcd, self.sub_pcd, step_1_src_action_indices, step_1_sub_action_indices)
        
        step_3_scaled_sub_pcd          = step_3_results[0]
        step_3_T_sub_action_part_scale = step_3_results[1] # scale appeared first, so for the contact area found with this method, first unrotate, and then unscale
        
        # use ICP to align the action part to find the contact area
        step_4_results = self.step_4_register_action_parts(self.src_pcd, step_3_scaled_sub_pcd)
        
        step_4_scaled_aligned_sub_action_pcd = step_4_results[0]
        step_4_T_sub_pcd                     = step_4_results[1]
        step_4_threshold                     = step_4_results[2]
        
        # find the corresponding contact area
        step_5_results = self.step_5_get_aligned_contact_area(self.src_pcd,
                                                              step_4_scaled_aligned_sub_action_pcd,
                                                              step_1_src_action_indices, 
                                                              step_1_sub_action_indices,                                                              
                                                              np.identity(4),
                                                              np.matmul(step_4_T_sub_pcd, step_3_T_sub_action_part_scale),
                                                              step_3_T_sub_action_part_scale)
        contact_area_2 = step_5_results
        
        # choose the contact area
        step_6_results = self.step_6_choose_contact_area(contact_area_1, contact_area_2, step_1_sub_action_indices)
        contact_area = step_6_results
        
        # descale and align the pc based on the contact area chosen
        Tsrc_sub = self.step_7_align_tools(contact_area)
        
        return Tsrc_sub, contact_area[0]
        
        # Scale action parts to same size for better comparison. (undistortional scale)
        #src_action_pnts = self._get_src_pnts()[src_action_indices]
        #src_action_pnts = self._get_src_pnts()[src_action_indices]
        #sub_action_pnts = self._get_sub_pnts()[sub_action_indices]

        #print "src_action_pnts"
        #print src_action_pnts
        #print "sub_action_pnts"
        #print sub_action_pnts

        #initial_distance, initial_distance_percentage, T_sub_action_part_scale, T_src_action_pcd, T_sub_action_pcd, temp_src_T, T_src_action, T_sub_action = self._align_pnts(src_action_pnts, sub_action_pnts, keep_proportion=True)
  
        #o3d.visualization.draw_geometries([T_src_action_pcd, T_sub_action_pcd], "action parts alignment")
        
        #self.temp_src_T = temp_src_T
        #self.T_src_pcd.transform(T_src)
        #self.T_sub_pcd.transform(T_sub)
        ##self.T_sub_pcd.transform(T_sub_action_part_scale)
        #self.scale_Ts.append(T_sub_action_part_scale)
        
        #o3d.visualization.draw_geometries([self.T_src_pcd, self.T_sub_pcd], "tool alignment based on action part")
        
        ## TODO: save those transformations
        
        ##print "SCALED SUB SEGMENTS: ", scaled_sub_pc.segment_list
        ##_, scale_f = ToolPointCloud(sub_action_part,
                                    ##normalize=False).scale_pnts_to_target(ToolPointCloud(src_action_part,
                                                                                         ##normalize=False))
        ##T_sub_action_part_scale = get_scaling_T(scale=scale_f)
        
        ##self.T_sub_pcd.transform(T_sub_action_part_scale)
        ##self.scale_Ts.append(T_sub_action_part_scale)    

               

        #raise Exception("stop")


        ##icp_trans, fit, sub_action_pcd, src_action_pcd = self.icp_alignment_select_size(sub_action_pnts,
                                                                                        ##src_action_pnts,
                                                                                        ##self.correspondence_thresh,
                                                                                        ##n_iter)
        #icp_trans, fit, sub_action_pcd, src_action_pcd, distance = self.icp_alignment_select_size(sub_action_pnts,
                                                                                                  #src_action_pnts,
                                                                                                  #self.correspondence_thresh,
                                                                                                  #n_iter)
        ##icp_trans, fit, sub_action_pcd, src_action_pcd = self.icp_alignment(sub_action_pnts,
                                                                            ##src_action_pnts,
                                                                            ##self.correspondence_thresh,
                                                                            ##n_iter)                                                                                        


        ##print "ORGINAL ALIGN FITNESS: ", align_fit
        ##print "INIT ICP FITNESS:      ", fit
        ##fit_ratio = fit / align_fit

        ##if fit_ratio > self.fit_ratio_thresh:
            ##print "USING INIT. ICP RESULTS.."
            ##T_align = icp_trans
        ##else:
            ##print "SKIPPING  INIT. ICP RESULTS, USING INIT. ALIGNMENT."
            ##T_align = get_T_from_R_p()
        
        #T_align = icp_trans
        #initial_T_align = get_T_from_R_p()
        #if initial_distance < distance:
            #T_align = initial_T_align
        
        #print "Refining registration..."
        ## Refine initial icp alignment.
        #_, aligned_sub_pcd, T_align, best_fit, best_distance = self.refine_registration_select_size(T_align,
                                                                                                    #sub_action_pcd,
                                                                                                    #src_action_pcd,
                                                                                                    #name="Aligned action parts")        
        
        ##_, aligned_sub_pcd, T_align, best_fit = self.refine_registration_select_size(T_align,
                                                                             ##sub_action_pcd,
                                                                             ##src_action_pcd,
                                                                             ##name="Aligned action parts")
        ##_, aligned_sub_pcd, T_align, fit = self.refine_registration(T_align,
                                                                 ##sub_action_pcd,
                                                                 ##src_action_pcd,
                                                                 ##self.voxel_size,
                                                                 ##name="Aligned action parts")

        ##print "????????? fit score is: ", best_fit
        #print "######################################################################"
        #if initial_distance < best_distance:
            #print "use original alignement"
            #T_align = initial_T_align
        #else:
            #print "use new alignment"
        
        #print "\tinitial_average_distance: ", initial_distance
        #print "\talignment average distance: ", best_distance
        #print "######################################################################"
        ##raise Exception("stop")

        #if self.visualize:
            #print "TEST ALIGNMENT"
            #o3d.visualization.draw_geometries([src_action_pcd, # duplicate ??
                                               #src_action_pcd,
                                               #self.T_src_pcd],
                                               #"Testing sec action part alignment")

        #self.T_sub_pcd.transform(T_align)
        ## aligned_sub_pcd = sub_action_pcd.transform(T_align)

        ## if self.visualize:
        ##     print "sub action part and full tool alignment"
        ##     o3d.visualization.draw_geometries([self.T_sub_pcd, aligned_sub_pcd])

        ## if self.visualize:
        ##     print "src action part and full tool alignment"
        ##     o3d.visualization.draw_geometries([src_action_pcd, self.T_src_pcd])

        #aligned_src = np.asarray(src_action_pcd.points)
        #aligned_sub = np.asarray(aligned_sub_pcd.points)

        ## visualize_two_pcs(aligned_sub, aligned_src)
        #src_action_part_cp_idx = self.src_tool.idx_to_segment_idx(self.src_tool.contact_pnt_idx)
        #src_contact_pnt = aligned_src[src_action_part_cp_idx, :]
        ## src_contact_pnt = self._get_src_pnts(get_segments=False)[src_action_part_cp_idx, :]
        ## src_contact_pnt = self.src_tool.pnts[src_action_part_cp_idx, :]
        ## Reshape point for easy computation
        #src_contact_pnt = src_contact_pnt.reshape(1,-1) \
            #if len(src_contact_pnt.shape) == 1 else src_contact_pnt
        
        #if self.visualize:
            #src_contact_o3d_pc = o3d.geometry.PointCloud()
            #src_contact_o3d_pc.points = o3d.utility.Vector3dVector(src_contact_pnt)
            #src_contact_o3d_pc.paint_uniform_color(np.array([1., 0., 0.]))
            #o3d.visualization.draw_geometries([src_contact_o3d_pc], "src_contact_o3d_pc")

        ## Estimate contact surface on the sub tool
        #sub_action_part_cp_idx = self._get_contact_surface(src_contact_pnt,
                                                    #aligned_sub,
                                                    #aligned_src)

        ## Corresponding idx of sub contact surface for original sub pointcloud.
        #sub_cp_idx = self.sub_tool.segment_idx_to_idx([self._sub_action_segment],
                                                      #sub_action_part_cp_idx)

        #self.sub_tool.contact_pnt_idx = sub_cp_idx
        #print "SUB CP IDX ", sub_cp_idx

        #mean_sub_action_part_cp = np.mean(aligned_sub[sub_action_part_cp_idx, :],axis=0)
        ## mean_sub_action_part_cp = np.mean(src_contact_pnt)
        #scale_T = T_inv(np.linalg.multi_dot(self.scale_Ts)) # Get inverted scaling T
        #scale_f = np.diag(scale_T)[:3] # get first 3 diag entries.
        #final_scale_T = get_scaling_T(scale_f, mean_sub_action_part_cp)

        #if self.visualize:
            #print "Visualizing sub contact point"
            #visualize_tool(aligned_sub_pcd, sub_action_part_cp_idx, 
                #name="Transformed sub tool with calc'd contact surface")
            #print "visualizing src contact point"
            #visualize_tool(self.T_sub_pcd, sub_cp_idx,
                #name="Transformed src tool with calc'd contact surface")


        #descaled_aligned_sub = deepcopy(self.T_sub_pcd).transform(final_scale_T)
        ## descaled_aligned_sub = deepcopy(aligned_sub_pcd).transform(scale_T)
        ## descaled_sub_action = deepcopy(aligned_sub_pcd).transform(final_scale_T)
        #if self.visualize:
            #print "DESCALING"
            #visualize_reg(aligned_sub_pcd, self.sub_pcd, descaled_aligned_sub, name="descaling")

        #self.T_sub_pcd.transform(final_scale_T)
        ## descaled_sub_mean_cp = np.mean(np.asarray(descaled_aligned_sub.points)[sub_cp_idx, :], axis=0)
        ## descaled_sub_mean_cp = np.mean(np.asarray(descaled_aligned_sub.points)[sub_action_part_cp_idx, :], axis=0)
        #if self.visualize:
            #print "VISUALIZING DESCALE BEFORE TRANS"
            #visualize_reg(aligned_sub_pcd, src_action_pcd, descaled_aligned_sub, name="visualizing descale before trans")

        ## T_translate = get_T_from_R_p(p=np.mean(src_contact_pnt,axis=0)-descaled_sub_mean_cp)
        ## descaled_aligned_sub.transform(T_translate)
        ## self.T_sub_pcd.transform(T_translate)

        ## print "VISUALIZING DESCALE AFTER TRANS"
        ## visualize_reg(aligned_sub_pcd, src_action_pcd, descaled_aligned_sub)


        ## print "HELOO"
        ## visualize_reg(descaled_aligned_sub, self.src_pcd,
        ##               deepcopy(descaled_aligned_sub).transform(self.temp_src_T))
        #descaled_aligned_sub.transform(self.temp_src_T)
        ## This transformation accounts for the alignment of the src tool to
        ## its principal axes
        #self.T_sub_pcd.transform(self.temp_src_T)

        #descaled_sub_pnts = np.asarray(self._get_sub_pnts(get_segments=False))
        #original_sub_pnts = np.asarray(self.sub_pcd.points)

        #new_vox_size = self.voxel_size
        #new_corr_thresh = self.correspondence_thresh

        ##icp_T, icp_fit, _, _= self.icp_alignment(original_sub_pnts,
                                                 ##descaled_sub_pnts,
                                                 ##new_corr_thresh,
                                                 ##n_iter=n_iter,
                                                 ##)        

        #icp_T, icp_fit, _, _, icp_distance = self.icp_alignment_select_size(original_sub_pnts,
                                                                            #descaled_sub_pnts,
                                                                            #new_corr_thresh,
                                                                            #n_iter=n_iter,
                                                                            #)

        ##aligned_sub_pcd,_, refine_T, refine_fit = self.refine_registration(icp_T,
                                                                           ##self.sub_pcd,
                                                                           ### descaled_aligned_sub,
                                                                           ##self.T_sub_pcd,
                                                                           ##self.voxel_size,
                                                                           ##name="Aligning sub tool in init pos to transformed sub tool")

        #aligned_sub_pcd, _, refine_T, refine_fit, refine_distance = self.refine_registration_select_size(icp_T,
                                                                                                        #self.sub_pcd,
                                                                                                        ## descaled_aligned_sub,
                                                                                                        #self.T_sub_pcd,
                                                                                                        #name="Aligning sub tool in init pos to transformed sub tool")


        ##print "ICP fit: ", icp_fit
        ##print "Refine fit: ", refine_fit
        
        #print "######################################################################"
        #print "ICP distance: ", icp_distance
        #print "Refine distance: ", refine_distance
        #print "######################################################################"

        ##final_trans = icp_T if icp_fit > refine_fit else refine_T
        #final_trans = icp_T if icp_distance < refine_distance else refine_T
        ## self.T_sub_pcd.transform(final_trans)

        ## Algin the centroids of contact areas of the tools.
        #src_mean_cp = np.mean(np.asarray(self.src_pcd.points)[self.src_tool.contact_pnt_idx,:], axis=0)
        #sub_mean_cp = np.mean(np.asarray(self.T_sub_pcd.points)[sub_cp_idx, :], axis=0)
        #T_translate = get_T_from_R_p(p=(src_mean_cp-sub_mean_cp))
        #final_trans = np.matmul(T_translate, final_trans)


        #if self.visualize:
            #print "FINAL RESULT"
            #visualize_reg(self.sub_pcd,
                          #self.src_pcd,
                          #deepcopy(self.sub_pcd).transform(final_trans),
                          #result_cp_idx=sub_cp_idx,
                          #target_cp_idx=self.src_tool.contact_pnt_idx,
                          #name="Final result applied to original sub tool.")


        ## Must account for the fact that the src and sub tool pcs have been centered at
        ## 0, so we create one final translation transformation
        #orig_sub_tool = self._np_to_o3d(self.sub_tool.get_unnormalized_pc())
        #orig_src_tool = self._np_to_o3d(self.src_tool.get_unnormalized_pc())
        #orig_sub_tool.transform(final_trans)

        #src_mean_cp = np.mean(np.asarray(orig_src_tool.points)[self.src_tool.contact_pnt_idx,:], axis=0)
        #sub_mean_cp = np.mean(np.asarray(orig_sub_tool.points)[sub_cp_idx, :], axis=0)
        #T_translate = get_T_from_R_p(p=(src_mean_cp-sub_mean_cp))
        #final_trans = np.matmul(T_translate, final_trans)

        #return final_trans, sub_cp_idx

    def self_substitute(self):
        """
        Given a calculated contact point on a substitute tool, we want to find other viable potential contact surfaces.
        https://stats.stackexchange.com/questions/14673/measures-of-similarity-or-distance-between-two-covariance-matrices
        """

        sub_cp = None
        N_pnts = 10

        if self.sub_tool.contact_pnt_idx is None:
            T, cp = self.get_T_cp(n_iter=1)
            sub_cp = cp

        sub_pnts = self.sub_tool.pnts
        cp_cov = np.cov(sub_cp.T) # Get covariance matrix of orig contact area.

        # Get N_pnts random pnts from the sub tool to be the centroid of new contact
        # surface.
        rand_idx = np.random.randint(0, self.sub_tool.pnts.shape[0], size=N_pnts)
        scores = []
        for i in rand_idx:
            rand_pnt = sub_pnts[i, :]
            mdist = cdist(sub_pnts, [rand_pnt], metric='mahalanobis', V=cp_cov)[:,0]

            new_cp_idx = mdist < 1 # All points < 1 stdev from rand_pnt
            new_cp_cov = np.cov(sub_pnts[new_cp_idx, :].T)
            # Correlation matrix distance to determine how similar the covs are.
            score = 1 - (np.trace(cp_cov.dot(new_cp_cov)) / (np.dot(norm(cp_cov, ord='fro'),
                                                                    norm(new_cp_cov, ord='fro'))))
            print "SCORE: ", score
            scores.append((new_cp_idx, score))

        # best_cp_idx = cp_idx_list[np.argmin(scores)]
        best_cp_idx, score = min(scores, key=lambda s: s[1])
        cp_idx_list = [idx for idx,_ in scores]

        if self.visualize:
            visualize_multiple_cps(self.sub_pcd, self.sub_tool.contact_pnt_idx, cp_idx_list)
            visualize_multiple_cps(self.sub_pcd, self.sub_tool.contact_pnt_idx, [best_cp_idx])

        return best_cp_idx

    def segment_tool_nonrigid(self):
        """
        Use non-rigid icp in order to determine contact surface on src tool.
        """
        self._scale_pcs()
        source, target, source_down, target_down, source_fpfh, target_fpfh = \
            prepare_dataset(self.sub_tool.pnts , self.src_tool.pnts, 0.005)
        # Get registered sub tool
        sub_result  = self.nonrigid_registration(target_down, source_down)
        print "source normals: ", np.asarray(source_down.normals)
        sub_result.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size*2, max_nn=50))
        print "new normals: ", np.asarray(sub_result.normals)
        # contact pnt of src tool.
        src_cp = self.src_tool.pnts[self.src_tool.contact_pnt_idx, :]
        # Estimates contact surface on downasampled sub tool.
        sub_downsamp_cp_idx = self._get_contact_surface(src_cp,
                                                        np.asarray(sub_result.points),
                                                        np.asarray(target_down))


        # Want to get the contact surface but on the full, upscaled pointcloud.
        # We do this by finding the points in a .005cm ball around above contact surface
        # superimposed on the upscaled pointcloud.
        tree = cKDTree(self.sub_tool.pnts)
        idx_list = tree.query_ball_point(np.asarray(source_down.points)[sub_downsamp_cp_idx, :],
                                         .005)
        idx = [i for l in idx_list for i in l]

        if self.visualize:
            visualize_tool(source_down, sub_downsamp_cp_idx)
            visualize_tool(source, idx)

        return idx


    def show_sub_grip_point(self):
        """
        Visualizes the grip point on the sub tool.
        """

        # Calculate the sub tool cp if it hasnt been calculated already.
        if self.sub_tool.contact_pnt_idx is None:
            T, cp = self.get_T_cp(n_iter=1)
            sub_cp = cp

        sub_action_seg = self.sub_tool.get_action_segment()

        cp_mean = np.mean(sub_cp, axis=0)
        other_segs = [s for s in self.sub_tool.segment_list if not s is sub_action_seg]


        # Determine the most distant segment from the action segment to be used as
        # grasp segment.
        scores = []
        for seg in other_segs:
            seg_pnts, _ = self.sub_tool.get_pnts_in_segment(seg)
            seg_mean = np.mean(seg_pnts, axis=0)

            scores.append((seg, seg_mean, norm(seg_mean-cp_mean)))

        grip_seg, grip_mean, _ = max(scores, key=lambda s: s[2])
        self.mahalanobis_thresh = .5 # This keeps the size of estimated grasp surface to 
                                     # to reasonable level.
        grip_surface_idx = self._get_contact_surface(grip_mean,
                                                     self.sub_tool.pnts,
                                                     self.sub_tool.pnts)

        # print "SUB CONTACT PNT"
        # visualize_tool(self.sub_pcd, self.sub_tool.contact_pnt_idx)
        print "SUB grip PNT"
        visualize_tool(self.sub_pcd, grip_surface_idx)

    def main(self):
        return self.get_T_cp(n_iter=7)
        # return self.self_substitute()
        # self.segment_tool_nonrigid()
        # self.show_sub_grip_point()
        # return self.get_random_contact_pnt()
if __name__ == '__main__':
    gp = GeneratePointcloud()
    n = 8000
    get_color = True

    # pnts1 = gp.get_random_pointcloud(n)
    # pnts2 = gp.get_random_pointcloud(n)
    # pnts2 = gp.load_pointcloud("../../tool_files/rake.ply", get_segments=False)
    # pnts1 = gp.load_pointcloud("../../tool_files/rake.ply", get_segments=True)
    #pnts1 = gp.load_pointcloud("../../tool_files/point_clouds/b_wildo_bowl.ply", get_segments=False)
    #pnts2 = gp.load_pointcloud("../../tool_files/point_clouds/a_bowl.ply", get_segments=False)
    pnts1 = gp.load_pointcloud("../../../meiying_crow_tool/pointcloud/tools/red_mallet.ply", gen_segments=True)
    pnts2 = gp.load_pointcloud("../../../meiying_crow_tool/pointcloud/tools/red_brush.ply", gen_segments=True)

    #pnts1 = np.random.uniform(0, 1, size=(n, 3))
    #pnts2 = np.random.uniform(1.4, 2, size=(n, 3))
    # 
    # pnts2 = gp.load_pointcloud("../../tool_files/point_clouds/a_knifekitchen2.ply",get_segments=True)
    # pnts2 = gp.load_pointcloud("../../tool_files/point_clouds/a_chineseknife.ply", get_segments=True)
    # # pnts2 = gp.load_pointcloud("./tool_files/point_clouds/a_bowlchild.ply", None)
    # pnts2 = gp.mesh_to_pointcloud("/rake_remove_box/2/rake_remove_box_out_2_40_fused.ply", n)
    # pnts1 = gp.load_pointcloud('./tool_files/point_clouds/a_bowl.ply', None)
    # pnts1 = gp.mesh_to_pointcloud('a_knifekitchen2/2/a_knifekitchen2_out_4_60_fused.ply',n , get_color)

    src = ToolPointCloud(pnts1, contact_pnt_idx=None, normalize=True)

    # src.visualize_bb()
    sub = ToolPointCloud(pnts2, normalize=True)
    # src = sub
    # sub = shrink_pc(sub)

    cntct_pnt = src.get_pc_bb_axis_frame_centered().argmin(axis=0)[0]
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

    visualize_reg(sub_pcd, src_pcd, sub_pcd.transform(T), name="Final Result on uncentered PCS")
