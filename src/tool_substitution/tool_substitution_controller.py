#!/usr/bin/env python
import numpy as np
from numpy import cross
from numpy.linalg import norm

from itertools import permutations

import open3d as o3d
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree, KDTree

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
from matplotlib.patches import Rectangle

from tool_pointcloud import ToolPointCloud
from sample_pointcloud import GeneratePointcloud
from util import (min_point_distance, rotation_matrix_from_vectors,
                  weighted_min_point_distance, visualize_two_pcs,
                  rotation_matrix_from_box_rots, visualize_vectors,
                  r_x, r_y, r_z)

from scipy.spatial.transform import Rotation as Rot

from get_target_tool_pose import get_T_from_R_p, get_pnts_world_frame, get_aruco_world_frame
from pointcloud_registration import prepare_dataset, draw_registration_result


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
    def __init__(self, src_tool_pc, sub_tool_pc, voxel_size=0.02, visualize=False):
        """
        Class for aligning substitute tool to source tool based on a given contact surface
        of the source tool.
        """
        # ToolPointClouds for src and sub tools
        self.src_tool = src_tool_pc
        self.sub_tool = sub_tool_pc

        # Open3d pointcloud of src and sub tool.   
        self.src_pcd = self._tpc_to_o3d(self.src_tool)
        self.sub_pcd = self._tpc_to_o3d(self.sub_tool)
        # Params for ICP
        self.voxel_size = voxel_size 
        self.correspondence_thresh = voxel_size * .7
        # 
        self.mahalanobis_thresh = 1.

        self.visualize = visualize
        self.Ts =[] # Tracks all transformations of sub tool.


    def _icp_wrapper(self, src,sub, src_fpfh, sub_fpfh, n_iter=5):
        """
        Wrapper function for configuring and running icp registration using feature mapping.
        """
        checker = [
            o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(self.correspondence_thresh),
            o3d.registration.CorrespondenceCheckerBasedOnNormal(np.pi/2)
                   ]


        RANSAC = o3d.registration.registration_ransac_based_on_feature_matching

        est_ptp = o3d.registration.TransformationEstimationPointToPoint()
        est_ptpln = o3d.registration.TransformationEstimationPointToPlane()

        criteria = o3d.registration.RANSACConvergenceCriteria(max_iteration=400000,
                                                              max_validation=1000)

        results = []

        # Apply icp n_iter times and get best result
        for i in range(n_iter):
            result = RANSAC(src, sub, src_fpfh, sub_fpfh,
                            max_correspondence_distance=self.correspondence_thresh,
                            estimation_method=est_ptp,
                            ransac_n=4,
                            checkers=checker,
                            criteria=criteria)

            results.append(result)
            if result.fitness == 1.0: break # 1.0 is best possible score.

        return  max(results, key=lambda i:i.fitness)

    @staticmethod
    def _tpc_to_o3d(tpc):
        """
        Get o3d pc from ToolPointcloud object.
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(tpc.pnts)

        return pcd

    def _center_and_align_pnts(self, pc):
        """
        Creates a centered and aligned ToolPointCloud from unaligned ToolPointCloud
        """
        pnts = pc.get_pc_bb_axis_frame_centered()
        # Add the segment labels back in.
        pnts = np.vstack([pnts.T, pc.segments]).T

        return ToolPointCloud(pnts)

    def _calc_best_orientation(self, src_pnts, sub_pnts, Rs):
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
            sub_pnts_rot = R.dot(sub_pnts.T).T
            sub_o3d = o3d.geometry.PointCloud()
            src_o3d = o3d.geometry.PointCloud()

            sub_o3d.points = o3d.utility.Vector3dVector(sub_pnts_rot)
            src_o3d.points = o3d.utility.Vector3dVector(src_pnts)
            # How well do the points align?
            dist = o3d.registration.evaluate_registration(src_o3d, sub_o3d,
                                                          self.correspondence_thresh)
            # score = np.asarray(dist).mean()
            score = dist.fitness

            print "ALIGNMENT SCORE: ", score

            scores.append(score)
            aligned_pnts.append(sub_pnts_rot)

        i = np.argmax(scores) # Higher the score the better.

        print "rotation {} is best".format(i + 1)
        return Rs[i], aligned_pnts[i], scores[i]

    def _get_contact_surface(self, src_cps, sub_pnts, src_pnts):
        """
        @src_cps: (nx3) ndarray contact surface of src tool.
        @sub_pnts: (mx3) ndarray points in sub tool pc. 

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


        est_sub_cp = self._get_closest_pnt(cp_mean, sub_pnts)
        # Get points arounnd est_sub_cp with similar distribution as src_cps.
        mdist = cdist(sub_pnts, [est_sub_cp], metric='mahalanobis', V=cov)[:,0]


        return mdist < self.mahalanobis_thresh




    def _get_closest_pnt(self, pnt, pntcloud):
        """
        returns the point in pntcloud closest to pnt.
        """
        tree = cKDTree(pntcloud )
        _, i = tree.query(pnt)

        return pntcloud[i,:]

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
        _, scale_f = self.scaled_sub_pc.scale_pnts_to_target(self.scaled_src_pc)
        for p in permutations([0,1,2]):
            permed_scale_f = scale_f[list(p)]
            scaled_sub_pnts = self.scaled_sub_pc.pnts * permed_scale_f


            R, rot_sub_pnts, score = self._calc_best_orientation(self.scaled_src_pc.pnts,
                                                                scaled_sub_pnts,
                                                                [R1, R2, R3, R4])

            scores.append((R, rot_sub_pnts, score))

        return max(scores, key=lambda s: s[2])




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
        sub_contact_pnt_idx = self._get_closest_pnt(src_cntct_pnt,
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

        self.centered_src_pc = self._center_and_align_pnts(self.src_tool)
        self.centered_sub_pc = self._center_and_align_pnts(self.sub_tool)

        # Get the lengths of the bounding box sides for each pc  
        src_norm = self.centered_src_pc.bb.norms[0]
        sub_norm = self.centered_sub_pc.bb.norms[0]

        # Scale by 1 / longest_side to ensure longest side is no longer than 1m.
        largest_span = src_norm if src_norm > sub_norm else sub_norm

        scaled_src_pnts = self.centered_src_pc.pnts *  1.0 / largest_span
        scaled_sub_pnts = self.centered_sub_pc.pnts *  1.0 / largest_span

        # Sub tool AND src tool have been transformed, so make sure to account for both   
        # for final sub rotation.
        T_align = get_T_from_R_p(np.zeros((1,3)), self.sub_tool.get_axis())
        T_inv   = get_T_from_R_p(np.zeros((1,3)), np.linalg.inv(self.src_tool.get_axis()))

        self.Ts.append(T_align) # To account fot alignment of sub tool along bb axis.
        self.Ts.append(T_inv) # To account for alignment of src tool along bb axis.

        self.scaled_src_pc = ToolPointCloud(scaled_src_pnts)
        self.scaled_sub_pc = ToolPointCloud(scaled_sub_pnts)


    def icp_alignment(self, n_iter=10):
        """
        Algin sub tool to src tool using ICP.
        """
        # Scale points to be ~1 meter so that self.voxel_size can
        # be consistent for all pointclouds.
        self._scale_pcs()

        # Find best initial alignment.
        init_R, sub_pnts, align_fit = self._align_pnts()
        T_init = get_T_from_R_p(np.zeros(3).reshape(1,-1), init_R)

        src_pnts = self.scaled_src_pc.pnts

        source, substitute, source_down, substitute_down, source_fpfh, substitute_fpfh = \
            prepare_dataset(src_pnts, sub_pnts, self.voxel_size)

        # Apply icp n_iter times and get best result
        best_icp = self._icp_wrapper(source_down, substitute_down,
                                     source_fpfh, substitute_fpfh,
                                     n_iter)


        icp_fit = best_icp.fitness # Fit score (between [0.0,1.0])
        icp_trans = best_icp.transformation # T matrix

        if align_fit < icp_fit:
            trans = icp_trans
            fit   = icp_fit
        else:
            trans = T_init
            fit   = align_fit
            self.Ts.append(T_init)

        return trans, fit, source, substitute

    def get_T_cp(self, n_iter=10):
        """
        Refine Initial ICP alignment and return final T rotation matrix and sub contact pnts.
        """

        # Apply initial icp alignment
        init_trans, init_fit, source, substitute = self.icp_alignment(n_iter)

        pose_graph = o3d.registration.PoseGraph()
        accum_pose = np.identity(4)
        pose_graph.nodes.append(o3d.registration.PoseGraphNode(accum_pose))

        pcds = [source, substitute]
        n_pcds = len(pcds)
        for source_id in range(n_pcds):
            for target_id in range(source_id + 1, n_pcds):
                src = pcds[source_id]
                sub = pcds[target_id]

                GTG_mat = o3d.registration.get_information_matrix_from_point_clouds(sub, src,
                                                                                    self.voxel_size,
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
        option = o3d.registration.GlobalOptimizationOption(max_correspondence_distance=self.voxel_size / 10,
                                                           edge_prune_threshold=self.voxel_size / 10,
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
            print trans
            trans_pcds[pcd_id].transform(trans)
            fit = o3d.registration.evaluate_registration(trans_pcds[0],
                                                         trans_pcds[1],
                                                         self.correspondence_thresh).fitness
            fitnesses.append(fit)

        # If these new transformations lower fit score, dont apply them.
        if fitnesses[1] > init_fit:
            aligned_sub = np.asarray(trans_pcds[1].points)
            trans = pose_graph.nodes[1].pose
            self.Ts.append(trans)
        else:
            print "USING INITIAL ICP RESULTS"
            aligned_sub = np.asarray(pcds[1].points)
            trans = init_trans

        aligned_src = np.asarray(trans_pcds[0].points)
        visualize_two_pcs(aligned_sub, aligned_src)
        src_contact_pnt = aligned_src[self.src_tool.contact_pnt_idx, :]
        # Reshape point for easy computation
        src_contact_pnt = src_contact_pnt.reshape(1,-1) \
            if len(src_contact_pnt.shape) == 1 else src_contact_pnt

        # Estimate contact surface on the sub tool
        sub_contact_pnt_idx = self._get_contact_surface(src_contact_pnt,
                                                    aligned_sub,
                                                    aligned_src)

        # sub_contact_pnt = self.sub_tool.get_pnt(sub_contact_pnt_idx)
        sub_contact_pnt = np.asarray(trans_pcds[1].points)[sub_contact_pnt_idx,:]

        # Rerun icp on the src contact surface and sub contact surface for final alignment
        _,_, source_down, substitute_down, source_fpfh, substitute_fpfh = \
            prepare_dataset(src_contact_pnt, sub_contact_pnt, self.voxel_size)

        T_translate = self._icp_wrapper(source_down, substitute_down, source_fpfh, substitute_fpfh).transformation
        # T_translate = get_T_from_R_p(p=translate.reshape(1,-1))

        trans_pcds[1].transform(T_translate) # apply translation
        self.Ts.append(T_translate) # Store final translation

        print "NUM SUB SURFACE pnts ", sub_contact_pnt.shape[0]

        visualize_two_pcs(aligned_sub, aligned_sub[sub_contact_pnt_idx, :])
        visualize_two_pcs(aligned_sub, sub_contact_pnt)
        final_trans = np.linalg.multi_dot(self.Ts) # Combine all transformation into one to return

        if self.visualize:
            o3d.visualization.draw_geometries(trans_pcds, "Aligned")
            # o3d.visualization.draw_geometries([self.src_pcd,
            #                                    self.sub_pcd.transform(final_trans)])


        # Retrun final T and contact surface on original sub tool model
        return final_trans, self.sub_tool.get_pnt(sub_contact_pnt_idx)


    def main(self):
        R, cp = self.get_T_cp(n_iter=8)



if __name__ == '__main__':
    gp = GeneratePointcloud()
    n = 10000
    get_color = True

   # pnts1 = gp.get_random_ply(n, get_color)
   # pnts2 = gp.get_random_ply(n, get_color)
    pnts1 = np.random.uniform(0, 1, size=(n, 3))
    pnts2 = np.random.uniform(1.4, 2, size=(n, 3))

    # pnts2 = gp.mesh_to_pointcloud("./tool_files/point_clouds/a_knifekitchen2.ply", n)
    # pnts2 = gp.mesh_to_pointcloud("a_knifekitchen3/2/a_knifekitchen3_out_8_60_fused.ply", n)
    # pnts2 = gp.load_pointcloud("./tool_files/point_clouds/a_bowlchild.ply", None)
    # pnts2 = gp.mesh_to_pointcloud("/rake_remove_box/2/rake_remove_box_out_2_40_fused.ply", n)
    # pnts1 = gp.mesh_to_pointcloud('ranch/3/ranch_out_6_20_fused.ply',n,  get_color)
    # pnts1 = gp.mesh_to_pointcloud('hammer/2/hammer_out_3_10_fused.ply',n , get_color)
    # pnts1 = gp.mesh_to_pointcloud('chineseknife_1_3dwh/2/chineseknife_1_3dwh_out_4_60_fused.ply',n , get_color)
    # pnts1 = gp.load_pointcloud('./tool_files/point_clouds/a_bowl.ply', None)
    # pnts1 = gp.mesh_to_pointcloud('a_knifekitchen2/2/a_knifekitchen2_out_4_60_fused.ply',n , get_color)
    # pnts1 = gp.mesh_to_pointcloud('clamp_left/2/clamp_left_out_2_10_fused.ply',n,  get_color)
    # pnts1 = gp.mesh_to_pointcloud('screwdriver_right/2/screwdriver_right_out_2_20_fused.ply',n , get_color)
    # pnts2 = gp.mesh_to_pointcloud('clamp_right/2/clamp_right_out_3_10_fused.ply', n, get_color)

    "./tool_files/data_demo_segmented_numbered/rake_remove_box/2/rake_remove_box_out_2_40_fused.ply"
    src = ToolPointCloud(pnts1, contact_pnt_idx=None)

    # src.visualize_bb()
    sub = ToolPointCloud(pnts2)
    # src = sub
    # sub = shrink_pc(sub)

    cntct_pnt = src.get_pc_bb_axis_frame_centered().argmax(axis=0)[1]
    cntct_pnt2 = sub.get_pc_bb_axis_frame_centered().argmax(axis=0)[0]
    # cntct_pnt = np.random.randint(0, src.pnts.shape[0], size = 1).item()
    # cntct_pnt = src.pnts.argmin(axis=0)[0]
    src.contact_pnt_idx = cntct_pnt


    print("SRC TOOL")
    # src.visualize()
    visualize(src.pnts, src.get_pnt(cntct_pnt), src.segments)
    print("SUB TOOL")
    # sub.visualize_bb()
    # sub_seg = [1 if sub.pnts[i,0] > -0.024 else 0 for i in range(sub.pnts.shape[0])]
    # sub.segments=sub_seg
    # visualize(sub.pnts, sub.get_pnt(cntct_pnt2), segments=sub.segments)
    # sub = ToolPointCloud(np.vstack([pnts2.T, sub_seg]).T)

    ts = ToolSubstitution(src, sub, voxel_size=0.02, visualize=True)
    ts.main()
