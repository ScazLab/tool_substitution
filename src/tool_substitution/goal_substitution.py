#!/usr/bin/env python

import random
import numpy as np
import open3d as o3d
from copy import deepcopy
from numpy.linalg import norm

from tool_pointcloud import ToolPointCloud
#from tool_substitution_controller_original import ToolSubstitution, visualize_reg, visualize_tool
from tool_substitution_controller import ToolSubstitution, visualize_reg, visualize_tool, get_np_pc_distance, get_o3d_pc_distance

from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree, KDTree
from get_target_tool_pose import get_T_from_R_p

from util import (np_to_o3d, min_point_distance, rotation_matrix_from_vectors,
                  weighted_min_point_distance, visualize_two_pcs,
                  rotation_matrix_from_box_rots, visualize_vectors,
                  r_x, r_y, r_z, visualize_contact_area,
                  visualize_reg, visualize_tool, visualize_multiple_cps,
                  align_pcd_select_size, get_homogenous_transformation_matrix_inverse)

from get_target_tool_pose import get_T_from_R_p, T_inv, get_scaling_T

# usage one
Tgoal_tool_1 = np.array([[ 0.30106726, -0.95092107, -0.07146767,  0.00624235],
                         [-0.12470171,  0.03504179, -0.99157529,  0.2993885 ],
                         [ 0.94541419,  0.307443,   -0.10803154,  0.06481518],
                         [ 0.,          0.,          0.,          1.        ]])

# usage two
Tgoal_tool_2 = np.array([[ 0.29871531, -0.95180629, -0.06952658,  0.05582589],
                         [-0.12141969,  0.03435668, -0.99200649,  0.42436265],
                         [ 0.94658672,  0.30476942, -0.10530516,  0.05464362],
                         [ 0.,          0.,          0.,          1.        ]])


def get_R_p_from_matrix(T):
    return T[0:-1, 0:-1], np.array([T[0:-1, -1]])

def get_homogenous_transformation_matrix(R, p):
    assert(R.shape[0] == R.shape[1])
    assert(R.shape[0] == p.shape[1])
    return np.c_[np.r_[R, np.zeros((1, R.shape[0]))], np.r_[p.T, [[1]]]]

def get_transformation_matrix_inverse(T):
    R, p = get_R_p_from_matrix(T)
    return get_homogenous_transformation_matrix(R.T, -np.matmul(R.T, p.T).T)

def add_color_normal(pcd): # in-place coloring and adding normal
    pcd.paint_uniform_color(np.random.rand(3))
    size = np.abs((pcd.get_max_bound() - pcd.get_min_bound())).max() / 30
    kdt_n = o3d.geometry.KDTreeSearchParamHybrid(radius=size, max_nn=50)
    pcd.estimate_normals(search_param=kdt_n, fast_normal_computation=False)

def get_Tgoal_rotation(alpha):
    return np.array([[np.cos(alpha), -np.sin(alpha), 0, 0],
                     [np.sin(alpha),  np.cos(alpha), 0, 0],
                     [0,           0,                1, 0],
                     [0,           0,                0, 1]])

def get_Tgoal_tool(alpha):
    Tworld_goal = np.identity(4)
    Tworld_tool = np.matmul(Tworld_goal, Tgoal_tool)

    Tworld_newtool = Tworld_tool.copy()
    Tgoal_rotation = get_Tgoal_rotation(alpha)

    Tworld_newgoal = np.matmul(Tgoal_rotation, Tworld_goal)

    Tnewgoal_newtool = np.matmul(get_transformation_matrix_inverse(Tworld_newgoal), Tworld_newtool)

    return Tnewgoal_newtool

def update_goal(goal_pcd, translation, alpha, scale=1):
    Tgoal_rotation = get_Tgoal_rotation(alpha)
    goal_pcd = deepcopy(goal_pcd)

    goal_pcd.transform(Tgoal_rotation)

    goal_pcd.scale(scale)

    goal_pcd.translate(translation)

    return goal_pcd

def update_tool(tool_pcd, translation):
    tool_pcd.transform(Tgoal_tool)

    tool_pcd.translate(translation)

    return tool_pcd

def load_goal(file_name):
    goal_pcd = o3d.io.read_point_cloud(file_name)
    add_color_normal(goal_pcd)

    return goal_pcd

def load_tool(file_name):
    tool_pcd = o3d.io.read_point_cloud(file_name)
    add_color_normal(tool_pcd)

    return tool_pcd

def calc_contact_surface(src_pnts, goal_pnts, r=.15):
    """
    @src_pnts: (n x 3) ndarray
    @goal_pnts: (m x 3) ndarray
    @r: float, Search radius multiplier for points in contact surface.

    return list of ints caontaining indicies closest points
    in src_pnts to goal_pnts
    """

    # Create ckdtree objs for faster point distance computations.
    src_kd = cKDTree(src_pnts)
    goal_kd = cKDTree(goal_pnts)

    # For each of goal_pnts find pnt in src_pnts with shortest distance and idx
    dists, i = src_kd.query(goal_pnts)
    sorted_pnts_idx = [j[0] for j in sorted(zip(i,dists), key=lambda d: d[1])]
    # Get search radius by finding the distance of the top rth point
    top_r_idx = int(r *dists.shape[0])

    # Get top r
    search_radius = sorted(dists)[top_r_idx]
    # return sorted_pnts_idx[0:top_r_idx]
    print "SEARCH RADIUS: {}".format( search_radius)

    # src_pnts that are within search_radius from goal_pnts are considered part
    # of the contact surface
    cntct_idx = src_kd.query_ball_tree(goal_kd, search_radius)
    # cntct_idx is a list of lists containing idx of goal_pnts within search_radius.
    cntct_idx = [i for i, l in enumerate(cntct_idx) if not len(l) == 0]

    print "SHAPE OF ESITMATED SRC CONTACT SURFACE: ", src_pnts[cntct_idx].shape
    #return src_pnts[cntct_idx, :]
    # visualize_two_pcs(src_pnts, src_pnts[cntct_idx, :])
    return cntct_idx


def calc_contact_surface_from_camera_view(src_pnts, goal_pnts, r=.15):
    """
    @src_pnts: (n x 3) ndarray
    @goal_pnts: (m x 3) ndarray
    @r: float, Search radius multiplier for points in contact surface.

    return list of ints caontaining indicies closest points
    in src_pnts to goal_pnts

    Calculates contact surface of src tool by calculating points visible from
    the vantage point of the contacted goal object.

    Refer to: http://www.open3d.org/docs/latest/tutorial/Basic/pointcloud.html#Hidden-point-removal
    """


    # First, get the closest r% of points in src tool to goal obj.
    initial_cntct_idx = calc_contact_surface(src_pnts, goal_pnts, r=r)

    src_pcd = np_to_o3d(src_pnts)
    goal_pcd = np_to_o3d(goal_pnts)

    src_kd = cKDTree(src_pnts)
    goal_kd = cKDTree(goal_pnts)

    # Find the closest point on the goal obj to the src obj and use
    # that as the position of the 'camera.'
    dists, i = goal_kd.query(src_pnts)
    min_i, min_d = sorted(zip(i,dists), key=lambda d: d[1])[0]

    camera = goal_pnts[min_i, :]
    # Use this heuristic to get the radius of the spherical projection
    diameter = np.linalg.norm(
    np.asarray(src_pcd.get_max_bound()) - np.asarray(src_pcd.get_min_bound()))
    radius = diameter * 100
    # Get idx of points in src_tool from the vantaage of the closest point in
    # goal obj.
    _, camera_cntct_idx = src_pcd.hidden_point_removal(camera, radius)
    # Get intersection of points from both contact surface calculating methods.
    camera_cntct_idx = list(set(camera_cntct_idx).intersection(set(initial_cntct_idx)))

    # NOTE: newer versions of o3d have renamed this function 'select_by_index'
    # src_pcd = src_pcd.select_down_sample(camera_cntct_idx)
    visualize_tool(src_pcd, camera_cntct_idx)
    # o3d.visualization.draw_geometries([src_pcd])

    return camera_cntct_idx

class GoalSubstitution(ToolSubstitution):
    def __init__(self, src_goal_pc, sub_goal_pc, voxel_size=0.02, visualize=False):
        "docstring"
        super(GoalSubstitution, self).__init__(src_goal_pc, sub_goal_pc,
                                            voxel_size, visualize)
        
        #source_goal_pcd = deepcopy(self.src_pcd)
        #source_contact_point_index = deepcopy(self.src_tool.contact_pnt_idx)
        
        #print "[goal_substitution][__init__] source_contact_point_index"
        #print source_contact_point_index
        
        #source_goal_pcd.paint_uniform_color(np.array([0., 1, 0.]))
        
        #source_goal_colors = np.asarray(source_goal_pcd.colors)
        #source_goal_colors[source_contact_point_index, :] = np.array([1., 0., 0.])
        
        #source_goal_pcd.colors = o3d.utility.Vector3dVector(source_goal_colors)
        #o3d.visualization.draw_geometries([source_goal_pcd], "source goal tool and its contact area")

    #original
    #def get_T_cp(self, n_iter=10):

        #self._scale_pcs()

        #align_fit = self._align_pnts()

        #sub_pnts = self._get_sub_pnts(False)
        #src_pnts = self._get_src_pnts(False)

        #icp_trans, fit, aligned_sub_pcd, aligned_src_pcd = self.icp_alignment(sub_pnts,
                                                                              #src_pnts,
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

        #_, aligned_sub_pcd, T_align,_ = self.refine_registration(T_align,
                                                                 #aligned_sub_pcd,
                                                                 #aligned_src_pcd,
                                                                 #self.voxel_size)


        #self.T_sub_pcd.transform(T_align)


        #aligned_src = np.asarray(self._get_src_pnts(False))
        #aligned_sub = np.asarray(aligned_sub_pcd.points)


        #src_contact_pnt = aligned_src[self.src_tool.contact_pnt_idx, :]
        #sub_cp_idx = self._get_contact_surface(src_contact_pnt,
                                              #aligned_sub,
                                              #aligned_src)

        ## mean_sub_cp = np.mean(aligned_sub[sub_cp_idx, :], axis=0)
        #mean_src_cp = np.mean(src_contact_pnt, axis=0)
        #scale_T = T_inv(np.linalg.multi_dot(self.scale_Ts)) # Get inverted scaling T
        #scale_f = np.diag(scale_T)[:3] # get first 3 diag entries.
        #final_scale_T = get_scaling_T(scale_f, mean_src_cp)

        #descaled_aligned_sub = deepcopy(self.T_sub_pcd).transform(final_scale_T)

        #if self.visualize:
            #print "DESCALING"
            #visualize_reg(aligned_sub_pcd, self.sub_pcd, descaled_aligned_sub,
                          #result_cp_idx=sub_cp_idx,
                          #target_cp_idx=sub_cp_idx)



        #self.T_sub_pcd.transform(final_scale_T)
        #self.T_sub_pcd.transform(self.temp_src_T)
        #print "TEST"
        #if self.visualize:
            #visualize_reg(aligned_sub_pcd, self.src_pcd, self.T_sub_pcd,
                          #result_cp_idx=sub_cp_idx,
                          #target_cp_idx=self.src_tool.contact_pnt_idx)

        #descaled_sub_pnts = np.asarray(self._get_sub_pnts(get_segments=False))
        #original_sub_pnts = np.asarray(self.sub_pcd.points)

        #new_vox_size = self.voxel_size
        #new_corr_thresh = self.correspondence_thresh

        #icp_T, icp_fit, _, _= self.icp_alignment(original_sub_pnts,
                                                 #descaled_sub_pnts,
                                                 #new_corr_thresh,
                                                 #n_iter)

        #aligned_sub_pcd,_, refine_T, refine_fit = self.refine_registration(icp_T,
                                                                           #self.sub_pcd,
                                                                           ## descaled_aligned_sub,
                                                                           #self.T_sub_pcd ,
                                                                           #new_vox_size)


        
        #print "ICP fit: ", icp_fit
        #print "Refine fit: ", refine_fit

        #final_trans = icp_T if icp_fit > refine_fit else refine_T

        #src_contact_pnt = self._get_sub_pnts(False)[sub_cp_idx, :]
        #new_sub_cp_idx = self._get_contact_surface(src_contact_pnt,
                                               #np.asarray(deepcopy(self.sub_pcd).transform(final_trans).points),
                                               #self._get_sub_pnts(False))

        #print "FINAL TEST"
        #if self.visualize:
            #visualize_reg(self.sub_pcd,
                          #self.T_sub_pcd,
                          #deepcopy(self.sub_pcd).transform(final_trans),
                          #result_cp_idx=new_sub_cp_idx,
                          #target_cp_idx=sub_cp_idx)
            #visualize_reg(self.sub_pcd,
                          #self.src_pcd,
                          #self.T_sub_pcd,
                          #result_cp_idx=sub_cp_idx,
                          #target_cp_idx=self.src_tool.contact_pnt_idx)



        #src_mean_cp = np.mean(np.asarray(self.src_pcd.points)[self.src_tool.contact_pnt_idx,:], axis=0)
        #sub_mean_cp = np.mean(np.asarray(self.T_sub_pcd.points)[sub_cp_idx, :], axis=0)
        #T_translate = get_T_from_R_p(p=(src_mean_cp-sub_mean_cp))
        #final_trans = np.matmul(T_translate, final_trans)


        #if self.visualize:
            #print "FINAL RESULT"
            #visualize_reg(self.sub_pcd,
                          #self.src_pcd,
                          #deepcopy(self.sub_pcd).transform(final_trans),
                          #result_cp_idx=new_sub_cp_idx,
                          #target_cp_idx=self.src_tool.contact_pnt_idx)


        ## Must account for the fact that the src and sub tool pcs have been centered at
        ## 0, so we create one final translation transformation
        #orig_sub_tool = self._np_to_o3d(self.sub_tool.get_unnormalized_pc())
        #orig_src_tool = self._np_to_o3d(self.src_tool.get_unnormalized_pc())
        #orig_sub_tool.transform(final_trans)

        #src_mean_cp = np.mean(np.asarray(orig_src_tool.points)[self.src_tool.contact_pnt_idx,:], axis=0)
        #sub_mean_cp = np.mean(np.asarray(orig_sub_tool.points)[new_sub_cp_idx, :], axis=0)
        #T_translate = get_T_from_R_p(p=(src_mean_cp-sub_mean_cp))
        #final_trans = np.matmul(T_translate, final_trans)

        ## print "TRUE FINAL"
        ## visualize_reg(orig_sub_tool,
        ##               orig_src_tool,
        ##               deepcopy(orig_sub_tool).transform(T_translate),
        ##               result_cp_idx=new_sub_cp_idx,
        ##               target_cp_idx=self.src_tool.contact_pnt_idx)

        #return final_trans, self.sub_tool.get_pnt(new_sub_cp_idx)
    
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
        
        #print "source center: ", T_src_pcd.get_center()
        #print "sub center: ", T_sub_pcd.get_center()
        #o3d.visualization.draw_geometries([T_src_pcd, T_sub_pcd], "goal initial normalize")
        
        copy_src_np_pnts = self._get_src_pnts(T_src_pcd=T_src_pcd, get_segments=False)
        copy_sub_np_pnts = self._get_sub_pnts(T_sub_pcd=T_sub_pcd, get_segments=False)

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

        original_score = 0.
        original_src_pcd = deepcopy(T_src_pcd)
        original_sub_pcd = self._np_to_o3d(copy_sub_np_pnts)
        src_tool_norm = deepcopy(src_bb.norms)
        sub_tool_norm = deepcopy(sub_bb.norms)
        permed_scale_f = src_tool_norm / sub_tool_norm
        original_T_sub_action_part_scale = get_scaling_T(scale=permed_scale_f)
        original_sub_pcd.transform(original_T_sub_action_part_scale)
        original_distance = get_o3d_pc_distance(T_src_pcd, original_sub_pcd)
        original_score = (np.identity(4), original_T_sub_action_part_scale, original_distance)
        
        for p in [[0, 1, 2], [1, 0, 2], [2, 1, 0]]:
            print "........................................................."
            print "p"
            print p
            #T, sub_pnts = self._rotate_np_with_segments(copy_sub_np_pnts, p)
            R = []
            for i in p:
                if i == 0:
                    R.append([1., 0., 0.])
                elif i == 1:
                    R.append([0., 1., 0.])
                elif i == 2:
                    R.append([0., 0., 1.])  
            R = np.array(R).T
            T = np.identity(4)
            T[:3, :3] = R
            src_tool_norm = src_bb.norms
            sub_tool_norm = deepcopy(sub_bb.norms)[list(p)]
            print "src_tool_norm: ", src_tool_norm
            print "sub_tool_norm: ", sub_tool_norm
            #sub_tool_norms = sub_tool_norms[list(p)]
            scaled_sub_pcd = self._np_to_o3d(copy_sub_np_pnts)
            scaled_sub_pcd.transform(T)
            #o3d.visualization.draw_geometries([T_src_pcd, scaled_sub_pcd], "rotate!")
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
            #o3d.visualization.draw_geometries([T_src_pcd, scaled_sub_pcd], "scaling!")

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
        
        print "[goal_substitution][_align_pnts]: "
        #if original_distance < distance * 1.05:
            #T_rot, T_sub_action_part_scale, distance = original_score[0], original_score[1], original_score[2]
        
        print "selected is T_rot"
        print T_rot
        print "selected is T_sub_action_part_scale"
        print T_sub_action_part_scale

        T_sub_pcd = self._np_to_o3d(copy_sub_np_pnts)
        T_sub_pcd.transform(T_rot)
        unscaled_T_sub_to_return = np.matmul(T_rot, T_sub_to_return)
        #self.T_sub_pcd.transform(T_rot)
        #o3d.visualization.draw_geometries([self.T_sub_pcd, self.T_src_pcd], "after rotate")
        T_sub_pcd.transform(T_sub_action_part_scale)
        T_sub_to_return = np.matmul(T_sub_action_part_scale, unscaled_T_sub_to_return)
        #self.T_sub_pcd.transform(T_sub_action_part_scale)
        #o3d.visualization.draw_geometries([self.T_sub_pcd, self.T_src_pcd], "after scaling")

        #scale_Ts = [T_sub_action_part_scale]

        #self.scale_Ts.append(T_sub_action_part_scale) # Append scaling matrix

        return distance, distance / max_length, T_sub_action_part_scale, T_src_pcd, T_sub_pcd, temp_src_T, T_src_to_return, T_sub_to_return, unscaled_T_sub_to_return    
    
    def _get_sub_contact_indices(self, src_pnts, sub_pnts, src_contact_indices):
        sub_action_part_cp_idx = self._get_contact_surface(src_pnts[src_contact_indices],
                                                           sub_pnts,
                                                           src_pnts)
    
        sub_cp_idx = []
        if len(sub_action_part_cp_idx) > 5:
            # Corresponding idx of sub contact surface for original sub pointcloud.
            sub_cp_idx = sub_action_part_cp_idx
        
        return sub_cp_idx    
    
    def step_2_get_initial_alignment_contact_area(self, src_pcd, sub_pcd, T_src, T_sub, T_sub_scale, unscaled_T_sub):
        sub_contact_point_idx = self._get_sub_contact_indices(np.asarray(src_pcd.points), np.asarray(sub_pcd.points), np.asarray(self.src_tool.contact_pnt_idx))
        
        if self.visualize:
            visualize_tool(sub_pcd, cp_idx=sub_contact_point_idx, name="Step 2: contact area with initial alignment on descaled sub tool")
            visualize_tool(self.sub_pcd, cp_idx=sub_contact_point_idx, name="Step 2: contact area with initial alignment on original sub tool")
        
        return (sub_contact_point_idx, deepcopy(T_src), deepcopy(T_sub), deepcopy(T_sub_scale), deepcopy(sub_pcd), deepcopy(unscaled_T_sub)) # when revert this, first unscale, and then unrotate            
    
    def step_3_scale_sub_tool(self, src_pcd, sub_pcd):
        src_action_part = deepcopy(np.array(src_pcd.points))
        sub_action_part = deepcopy(np.array(sub_pcd.points))
        
        #if self.visualize:
            #copy_sub_pcd = o3d.geometry.PointCloud()
            #copy_sub_pcd.points = o3d.utility.Vector3dVector(deepcopy(src_action_part))
            #copy_sub_pcd.paint_uniform_color(np.array([1., 0., 0.]))
            #copy_src_pcd = o3d.geometry.PointCloud()
            #copy_src_pcd.points = o3d.utility.Vector3dVector(deepcopy(sub_action_part))
            #copy_src_pcd.paint_uniform_color(np.array([0., 1., 0.]))
            #o3d.visualization.draw_geometries([copy_src_pcd, copy_sub_pcd], "Step 3: action parts chosen to be scaled")        

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
        
        if self.visualize:
            copy_scaled_sub_pcd = deepcopy(scaled_sub_pcd)
            copy_scaled_sub_pcd.paint_uniform_color(np.array([1., 0., 0.]))
            copy_src_pcd = deepcopy(self.src_pcd)
            copy_src_pcd.paint_uniform_color(np.array([0., 1., 0.]))
            o3d.visualization.draw_geometries([copy_src_pcd, copy_scaled_sub_pcd], "Step 3: scale sub action part")
        
        return scaled_sub_pcd, T_sub_action_part_scale    
    
    def step_4_register_action_parts(self, src_pcd, sub_pcd):
        copy_src_pcd = deepcopy(src_pcd)
        copy_sub_pcd = deepcopy(sub_pcd)
        
        src_action_pcd = o3d.geometry.PointCloud()
        src_action_pcd.points = o3d.utility.Vector3dVector(np.asarray(copy_src_pcd.points))
        
        sub_action_pcd = o3d.geometry.PointCloud()
        sub_action_pcd.points = o3d.utility.Vector3dVector(np.asarray(copy_sub_pcd.points))
        
        aligned_set, min_transformations, min_threshold = align_pcd_select_size([src_action_pcd, sub_action_pcd])
        
        if self.visualize:
            copy_src_pcd = deepcopy(aligned_set[0])
            copy_src_pcd.paint_uniform_color(np.array([0., 1., 0.]))
            copy_sub_pcd = deepcopy(aligned_set[1])
            copy_sub_pcd.paint_uniform_color(np.array([1., 0., 0.]))            
            o3d.visualization.draw_geometries([copy_src_pcd, copy_sub_pcd], "Step 4: align action parts")
        
        return aligned_set[1], min_transformations[1], min_threshold    
    
    def step_5_get_aligned_contact_area(self, src_pcd, sub_pcd, T_src, T_sub, T_sub_scale, unscaled_T_sub):
        sub_contact_point_idx = self._get_sub_contact_indices(np.asarray(src_pcd.points), np.asarray(sub_pcd.points), np.asarray(self.src_tool.contact_pnt_idx))
        
        if self.visualize:
            visualize_tool(sub_pcd, cp_idx=sub_contact_point_idx, name="Step 5: Contact area from ICP")
        
        return (sub_contact_point_idx, deepcopy(T_src), deepcopy(T_sub), deepcopy(T_sub_scale), deepcopy(sub_pcd), deepcopy(unscaled_T_sub))            
    
    def step_6_choose_contact_area(self, contact_area_1, contact_area_2):
        sub_pcd = deepcopy(self.sub_pcd)
        contact_area_1_pnts = np.asarray(sub_pcd.points)[contact_area_1[0]]
        contact_area_2_pnts = np.asarray(sub_pcd.points)[contact_area_2[0]]
        
        contact_area = None
        
        if len(contact_area_1_pnts) == 0 and len(contact_area_2_pnts) == 0:
            print "[tool_substitution_controller][step_6_choose_contact_area] Both contact areas are empty. Choose 1"
            contact_area = contact_area_1
            return contact_area
        
        if len(contact_area_1_pnts) == 0:
            print "[tool_substitution_controller][step_6_choose_contact_area] initial alignment contact areas is empty. Choose 2"
            contact_area = contact_area_2
            return contact_area
        
        if len(contact_area_2_pnts) == 0:
            print "[tool_substitution_controller][step_6_choose_contact_area] ICP alignment contact areas is empty. Choose 1"
            contact_area = contact_area_1
            return contact_area            
        
        print "[tool_substitution_controller][step_6_choose_contact_area] num of points on contact_area_1_indices: ", len(contact_area_1_pnts)
        print "[tool_substitution_controller][step_6_choose_contact_area] num of points on contact_area_2_indices: ", len(contact_area_2_pnts)
    
        if len(contact_area_1_pnts) * 1. / (len(contact_area_2_pnts) * 1.) > 2:
            print "[tool_substitution_controller][step_6_choose_contact_area] Initial alignment contact areas has a lot more points. Choose 1"
            contact_area = contact_area_1
            return contact_area

        if len(contact_area_2_pnts) * 1. / (len(contact_area_1_pnts) * 1.) > 2:
            print "[tool_substitution_controller][step_6_choose_contact_area] ICP alignment contact areas has a lot more points. Choose 2"
            contact_area = contact_area_2
            return contact_area        

        sub_action_part = deepcopy(np.array(self.sub_pcd.points))
        sub_action_part_bb = ToolPointCloud(sub_action_part, normalize=False).bb
        sub_action_part_bb._calculate_axis()
        sub_action_part_norm = sub_action_part_bb.norms
        
        contact_area_1_pcd = deepcopy(self._np_to_o3d(contact_area_1_pnts))
        contact_area_2_pcd = deepcopy(self._np_to_o3d(contact_area_2_pnts))
        contact_area_1_pcd_center = contact_area_1_pcd.get_center()
        contact_area_2_pcd_center = contact_area_2_pcd.get_center()
        
        contact_area_distance = norm(contact_area_1_pcd_center - contact_area_2_pcd_center)
        contact_area_distance_percentage = contact_area_distance / max(sub_action_part_norm)
        print "[tool_substitution_controller][step_6_choose_contact_area] contact area distance: ", contact_area_distance
        print "[tool_substitution_controller][step_6_choose_contact_area] contact area distance relative to tool action: ", contact_area_distance_percentage
        if contact_area_distance_percentage < 0.05: # the two are very closest
            print "[tool_substitution_controller][step_6_choose_contact_area] The contact areas are close. Choose 1: the initial alignment contact area"
            contact_area = contact_area_1
        else:
            print "[tool_substitution_controller][step_6_choose_contact_area] The contact areas are far away from each other."
            contact_area_2_pcd.translate(contact_area_1_pcd_center - contact_area_2_pcd_center)
            
            aligned_set, min_transformations, min_threshold = align_pcd_select_size([contact_area_1_pcd, contact_area_2_pcd])
            
            aligned_set_1_center = aligned_set[0].get_center()
            aligned_set_2_center = aligned_set[1].get_center()
            aligned_set[1].translate(aligned_set_1_center - aligned_set_2_center)
            distance = aligned_set[0].compute_point_cloud_distance(aligned_set[1])
            
            if self.visualize:
                copy_contact_area_1_pcd = deepcopy(aligned_set[0])
                copy_contact_area_1_pcd.paint_uniform_color(np.array([1., 0., 0.]))
                copy_contact_area_2_pcd = deepcopy(aligned_set[1])
                copy_contact_area_2_pcd.paint_uniform_color(np.array([0., 1., 0.]))
                o3d.visualization.draw_geometries([copy_contact_area_1_pcd, copy_contact_area_2_pcd], "Step 6: align the two contact areas")        
            
            dislikeness = np.average(distance) / max(sub_action_part_norm) # the high the value, the more dislike the two contact areas are
            
            print "[tool_substitution_controller][step_6_choose_contact_area] average distance: ", np.average(distance)
            print "[tool_substitution_controller][step_6_choose_contact_area] dislikeness: ", dislikeness
            print "[tool_substitution_controller][step_6_choose_contact_area] max distance (0, 1): ", np.max(distance)
            print "[tool_substitution_controller][step_6_choose_contact_area] max distance percentage(0, 1): ", np.max(distance) / max(sub_action_part_norm)
            print "[tool_substitution_controller][step_6_choose_contact_area] sub action dimension: ", max(sub_action_part_norm)
            
            if dislikeness > 0.02: # tune this value
                print "[tool_substitution_controller][step_6_choose_contact_area] The contact areas are different."
                src_contact_area_pcd = o3d.geometry.PointCloud()
                src_contact_area_pcd.points = o3d.utility.Vector3dVector(np.asarray(self.src_pcd.points)[self.src_tool.contact_pnt_idx])
                
                aligned_set_1, _, _ = align_pcd_select_size([contact_area_1_pcd, src_contact_area_pcd])
                aligned_set_1_sub_center = aligned_set_1[0].get_center()
                aligned_set_1_src_center = aligned_set_1[1].get_center()
                aligned_set_1[1].translate(aligned_set_1_sub_center - aligned_set_1_src_center)
                distance_1 = np.average(aligned_set_1[1].compute_point_cloud_distance(aligned_set_1[0]))
                
                aligned_set_2, _, _ = align_pcd_select_size([contact_area_2_pcd, src_contact_area_pcd])
                aligned_set_2_sub_center = aligned_set_2[0].get_center()
                aligned_set_2_src_center = aligned_set_2[1].get_center()
                aligned_set_2[1].translate(aligned_set_2_sub_center - aligned_set_2_src_center)
                distance_2 = np.average(aligned_set_2[1].compute_point_cloud_distance(aligned_set_2[0]))
                
                if self.visualize:
                    o3d.visualization.draw_geometries(aligned_set_1, "Step 6: align contact area 1 and source contact area")
                    o3d.visualization.draw_geometries(aligned_set_2, "Step 6: align contact area 2 and source contact area")
                print "[tool_substitution_controller][step_6_choose_contact_area] contact area 1 distance to source: ", distance_1
                print "[tool_substitution_controller][step_6_choose_contact_area] contact area 2 distance to source: ", distance_2
                if distance_1 < distance_2:
                    print "[tool_substitution_controller][step_6_choose_contact_area] Initial Alignment contact areas looks more like the source contact area. Choose 1: the initial alignment contact area"
                    contact_area = contact_area_1
                else:
                    print "[tool_substitution_controller][step_6_choose_contact_area] ICP Alignment contact areas looks more like the source contact area. Choose 2: the ICP contact area"
                    contact_area = contact_area_2
            else:
                print "[tool_substitution_controller][step_6_choose_contact_area] The contact areas are similar. Choose 1: the initial alignment contact area"
                contact_area = contact_area_1
        
        return contact_area    
    
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
        step_0_unscaled_T_sub              = step_0_results[8] # re_oriented, not scaled
        # the previous scaling or transformations can be ignored, as the whole was to find the action segment

        #step_1_results = self.step_1_find_action_part(step_0_src_pcd, step_0_sub_pcd)
        
        #step_1_scaled_src_action_pnts = step_1_results[0]
        #step_1_scaled_sub_action_pnts = step_1_results[1]
        #step_1_src_action_indices     = step_1_results[2]
        #step_1_sub_action_indices     = step_1_results[3]
        
        
        #src action segment: self._src_action_segment
        #sub action segment: self._sub_action_segment
        #src contact area index: self.src_tool.contact_pnt_idx
        
        # the contact area of based on the initial alignment
        step_2_results = self.step_2_get_initial_alignment_contact_area(step_0_src_pcd,
                                                                        step_0_sub_pcd,
                                                                        step_0_T_src, 
                                                                        step_0_T_sub, 
                                                                        step_0_T_sub_scale,
                                                                        step_0_unscaled_T_sub)
        contact_area_1 = step_2_results # when revert this, first unscale, and then unrotate  

        # scale the two goals based on the size of the two goals
        step_3_results = self.step_3_scale_sub_tool(self.src_pcd, self.sub_pcd)
        
        step_3_scaled_sub_pcd          = step_3_results[0]
        step_3_T_sub_action_part_scale = step_3_results[1] # scale appeared first, so for the contact area found with this method, first unrotate, and then unscale
        
        # use ICP to align the two objects
        step_4_results = self.step_4_register_action_parts(self.src_pcd, step_3_scaled_sub_pcd)
        
        step_4_scaled_aligned_sub_action_pcd = step_4_results[0]
        step_4_T_sub                         = step_4_results[1]
        step_4_threshold                     = step_4_results[2]
        
        # find the corresponding contact area
        scaled_aligned_sub_pcd = deepcopy(step_4_scaled_aligned_sub_action_pcd)
        #scaled_aligned_sub_pcd.transform(step_4_T_sub)
        step_5_results = self.step_5_get_aligned_contact_area(self.src_pcd,
                                                              scaled_aligned_sub_pcd,                                                            
                                                              np.identity(4),
                                                              step_4_T_sub,
                                                              np.identity(4),
                                                              step_4_T_sub)
        contact_area_2 = step_5_results
        
        # choose the contact area
        step_6_results = self.step_6_choose_contact_area(contact_area_1, contact_area_2)
        contact_area = step_6_results
        
        # descale and align the pc based on the contact area chosen
        Tsrc_sub = self.step_7_align_tools(contact_area)
        
        print "[goal_substitution][get_T_cp] RETURN Tsrc_sub: "
        print Tsrc_sub
        
        print "[goal_substitution][get_T_cp] RETURN contact area: "
        print contact_area[0]
        
        return Tsrc_sub, contact_area[0]    

if __name__ == '__main__':
    """
    First, load the original point cloud
    """
    goal_file_path = "../../tool_files/push.ply"
    tool_file_path = "../../tool_files/rake.ply"

    original_goal = load_goal(goal_file_path)
    original_tool = load_tool(tool_file_path)

    o3d.visualization.draw_geometries([original_goal, original_tool], "original")

    """
    Second, get a training sample of how the tool should be used
    """
    # get a training sample with the original size of the goal
    # usage_option = raw_input("which usage? 1-pull; 2-push: ")
    # Tgoal_tool = np.identity(4)
    # if usage_option == "1":
    #     Tgoal_tool = Tgoal_tool_1
    # elif usage_option == "2":
    #     Tgoal_tool = Tgoal_tool_2


    Tgoal_tool = Tgoal_tool_2

    translation = np.array([random.uniform(-10, 10), random.uniform(-10, 10), random.uniform(-10, 10)]).T
    alpha = random.uniform(0, 2 * np.pi)

    source_goal_pcd = update_goal(original_goal, translation, alpha, scale=1)
    tool_pcd = update_tool(original_tool, translation)

    print "TRAINING"
    o3d.visualization.draw_geometries([tool_pcd, source_goal_pcd], "training sample")

    tool_cp_idx = calc_contact_surface_from_camera_view(np.asarray(tool_pcd.points),
                                                    np.asarray(source_goal_pcd.points))
    goal_cp_idx = calc_contact_surface_from_camera_view(np.asarray(source_goal_pcd.points),
                                                    np.asarray(tool_pcd.points), r=.05)
    """
    Third, give a goal with a different size
    """
    # Given the known point cloud of the tool tool_pcd, and the known point cloud of the goal source_goal_pcd
    # Given the relative pose between the tool and the goal Tsourcegoal_tool
    # Given a smaller/larger goal, substitute_goal_pcd
    # find the Tsubgoal_tool

    sub_translation = np.array([random.uniform(-1.5, 1.5), random.uniform(-1.5, 1.5), random.uniform(-1.5, 1.5)]).T
    sub_alpha = random.uniform(0, 2 * np.pi)
    sub_scale = random.uniform(0.1, .1)

    original_tool = load_tool(tool_file_path)
    original_goal = load_goal(goal_file_path)
    substitute_goal_pcd = update_goal(original_goal, sub_translation,
                                      sub_alpha, scale=sub_scale)


    src_translation = np.array([random.uniform(-1.5, 1.5), random.uniform(-1.5, 1.5), random.uniform(-1.5, 1.5)]).T
    src_alpha = random.uniform(0, 2 * np.pi)
    src_scale = random.uniform(1, 1.1)

    
    original_goal_pcd = update_goal(original_goal, src_translation,
                                      src_alpha, scale=src_scale)

    o3d.visualization.draw_geometries([original_tool, substitute_goal_pcd], "question to solve")

    print "GOAL"
    o3d.visualization.draw_geometries([original_goal, substitute_goal_pcd])

    src_goal = ToolPointCloud(np.asarray(original_goal.points))
    sub_goal = ToolPointCloud(np.asarray(substitute_goal_pcd.points))
    print "SRC NORMS: ", src_goal.bb.norms

    src_goal.contact_pnt_idx = goal_cp_idx
    goal_sub = GoalSubstitution(src_goal, sub_goal,
                                voxel_size=.001, visualize=True)

    goal_sub.get_T_cp()
