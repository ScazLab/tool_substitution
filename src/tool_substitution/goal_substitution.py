#!/usr/bin/env python

import random
import numpy as np
import open3d as o3d
from copy import deepcopy

from tool_pointcloud import ToolPointCloud
from tool_substitution_controller import ToolSubstitution, visualize_reg, visualize_tool

from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree, KDTree
from get_target_tool_pose import get_T_from_R_p

from util import np_to_o3d

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


    def get_T_cp(self, n_iter=10):

        self._scale_pcs()

        align_fit = self._align_pnts()

        sub_pnts = self._get_sub_pnts(False)
        src_pnts = self._get_src_pnts(False)

        icp_trans, fit, aligned_sub_pcd, aligned_src_pcd = self.icp_alignment(sub_pnts,
                                                                              src_pnts,
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
                                                                 aligned_sub_pcd,
                                                                 aligned_src_pcd,
                                                                 self.voxel_size)


        self.T_sub_pcd.transform(T_align)


        aligned_src = np.asarray(self._get_src_pnts(False))
        aligned_sub = np.asarray(aligned_sub_pcd.points)


        src_contact_pnt = aligned_src[self.src_tool.contact_pnt_idx, :]
        sub_cp_idx = self._get_contact_surface(src_contact_pnt,
                                              aligned_sub,
                                              aligned_src)

        # mean_sub_cp = np.mean(aligned_sub[sub_cp_idx, :], axis=0)
        mean_src_cp = np.mean(src_contact_pnt, axis=0)
        scale_T = T_inv(np.linalg.multi_dot(self.scale_Ts)) # Get inverted scaling T
        scale_f = np.diag(scale_T)[:3] # get first 3 diag entries.
        final_scale_T = get_scaling_T(scale_f, mean_src_cp)

        descaled_aligned_sub = deepcopy(self.T_sub_pcd).transform(final_scale_T)

        if self.visualize:
            print "DESCALING"
            visualize_reg(aligned_sub_pcd, self.sub_pcd, descaled_aligned_sub,
                          result_cp_idx=sub_cp_idx,
                          target_cp_idx=sub_cp_idx)



        self.T_sub_pcd.transform(final_scale_T)
        self.T_sub_pcd.transform(self.temp_src_T)
        print "TEST"
        if self.visualize:
            visualize_reg(aligned_sub_pcd, self.src_pcd, self.T_sub_pcd,
                          result_cp_idx=sub_cp_idx,
                          target_cp_idx=self.src_tool.contact_pnt_idx)

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

        src_contact_pnt = self._get_sub_pnts(False)[sub_cp_idx, :]
        new_sub_cp_idx = self._get_contact_surface(src_contact_pnt,
                                               np.asarray(deepcopy(self.sub_pcd).transform(final_trans).points),
                                               self._get_sub_pnts(False))

        print "FINAL TEST"
        if self.visualize:
            visualize_reg(self.sub_pcd,
                          self.T_sub_pcd,
                          deepcopy(self.sub_pcd).transform(final_trans),
                          result_cp_idx=new_sub_cp_idx,
                          target_cp_idx=sub_cp_idx)
            visualize_reg(self.sub_pcd,
                          self.src_pcd,
                          self.T_sub_pcd,
                          result_cp_idx=sub_cp_idx,
                          target_cp_idx=self.src_tool.contact_pnt_idx)



        src_mean_cp = np.mean(np.asarray(self.src_pcd.points)[self.src_tool.contact_pnt_idx,:], axis=0)
        sub_mean_cp = np.mean(np.asarray(self.T_sub_pcd.points)[sub_cp_idx, :], axis=0)
        T_translate = get_T_from_R_p(p=(src_mean_cp-sub_mean_cp))
        final_trans = np.matmul(T_translate, final_trans)


        if self.visualize:
            print "FINAL RESULT"
            visualize_reg(self.sub_pcd,
                          self.src_pcd,
                          deepcopy(self.sub_pcd).transform(final_trans),
                          result_cp_idx=new_sub_cp_idx,
                          target_cp_idx=self.src_tool.contact_pnt_idx)


        # Must account for the fact that the src and sub tool pcs have been centered at
        # 0, so we create one final translation transformation
        orig_sub_tool = self._np_to_o3d(self.sub_tool.get_unnormalized_pc())
        orig_src_tool = self._np_to_o3d(self.src_tool.get_unnormalized_pc())
        orig_sub_tool.transform(final_trans)

        src_mean_cp = np.mean(np.asarray(orig_src_tool.points)[self.src_tool.contact_pnt_idx,:], axis=0)
        sub_mean_cp = np.mean(np.asarray(orig_sub_tool.points)[new_sub_cp_idx, :], axis=0)
        T_translate = get_T_from_R_p(p=(src_mean_cp-sub_mean_cp))
        final_trans = np.matmul(T_translate, final_trans)

        # print "TRUE FINAL"
        # visualize_reg(orig_sub_tool,
        #               orig_src_tool,
        #               deepcopy(orig_sub_tool).transform(T_translate),
        #               result_cp_idx=new_sub_cp_idx,
        #               target_cp_idx=self.src_tool.contact_pnt_idx)

        return final_trans, self.sub_tool.get_pnt(new_sub_cp_idx)

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
