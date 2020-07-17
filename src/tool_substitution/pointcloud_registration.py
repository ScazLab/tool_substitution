#!/usr/bin/env python

import copy
import open3d as o3d
import numpy as np

from sample_pointcloud import GeneratePointcloud
from tool_pointcloud import ToolPointCloud




def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def preprocess_point_cloud(pcd, voxel_size, norm_fn=None):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    if norm_fn is None:

        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    else:
        pcd_down.normals = o3d.io.read_point_cloud(norm_fn)

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_dataset(pnts1, pnts2, voxel_size):
    print(":: Load two point clouds and disturb initial pose.")

    source = o3d.geometry.PointCloud()
    target = o3d.geometry.PointCloud()

    source.points = o3d.utility.Vector3dVector(pnts1)
    target.points = o3d.utility.Vector3dVector(pnts2)


    trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    trans_init = np.identity(4)
    # source.transform(trans_init)
    # draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.registration.RANSACConvergenceCriteria(4000000, 1000))
    return result


def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, result_ransac):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        # o3d.registration.TransformationEstimationPointToPlane()
        o3d.registration.TransformationEstimationPointToPoint()
    )

    return result


def get_rake_pcs(n=2000):

    gp = GeneratePointcloud()

    pnts1, pnts2 = gp.get_both_rake_points(n)
    pc1 = ToolPointCloud(pnts1)
    pnts1 = pc1.get_pc_bb_axis_frame_centered()

    # pnts2 = gp.mesh_to_pointcloud(n, "./tool_files/rake.stl")
    pc2 = ToolPointCloud(pnts2)
    # pnts2 = pc2.scale_pnts_to_target(pc1)
    # pc2 = ToolPointCloud(pnts2)

    pnts2 = pc2.get_pc_bb_axis_frame_centered()

    x_range = pnts2[0].max() - pnts2[0].min()
    voxel_size = x_range / 30

    means = pnts2.mean(axis=0)
    y_mean = means[1]
    z_mean = means[2]

    pnts2 = pnts2[np.where(pnts2[:,2] < z_mean), :][0]
    # pnts2 = pnts2[np.where(pnts2[:,1] < y_mean), :][0]
    return pnts1, pnts2, voxel_size

def pc_registration(src_fn, target_fn, n_pnts=None, n_iter=3):
    """
    Aligns a src pointcloud with a target pointcloud.
    @src_fn, str: filepath to src pointcloud (.ply or .pcd)
    @target_fn, str: filepath to target pointcloud
    @n_pnts, int: The number of points to sample from both pcs
    @n_iter, int:  Number of attempts to find best alignment.

    """

    # Load src and target pointclouds
    gp = GeneratePointcloud()

    src_pnt = gp.load_pointcloud(src_fn, n_pnts)
    # target_pnt = gp.load_pointcloud(target_fn, n_pnts)
    # src_pnt = gp.mesh_to_pointcloud(src_fn, n_pnts)
    target_pnt = gp.mesh_to_pointcloud(target_fn, n_pnts)

    src_pc = ToolPointCloud(src_pnt)
    target_pc = ToolPointCloud(target_pnt)

    target_pnt = target_pc.get_pc_bb_axis_frame_centered()
    src_pnt = src_pc.get_pc_bb_axis_frame_centered()

    # Heuristic for determining the size of the voxels.
    x_range = target_pnt[0].max() - target_pnt[0].min()
    voxel_size = x_range /  .5

    source, target, source_down, target_down, source_fpfh, target_fpfh = \
            prepare_dataset(src_pnt, target_pnt, voxel_size)

    results = []
    for i in range(n_iter):
        result_ransac = execute_global_registration(source_down, target_down,
                                                    source_fpfh, target_fpfh,
                                                    voxel_size)
        fitness = result_ransac.fitness

        print "\n\nIter: {} Correspondence score: {}".format(i, result_ransac.fitness)
        results.append((result_ransac, result_ransac.fitness, i))

        draw_registration_result(source, target, result_ransac.transformation)

        if fitness == 1.0:
            break


    best_ransac, _, i = max(results, key=lambda r: r[1])

    result_icp = refine_registration(source, target, source_fpfh, target_fpfh,
                                     voxel_size, best_ransac)
    print "Final fitness: ", result_icp.fitness
    draw_registration_result(source, target, result_icp.transformation)

    return result_icp.transformation, src_pnt, target_pnt


def get_two_pc(n=2000):
    gp = GeneratePointcloud()

    pnts1 = gp.get_random_ply(n)
    pc1 = ToolPointCloud(pnts1)

    pnts2 = pc1.get_pc_bb_axis_frame_centered()

    x_range = pnts2[0].max() - pnts2[0].min()
    voxel_size = x_range / 68.

    print "VOXEL SIZE: ", voxel_size

    means = pnts2.mean(axis=0)
    x_mean = means[0]
    z_mean = means[2]
    # pnts2 = pnts2[np.where(pnts2[:,2] < z_mean), :][0]
    # pnts2 = pnts2[np.where(pnts2[:,0] < x_mean), :][0]
    pnts2 -= pnts2.mean()



    return pnts1, pnts2, voxel_size




def _center_and_align_pnts(pc):
    """
    Creates a centered and aligned ToolPointCloud from unaligned ToolPointCloud
    """
    pnts = pc.get_pc_bb_axis_frame_centered()

    return ToolPointCloud(pnts)

if __name__ == "__main__":
    # voxel_size = 0.01  # means 5cm for the dataset

    n_points = 600000
    n_iters  = 3
    # pc_registration("./tool_files/rake.ply", "./tool_files/test.ply",
    #                 n_points, n_iters)
    pc_registration("./tool_files/rake.ply", "./tool_files/rake_side_view.ply",
                    n_points, n_iters)
    # pc_registration("./tool_files/rake.ply", "./tool_files/partial_rake.ply",
    #                 n_points, n_iters)
    # pc_registration("./tool_files/partial_rake.ply", "./tool_files/rake.ply",
                    # n_points, n_iters)
    # pc_registration("./tool_files/test_mesh_norm.ply", "./tool_files/background_mesh_norm.ply",
                    # n_points, n_iters)
    # pc_registration("./tool_files/tool.pcd", "./tool_files/asc_tool.pcd", n)
