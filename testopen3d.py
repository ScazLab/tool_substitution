#!/usr/bin/env python

import open3d as o3d
#from open3d.open3d.geometry import select_by_index
#import numpy as np
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

pcd = o3d.io.read_point_cloud("./tool_files/chineseknife_1_3dwh_out_2_50_fused.ply")
#pc = mesh.sample_points_uniformly(number_of_points=7000)
print(pcd.points)
#o3d.visualization.draw_geometries([pcd])
plane_model, inliers = pcd.segment_plane(distance_threshold=0.001,
                                         ransac_n=100,
                                         num_iterations=1000)
[a, b, c, d] = plane_model
print("Plane equation: {:.2f}x + {:.2f}y + {:.2f}z + {:.2f} = 0".format(a, b, c, d))
print(len(inliers))
inlier_cloud = pcd.select_by_index(inliers)
#inlier_cloud.paint_uniform_color([1.0, 0, 0])
#outlier_cloud = pcd.select_by_index(inliers, invert=True)
#o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud], zoom=0.8, 
                                  #front=[-0.4999, -0.1659, -0.8499],
                                  #lookat=[2.1813, 2.0619, 2.0999],
                                  #up=[0.1204, -0.9852, 0.1215])


# visualize
#print("Try to render a mesh with normals (exist: " +
      #str(mesh.has_vertex_normals()) + ") and colors (exist: " +
      #str(mesh.has_vertex_colors()) + ")")
#o3d.visualization.draw_geometries([mesh])
#print("A mesh with no normals and no colors does not seem good.")

#points = np.asarray(o3d.utility.Vector3dVector(mesh.vertices))
#print points.shape
#rotation_matrix = np.array([[1, 0, 0], 
                            #[0, np.cos(np.pi/4), -np.sin(np.pi/4)], 
                            #[0, np.sin(np.pi/4), np.cos(np.pi/4)]])
#points = np.matmul(rotation_matrix, points.T).T
#points = o3d.utility.Vector3dVector(points)
#bb = o3d.geometry.OrientedBoundingBox().create_from_points(points).get_oriented_bounding_box()
#print np.asarray(bb.get_box_points())

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
##ax.axis('equal')

#pnts = np.asarray(points)
#bb_raw = np.asarray(bb.get_box_points())
##print bb_raw[:4, :]
##print bb_raw[0, :]
##print bb_raw[:5, :]
##print bb_raw[-1, :]
#bb = np.vstack([bb_raw[:4, :], bb_raw[0, :], bb_raw[4:, :], bb_raw[4, :]])
#print "bb"
#print bb
#ax.scatter(xs=pnts[:,0], ys=pnts[:,1], zs=pnts[:,2], c='b')
#ax.scatter(xs=bb[:,0], ys=bb[:,1], zs=bb[:,2], c='r', s=100)

##ax.plot(bb.T[0], bb.T[1], bb.T[2], c='r')
##ax.plot(bb[(1, 6), :].T[0], bb[(1, 6), :].T[1], bb[(1, 6), :].T[2], c='r')
##ax.plot(bb[(2, 7), :].T[0], bb[(2, 7), :].T[1], bb[(2, 7), :].T[2], c='r')
##ax.plot(bb[(3, 8), :].T[0], bb[(3, 8), :].T[1], bb[(3, 8), :].T[2], c='r')

#plt.show()