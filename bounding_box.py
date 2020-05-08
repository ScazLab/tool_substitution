#!/usr/bin/env python

import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from scipy.spatial import ConvexHull

from util import normalize_vector, close_to

class BoundingBox(object):
    """Creates bounding box around pointcloud data"""
    def __init__(self, pnts):
        "Generates 2D or 3D bounding box around points"
        self.pnts = pnts
        self.axis = None
        self.projection_axis = None
        self.norm_axis = None
        
        self.eps = 0.05 # Error term for deciding best bounding box
        
        self.bb3D = None
    
    def set_axis(self, axis = None):
        """
        The axis should be n by 3 (each row is an axis)
        """
        if axis is None:
            pca = PCA(n_components=3)
            self.axis = pca.fit(self.pnts).components_
        else:
            self.axis = axis.copy()
    
    def set_projection_axis(self, projection_index, norm_index):
        if len(projection_index) < 2:
            raise Exception("The number of projection indices should be >= 2")

        self.projection_axis = self.axis[projection_index, :].copy()
        self.norm_axis = normalize_vector(self.axis[norm_index, :].copy())
    
    def calculate_bounding_box(self):
        axis_3d = self.projection_axis
        
        pnts_2d = self._3D_to_2D()
        bb2D = self._mbb2D(pnts_2d)
        
        b3 = self.norm_axis
        
        pnts_1d = self.pnts.dot(b3)
    
        # initial 2D bb is located in the center of the point cloud.
        # We want to copy it and shift both in the directions of 3rd PC
    
        width = pnts_1d.max()  - pnts_1d.min()
        width_dir = b3 * width / 2.0 # This tells us how to translate points
    
        # width_dir = b3 * pca3D.singular_values_[2] / 2
    
        # bb3D = self.pca2D.inverse_transform(bb2D)
        print(bb2D)
        bb3D = self._2D_to_3D(bb2D)
    
        bb_side1 = bb3D + width_dir
        bb_side2 = bb3D - width_dir
    
        self.bb3D = np.vstack([bb_side1, bb_side2])
    
    def bounding_box(self):
        return self.bb3D
    
    def _3D_to_2D(self):
        """
        Projects the 3D point cloud onto the projection axis.
        """
        return np.dot(self.pnts, self.projection_axis.T)
    
    def _2D_to_3D(self, pnts):
        return np.dot(pnts.T, self.projection_axis)
    
    def _mbb2D(self, pnts):
        """
        Analytically fit a best minimum bounding box.

        @pnts: A 2D set of points.
        """

        pca_2D = PCA(n_components=2)
        pca_2D.fit(pnts)
        pca_axis = pca_2D.components_        

        pca_bb = self._2D_axis_to_bb(pnts, pca_axis)
        
        pca_projected_pnts = np.matmul(np.linalg.inv(pca_axis), pnts.T)
        
        combined_pca_projected_pnts = self._2D_get_point_reflection(pca_projected_pnts, pca_bb)     
        
        combined_min_axis = self._2D_min_bb_axis(combined_pca_projected_pnts)

        combined_min_principle = combined_min_axis[0, 0]
        
        combined_axis = None
        if close_to(combined_min_principle, 0, error=0.15) or close_to(abs(combined_min_principle), 1, error=0.15):
            #print("take original pca result")
            combined_axis = pca_axis.copy()
        else:
            #print("new result")
            combined_axis = np.matmul(pca_axis, combined_min_axis)        
        
        bb = self._2D_axis_to_bb(pnts, combined_axis)
        
        return bb
    
    def _2D_axis_to_bb(self, pnts, axis):
        projection_x = np.matmul(np.linalg.inv(axis)[0, :], pnts.T)
        projection_y = np.matmul(np.linalg.inv(axis)[1, :], pnts.T)
    
        project_x_min, project_x_max = np.min(projection_x), np.max(projection_x)
        project_y_min, project_y_max = np.min(projection_y), np.max(projection_y)
    
        projected_boundary = np.array([[project_x_min, project_y_min],
                                       [project_x_min, project_y_max],
                                       [project_x_max, project_y_max],
                                       [project_x_max, project_y_min],
                                       [project_x_min, project_y_min]])
    
        bb = np.matmul(axis, projected_boundary.T)
        
        return bb
    
    def _2D_symmetric_axis(self, pnt, pca_axis, pca_bb):
        pca_projected = np.matmul(np.linalg.inv(pca_axis), pnts.T)
        pca_project_x_min, pca_project_x_max = pca_bb[0, 0], pca_bb[2, 0]
        pca_project_y_min = pca_bb[0, 1]
        adjusted_origin = np.array([(pca_project_x_min + pca_project_x_max) / 2, pca_project_y_min])
    
        adjusted_pca_projected = pca_projected.copy()
        adjusted_pca_projected[0] -= adjusted_origin[0]
        adjusted_pca_projected[1] -= adjusted_origin[1]      

        symmetric_pca_projected = adjusted_pca_projected.copy() * -1.0
        combined_pca_projected = np.hstack([adjusted_pca_projected, symmetric_pca_projected]).T
    
        combined_min_axis = self._2D_min_bb_axis(combined_pca_projected)
    
        return combined_min_axis
    
    def _2D_get_point_reflection(self, pnts, bb):
        project_x_min, project_x_max = bb.T[0, 0], bb.T[2, 0]
        project_y_min = bb.T[0, 1]
        adjusted_origin = np.array([(project_x_min + project_x_max) / 2, project_y_min])
    
        adjusted_pnts = pnts.copy()
        adjusted_pnts[0] -= adjusted_origin[0]
        adjusted_pnts[1] -= adjusted_origin[1]      
    
        symmetric_pnts = adjusted_pnts.copy() * -1.0
        combined_pnts = np.hstack([adjusted_pnts, symmetric_pnts])        
        
        combined_pnts[0] += adjusted_origin[0]
        combined_pnts[1] += adjusted_origin[1]
        
        return combined_pnts.T
    
    def _2D_min_bb_axis(self, pnts):
        """
        Get the bounding box of the min area. 
        TODO: get average rather than the absolute min area
        """
        
        hull_pnts           = pnts[ConvexHull(pnts).vertices]
        connected_hull_pnts = np.vstack([hull_pnts, hull_pnts[0,:]])
        # The direction of all the edges
        edges = connected_hull_pnts[1:] - connected_hull_pnts[:-1]
    
        b1 = normalize(edges, axis=1, norm='l2') # First axes
        b2 = np.vstack([-b1[:,1], b1[:, 0]]).T # Orthonomral axes
    
        print("b1 shape: {}, b2 shape: {}".format(b1.shape, b2.shape))
        # print("b1 and b2 orhtonormal?: {}".format(b1[4,:].dot(b2[4,:])))
    
        # Get extremes along each axis
        print("Hull pnts shape: ", hull_pnts.shape)
        x = np.apply_along_axis(lambda col: np.array([col.min(), col.max()]),
                                    arr=hull_pnts.dot(b1.T),axis=0)
        y = np.apply_along_axis(lambda col: np.array([col.min(), col.max()]),
                                    arr=hull_pnts.dot(b2.T),axis=0)
    
        areas      = (y[0, :] - y[1, :]) * (x[0, :] - x[1, :])
        smallest_areas_idx = np.where(areas == np.min(areas))[0]
    
        print "smallest_areas_idx: ", smallest_areas_idx
    
        x = b1[smallest_areas_idx, :][0]
        y = b2[smallest_areas_idx, :][1]

        return np.array([x, y])
    
    def visualize(self, n="3D"):
        """Visualize points and bounding box"""

        fig = plt.figure()

        if n is "3D":
            ax = fig.add_subplot(111, projection='3d')
            ax.axis('equal')
            #x = bb[0] - bb[1]
            #y = bb[0] - bb[3]
            #z = bb[0] - bb[5]
            #print x / np.linalg.norm(x)
            #print y / np.linalg.norm(y)
            #print z / np.linalg.norm(z)
            #print "x, y", np.dot(x, y)
            #print "x, z", np.dot(x, z)
            #print "y, z", np.dot(y, z)
            ax.scatter(xs=self.pnts[:,0], ys=self.pnts[:,1], zs=self.pnts[:,2], c='b')
            ax.scatter(xs=self.bb3D[:,0], ys=self.bb3D[:,1], zs=self.bb3D[:,2], c='r', s=100)
            
            ax.plot(self.bb3D.T[0], self.bb3D.T[1], self.bb3D.T[2], c='r')
            ax.plot(self.bb3D[(1, 6), :].T[0], self.bb3D[(1, 6), :].T[1], self.bb3D[(1, 6), :].T[2], c='r')
            ax.plot(self.bb3D[(2, 7), :].T[0], self.bb3D[(2, 7), :].T[1], self.bb3D[(2, 7), :].T[2], c='r')
            ax.plot(self.bb3D[(3, 8), :].T[0], self.bb3D[(3, 8), :].T[1], self.bb3D[(3, 8), :].T[2], c='r')
            
        elif n is "2D":
            pnts = self._3D_to_2D()
            bb = self._mbb2D(pnts).T
            ax = fig.add_subplot(111)
            ax.axis('equal')

            ax.scatter(x=pnts[:, 0], y=pnts[:, 1], c='b')
            
            ax.scatter(x=bb[0, 0], y=bb[0, 1], c='r')
            ax.scatter(x=bb[1, 0], y=bb[1, 1], c='r')
            ax.scatter(x=bb[2, 0], y=bb[2, 1], c='r')
            ax.scatter(x=bb[3, 0], y=bb[3, 1], c='r')

            ax.plot(bb.T[0], bb.T[1], c='r')            

        plt.show()
    