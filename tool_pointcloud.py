#!/usr/bin/env python

import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from scipy.spatial import ConvexHull

from bounding_box import BoundingBox3D
from util import close_to

class ToolPointCloud(object):
    """Creates bounding box around pointcloud data"""
    def __init__(self, pnts, eps=0.001):
        "Point cloud of the tool"
        self.pnts = pnts
        self.eps = eps # Error term for deciding best bounding box
        self.mean = None
        self.bb = None # 10 by 3, the 5th and 10th is the reptead point
        self._normalize_pointcloud()
        self.bounding_box()
        self.aruco_frame = None # in unnormalized_pc frame
    
    def get_bb(self):
        return self.bb
    
    """
    The aruco_frame related functions are yet to be integrated with the rest of this class.
    Code needs to be refactored
    """
    def set_aruco_frame(self, aruco_frame):
        self.aruco_frame = aruco_frame
    
    def set_aruco_frame_with_four_corners(self, corners, aruco_size, aruco_id):
        # corners are in the scanned object frame, not the world frame when run the experiment
        # The corners are in the order of: TBD
        # aruco size is in meters
        
        pass
    """
    aruco related functions finished
    """
    
    def get_axis(self):
        return self.bb.get_normalized_axis()
    
    def get_unnormalized_pc(self):
        return self.pnts + self.mean
    
    def get_normalized_pc(self):
        return self.pnts
    
    def get_pc_aruco_frame(self):
        unnomalized_pc = self.get_unnormalized_pc()
        return np.matmul(np.linalg.inv(self.aruco_frame), unnomalized_pc)
    
    def get_pc_bb_axis_frame(self):
        return np.matmul(np.linalg.inv(self.get_axis()), self.pnts)
    
    def get_pc_bb_axis_frame_centered(self):
        pc_bb_axis_frame = self.get_pc_bb_axis_frame()
        bb_trimed = self.bb.copy()
        bb_trimed = np.delete(bb_trimed, np.s_[4], axis=0)
        bb_trimed = np.delete(bb_trimed, np.s_[-1], axis=0)
        bb_centroid = np.mean(bb_trimed, axis=0)
        return pc_bb_axis_frame - bb_centroid
    
    def bb_2d_projection(self, projection_index, norm_index, visualize=True):
        # returns both the pc and the 2D bb
        return self.bb.bb_2d_projection(projection_index, norm_index, visualize)

    def _normalize_pointcloud(self):
        self.mean = self.pnts.mean(axis = 0)
        self.pnts -= self.mean

    def bounding_box(self):
        """
        get the bounding box of the point cloud
        TODO: add correction
        """
        found_box = False
        current_axis = None
        result_box = None
        max_loop = 10
        i = 0
        box = None
        
        while not found_box and i < max_loop:
            vols = []
            bbs = []
            for [projection_axis_index, norm_axis_index] in [[[0, 1], 2], [[0, 2], 1], [[1, 2], 0]]:
                print "projection index: ", projection_axis_index
                print "norm_axis_index: ", norm_axis_index
                bb = self._get_bb_helper(current_axis, projection_axis_index, norm_axis_index)
                vols.append(bb.volumn())
                bbs.append(bb)
                print "axis: "
                print bb.get_normalized_axis()
            print "volumnes: ", vols
            max_vol, min_vol = max(vols), min(vols)
            if close_to(max_vol / min_vol, 1, self.eps):
                found_box = True
                self.bb = bbs[vols.index(min(vols))]
            else:
                bb = bbs[vols.index(min(vols))]
                current_axis = bb.get_normalized_axis()
                print "new current axis is"
                print current_axis
            print "=================================================="
                
            i += 1

        print "final round: ", i
        print "current axis"
        print current_axis

    def _get_bb_helper(self, axis, projection_axis_index, norm_axis_index):
        box = BoundingBox3D(self.pnts)
        box.set_axis(axis)
        box.set_projection_axis(projection_axis_index, norm_axis_index)
        return box

    def visualize(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.axis('equal')
        ax.scatter(xs=self.pnts[:, 0], ys=self.pnts[:, 1], zs=self.pnts[:, 2], c='b')
        plt.show()
    
    def visualize_bb(self):
        self.bb.visualize("3D")
        self.bb.visualize("2D")