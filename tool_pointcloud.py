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
        self.bb = None
        self._normalize_pointcloud()
        self.bounding_box()
    
    def get_bb(self):
        return self.bb
    
    def get_axis(self):
        return self.bb.get_normalized_axis()
    
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