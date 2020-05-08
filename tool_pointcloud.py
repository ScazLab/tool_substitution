#!/usr/bin/env python

import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from scipy.spatial import ConvexHull

from bounding_box import BoundingBox

class ToolPointCloud(object):
    """Creates bounding box around pointcloud data"""
    def __init__(self, pnts):
        "Point cloud of the tool"
        self.pnts = pnts
        self.eps = 0.05 # Error term for deciding best bounding box
        self.mean = None
        self._normalize_pointcloud()

    def _normalize_pointcloud(self):
        self.mean = self.pnts.mean(axis = 0)
        self.pnts -= self.mean        

    def _is_same_bounding_boxes(self, bounding_boxes):
        return _close_to(np.linalg.norm(bounding_boxes - bounding_boxes[:, 0]), 0, error=-0.01)

    def _choose_bounding_box(self, boxes):
        dtype = [('index', 'i4'), ('area', 'f8'), ('parameter', 'f8')]
        data = np.array([(i, self._get_bounding_box_area(boxes[:, :, i]), self._get_bounding_box_parameter(boxes[:, :, i])) for i in range(3)], dtype = dtype)

        min_area_bbs = data[data['area'] < data['area'][0] * 1.05]
        min_area_bbs.sort(order='parameter')

        index = min_area_bbs['index'][0]

        return boxes[:, :, index]

    def bounding_box(self):
        """
        get the bounding box of the point cloud
        TODO: add correction
        """
        found_box = False
        box = None
        current_axis = None

        box_1 = BoundingBox(self.pnts)
        box_1.set_axis(current_axis)
        box_1.set_projection_axis([0, 1], 2)
        box_1.calculate_bounding_box()
        box_1.visualize("2D")
        box_1.visualize("3D")
        
        return box_1.bounding_box()
        
        #while not found_box:
            #box_1 = BoundingBox(self.pnts).get_bounding_box([0, 1], 2, current_axis)
            #box_2 = BoundingBox(self.pnts).get_bounding_box([0, 2], 1, current_axis)
            #box_3 = BoundingBox(self.pnts).get_bounding_box([1, 2], 0, current_axis)
            #boxes = np.array([box_1, box_2, box_3])
            #found_box = self._is_same_bounding_boxes(boxes)
            #if found_box:
                #found_box = True
                #box = np.mean(boxes, axis=3) # test and fix This
            #else:
                #box = self._choose_bounding_box(boxes)
                #current_axis = self._bounding_box_to_normalized_axis(box)

        #return box, self._bounding_box_to_axis(box)

    def visualize(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.axis('equal')
        ax.scatter(xs=self.pnts[:,0], ys=self.pnts[:,1], zs=self.pnts[:,2], c='b')
        plt.show()