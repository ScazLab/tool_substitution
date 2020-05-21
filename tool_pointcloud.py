#!/usr/bin/env python

import cv2
import math
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
        self.aruco_frame = {} # there are multiple arucos on an object. in unnormalized_pc frame
        self.scale = 0 # The scale used to scale to unit in meters.
        self.scales = []
        self.is_scaled = False

        self._normalize_pointcloud()
        self.bounding_box()

    def scale_pnts_to_target(self, target_tpc, keep_proportional=False):
        """
        Scale points to match the dims of a target tool pointcloud.
        If keep_proportional == True, volumns of bbs will be scaled while
        keeping side proportions constant.
        """

        src_dim_lens   = self.bb.dim_lens
        target_dim_lens = target_tpc.bb.dim_lens

        if keep_proportional:
            # Get volumns of both bbs
            targ_vol =  target_dim_lens[0] * target_dim_lens[1] * target_dim_lens[2]
            src_vol = src_dim_lens[0] * src_dim_lens[1] * src_dim_lens[2]

            # Get ratio of columns
            scale_val = targ_vol / src_vol
            # Proportionally scale src pc based on this ratio
            scale_factor = math.pow(scale_val, 1.0/3.0)


        else:
            scale_factor = target_dim_lens / src_dim_lens

        # Scale points.
        self.pnts *= scale_factor
        self.bb.scale_bb(scale_factor)

    def get_bb(self):
        return self.bb

    """
    The aruco_frame related functions are yet to be integrated with the rest of this class.
    Code needs to be refactored
    """
    #def set_aruco_frame(self, aruco_frame):
        #self.aruco_frame = aruco_frame

    def set_scale(self, model_length, actual_length):
        if model_length == 0 or actual_length == 0:
            raise Exception("either the model length of the actual length is 0!! Cannot determin the scale")

        scale = actual_length / model_length

        if self.scale == 0: # not set yet
            self.scale = scale
        else:
            if close_to(self.scale, scale):
                self.scales.append(scale)
            else:
                raise Exception("Current scale {} is very different from saved scale {}".format(scale, self.scale))

        self.scale = sum(self.scales) / len(self.scales)

    def scale(self):
        self.pnts *= self.scale
        self.mean *= self.scale
        self.bounding_box()

        for frame in self.aruco_frame.values():
            frame[:-1, 3] *= self.scale

        self.is_scaled = True

    def set_aruco_frame_with_four_corners(self, corners, aruco_size, aruco_id):
        # corners are in the scanned object frame, not the world frame when run the experiment
        # https://docs.opencv.org/trunk/d5/dae/tutorial_aruco_detection.html
        # https://github.com/pal-robotics/aruco_ros
        # The corners are in the order of:
        #      - 1. the first corner is the top left corner;
        #      - 2. followed by the top right;
        #      - 3. bottom right;
        #      - 4. and bottom left
        # X direction: 1 -> 4; 2 -> 3
        # Y direction: 1 -> 2; 4 -> 3
        # Z direction: cross produce
        # center: the centroid of the shape
        # aruco size is in meters

        if self.aruco_frame.has_key(aruco_id):
            print "aruco_id: ", aruco_id, " has already set!"
            return

        x_direction_14 = corners[3, :]  - corners[0, :]
        x_direction_23 = corners[2, :]  - corners[1, :]

        x_direction_14_length = np.linalg.norm(x_direction_14)
        x_direction_23_length = np.linalg.norm(x_direction_23)

        x_direction_14 = normalize(x_direction_14)
        x_direction_23 = normalize(x_direction_23)

        if not close_to(np.dot(x_direction_14, x_direction_23), 1):
            raise Exception("The corners of the markers are not right. The line of 1(top left)-4(bottom left) and 2(top right)-3(bottom right) are not parallel.")
        if not close_to(x_direction_14_length, x_direction_23_length):
            raise Exception("The corners of the markers are not right. The length of 1(top left)-4(bottom left) and 2(top right)-3(bottom right) are not the same.")

        x_length = (x_direction_14_length, x_direction_23_length) / 2
        x_direction = (x_direction_14 + x_direction_23) / 2
        x_direction = normalize(x_direction)

        y_direction_12 = corners[1, :]  - corners[0, :]
        y_direction_43 = corners[2, :]  - corners[3, :]

        y_direction_12_length = np.linalg.norm(y_direction_12)
        y_direction_43_length = np.linalg.norm(y_direction_43)

        y_direction_12 = normalize(y_direction_12)
        y_direction_43 = normalize(y_direction_43)

        if not close_to(np.dot(y_direction_12, y_direction_43), 1):
            raise Exception("The corners of the markers are not right. The line of 1(top left)-2(top right) and 4(bottom left)-3(bottom right) are not parallel.")
        if not close_to(y_direction_12_length, y_direction_43_length):
            raise Exception("The corners of the markers are not right. The length of 1(top left)-2(top right) and 4(bottom left)-3(bottom right) are not the same.")

        y_length = (y_direction_12_length, y_direction_43_length) / 2
        y_direction = (y_direction_12 + y_direction_43) / 2
        y_direction = normalize(y_direction)

        if not close_to(np.dot(x_direction, y_direction), 0):
            raise Exception("The corners of the markers are not right. x and y are not perpendicular.")
        if not close_to(x_length, y_length):
            raise Exception("The corners of the markers are not right. The length of x and y are not the same.")

        z_direction = np.cross(x_direction, y_direction)
        z_direction = normalize(z_direction)

        centroid = np.mean(corners, axis=0)

        aruco_frame = np.vstack([x_direction, y_direction, z_direction, centroid]).T
        aruco_frame = np.vstack([aruco_frame], np.array([0, 0, 0, 1]))

        self.aruco_frame[aruco_id] = aruco_frame


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
        return np.matmul(np.linalg.inv(self.get_axis()), self.pnts.T).T
    
    def get_pc_bb_axis_frame_centered(self):
        pc_bb_axis_frame = self.get_pc_bb_axis_frame()
        bb_trimed = self.bb.bb.copy()
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
        bbs = []
        self.bb = None

        while not found_box:
            vols = []
            bbs = []
            for [projection_axis_index, norm_axis_index] in [[[0, 1], 2], [[0, 2], 1], [[1, 2], 0]]:
                #print "projection index: ", projection_axis_index
                #print "norm_axis_index: ", norm_axis_index
                bb = self._get_bb_helper(current_axis, projection_axis_index, norm_axis_index)
                vols.append(bb.volumn())
                bbs.append(bb)
                #print "axis: "
                #print bb.get_normalized_axis()
            #print "volumnes: ", vols
            max_vol, min_vol = max(vols), min(vols)
            print "max_vol: ", max_vol
            print "min_vol: ", min_vol
            print "ratio: ", max_vol / min_vol
            if close_to(max_vol / min_vol, 1, self.eps):
                found_box = True
                #self.bb = bbs[vols.index(min(vols))]
            #else:
                ##bb = bbs[vols.index(min(vols))]
                #current_axis = bb.get_normalized_axis()

            self.bb = bbs[vols.index(min(vols))]
            current_axis = self.bb.get_normalized_axis()
            #print "new current axis is"
            #print current_axis
            print "=================================================="

            i += 1
        
        #for bb in bbs:
            #bb.visualize("2D")
            #bb.visualize("3D")
            
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
