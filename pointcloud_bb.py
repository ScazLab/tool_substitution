#!/usr/bin/env python

import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from scipy.spatial import ConvexHull


class BoundingBox(object):
    """Creates bounding box around pointcloud data"""
    def __init__(self, pnts):
        "Generates 2D or 3D bounding box around points"
        self.pnts = pnts
        self.mean = 0. # Stores the mean of the points

        self.eps = 0.05 # Error term for deciding best bounding box

        self.pca2D = PCA(n_components=2)
        self.pca3D = PCA(n_components=3)

        self.pca2D.fit(pnts)
        self.pca3D.fit(pnts)

    def mbb2D(self):
        """
        Project 3D to 2D along first PCs and then analytically
        fit best minimum bounding box.
        """

        pnts                = self.pca2D.transform(self.pnts) #TODO Transofrm for any arbitrary axis
        hull_pnts           = pnts[ConvexHull(pnts).vertices]
        connected_hull_pnts = np.vstack([hull_pnts, hull_pnts[0,:]])
        # The direction of all the edges
        edges = connected_hull_pnts[1:] - connected_hull_pnts[:-1]

        b1 = normalize(edges, axis=1, norm='l2') # First axes
        b2 = np.vstack([-b1[:,1], b1[:, 0]]).T # Orthonomral axes

        print("b1 shape: {}, b2 shape: {}".format(b1.shape, b2.shape))
        print("b1 and b2 orhtonormal?: {}".format(b1[4,:].dot(b2[4,:])))

        # Get extremes along each axis
        print("Hull pnts shape: ", hull_pnts.shape)
        x = np.apply_along_axis(lambda col: np.array([col.min(), col.max()]),
                                arr=hull_pnts.dot(b1.T),axis=0)
        y = np.apply_along_axis(lambda col: np.array([col.min(), col.max()]),
                                arr=hull_pnts.dot(b2.T),axis=0)


        areas = (y[0, :] - y[1, :]) * (x[0, :] - x[1, :])
        # ratio of side legnths. Closer to 0 the more square it is.
        perimeters = abs(1.0 - ((y[0, :] - y[1, :]) / (x[0, :] - x[1, :])))
        print(perimeters)

        k_best = int(len(areas) * self.eps) # How many of best bbs to consider.

        # if zero, default to bb with smallest area
        if k_best == 0:
            k = np.argmin(areas) # index of points with smallest bb area
        else:
            print("k best: {}".format(k_best))
            # Get indices of k_best smallest areas
            best_areas_idx = np.argpartition(areas, k_best)[:k_best]
            print("best areas: {}".format(best_areas_idx))

            # Get corresponding perimeters and choose most square one.
            k_p = np.argmin(perimeters[best_areas_idx])
            k = best_areas_idx[k_p]

        k_a = np.argmin(areas) # index of points with smallest bb area


        print("k_a: {}".format(k_a))
        print("k: {}".format(k))
        print("x shape: {} y shape: {}".format(x.shape, y.shape))

        print(x[:, k])
        print(y[:,k])

        inverse_rot = np.vstack([b1[k,:], b2[k,:]]) # rotate back to orig coords

        rot_bb = np.array([x[[0,1,1,0,0],k],
                        y[[0,0,1,1,0],k]])

        print("orig x and y range: {}".format(np.max(hull_pnts, axis=0) - np.min(hull_pnts, axis=0)))
        print("bb x and y range: {}".format(np.max(rot_bb, axis=1) - np.min(rot_bb, axis=1)))
        print(rot_bb)


        bb = inverse_rot.T.dot(rot_bb) # rotate points back to orig frame

        return bb.T

    def mbb2D_OLD(self):
        """
        Project 3D to 2D along first PCs and then analytically
        fit best minimum bounding box.
        """

        pnts                = self.pca2D.transform(self.pnts) #TODO Transofrm for any arbitrary axis
        hull_pnts           = pnts[ConvexHull(pnts).vertices]
        connected_hull_pnts = np.vstack([hull_pnts, hull_pnts[0,:]])
        # The direction of all the edges
        edges = connected_hull_pnts[1:] - connected_hull_pnts[:-1]

        b1 = normalize(edges, axis=1, norm='l2') # First axes
        b2 = np.vstack([-b1[:,1], b1[:, 0]]).T # Orthonomral axes

        print("b1 shape: {}, b2 shape: {}".format(b1.shape, b2.shape))
        print("b1 and b2 orhtonormal?: {}".format(b1[4,:].dot(b2[4,:])))

        # Get extremes along each axis
        print("Hull pnts shape: ", hull_pnts.shape)
        x = np.apply_along_axis(lambda col: np.array([col.min(), col.max()]),
                                arr=hull_pnts.dot(b1.T),axis=0)
        y = np.apply_along_axis(lambda col: np.array([col.min(), col.max()]),
                                arr=hull_pnts.dot(b2.T),axis=0)


        areas = (y[0, :] - y[1, :]) * (x[0, :] - x[1, :])
        perimeters = abs((y[0, :] - y[1, :]) - (x[0, :] - x[1, :]))


        print('Areas: {}'.format(sorted(areas)))

        k_p = np.argmin(perimeters)
        k_a = np.argmin(areas) # index of points with smallest bb area

        area_ratio = areas[k_a] / areas[k_p]
        print("Area ratio: {}".format(area_ratio))

        # If the most square-like bounding box has a tolerably
        # small area, we choose it, otherwise we choose box with smallest area.
        if  area_ratio >  1.0 - self.eps:
            k = k_p
        else:
            k = k_a

        print("k_a: {}".format(k_a))
        print("k_p: {}".format(k_p))
        print("x shape: {} y shape: {}".format(x.shape, y.shape))

        print(x[:, k])
        print(y[:,k])

        inverse_rot = np.vstack([b1[k,:], b2[k,:]]) # rotate back to orig coords

        rot_bb = np.array([x[[0,1,1,0,0],k],
                        y[[0,0,1,1,0],k]])

        print("orig x and y range: {}".format(np.max(hull_pnts, axis=0) - np.min(hull_pnts, axis=0)))
        print("bb x and y range: {}".format(np.max(rot_bb, axis=1) - np.min(rot_bb, axis=1)))
        print(rot_bb)


        bb = inverse_rot.T.dot(rot_bb) # rotate points back to orig frame

        return bb.T

    def _transform(self, pnts, components, norm=True):
        """
        Projects @pnts onto @components bases.
        """

        self.bases = components # Store bases in order to do inverse transform.

        if norm:
            self.mean = pnts.mean(axis=1)
            pnts -= self.mean

        return np.dot(pnts, components.T)

    def _inverse_transform(self, pnts):

        return np.dot(pnts, self.bases) + self.mean

    def _inverse_pc_transform(self, bb2D, b3):
        # Meiying
        b3 = b3 / np.linalg.norm(b3) # normalize

        pnts_1d = self.pnts.dot(b3)

        # initial 2D bb is located in the center of the point cloud.
        # We want to copy it and shift both in the directions of 3rd PC

        width = pnts_1d.max()  - pnts_1d.min()
        width_dir = b3 * width / 2.0 # This tells us how to translate points

        # width_dir = b3 * pca3D.singular_values_[2] / 2

        bb3D = self.pca2D.inverse_transform(bb2D)

        bb_side1 = bb3D + width_dir
        bb_side2 = bb3D - width_dir

        bb3D = np.vstack([bb_side1, bb_side2])

        return bb3D

    def _bounding_box_to_normalized_axis(self, bb):
        # Meiying
        # bb is 10 by 3. The 5th and 10th one is a repetition
        unnormialzied_axis = self._bounding_box_to_axis(bb)
        return unnormialzied_axis / np.linalg.norm(unnormialzied_axis, axis=1, keepdims=True)

    def _bounding_box_to_axis(self, bb):
        # Meiying
        # bb is 10 by 3. The 5th and 10th one is a repetition
        dtype = [('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('length', 'f8')]
        axis_with_length = np.array([(bb[0, i], bb[1, i], bb[2, i], np.linalg.norm(bb[:, i])) for i in range(3)], dtype = dtype)
        axis_with_length.sort(order = 'length')
        return axis_with_length[:, :3].T

    def _get_bounding_box(self, axis, projection_indices, other_index):
        # Meiying
        pnts = self._transform(self.pca3D.components_[indices[(projection_indices], indices[projection_indices]), :]) # test and fix this
        bb2D = self.mbb2D(pnts)
        bb3D = self._inverse_pc_transform(bb2D, self.pca3D.components_[other_index, :])
        return bb3D

    def _get_bounding_box_area(self, bounding_box):
        # Meiying
        axes = self._bounding_box_to_axis(bounding_box)
        return np.linalg.norm(axes[:, 0]) * np.linalg.norm(axes[:, 1]) * np.linalg.norm(axes[:, 2])

    def _close_to(m, n, error=1e-6):
        return m >= n - error and m <= n + error

    def _get_bounding_box_parameter(self, bounding_box):
        # Meiying
        axes = self._bounding_box_to_axis(bounding_box)
        return 2 * (np.linalg.norm(axes[:, 0]) + np.linalg.norm(axes[:, 1]) + np.linalg.norm(axes[:, 2]))

    def _is_same_bounding_boxes(self, bounding_boxes):
        self._close_to(np.linalg.norm(bounding_boxes - bounding_boxes[:, 0]), 0, error=-0.01)

    def _choose_bounding_box(self, boxes):
        dtype = [('index', 'i4'), ('area', 'f8'), ('parameter', 'f8')]
        data = np.array([(i, self._get_bounding_box_area(boxes[:, :, i]), self._get_bounding_box_parameter(boxes[:, :, i])) for i in range(3)], dtype = dtype)

        min_area_bbs = data[data['area'] < data['area'][0] * 1.05]
        min_area_bbs.sort(order='parameter')

        index = min_area_bbs['index'][0]

        return boxes[:, :, index]

    def main(self):
        # Meiying
        found_box = False
        box = None
        #current_axis = [self.pca3D.components_[0,:], self.pca3D.components_[1,:], self.pca3D.components_[2,:]]
        while not found_box:
            box_1 = self._get_bounding_box(current_axis, [0, 1], 2)
            box_2 = self._get_bounding_box(current_axis, [0, 2], 1)
            box_3 = self._get_bounding_box(current_axis, [1, 2], 0)
            boxes = np.array([box_1, box_2, box_3])
            found_box = self._is_same_bounding_boxes(boxes)
            if found_box:
                found_box = True
                box = np.mean(boxes, axis=3) # test and fix This
            else:
                box = self._choose_bounding_box(boxes)
                current_axis = self._bounding_box_to_normalized_axis(box)

        return box, self._bounding_box_to_axis(box)


    def mbb3D(self):
        """
        Estimate vertices of 3d bounding box using 2d bb and 3rd P.C.
        """

        bb2D = self.mbb2D()

        b3 = self.pca3D.components_[2,:] # 3rd PC
        b3 = b3 / np.linalg.norm(b3) # normalize

        pnts_1d = self.pnts.dot(b3)

        # initial 2D bb is located in the center of the point cloud.
        # We want to copy it and shift both in the directions of 3rd PC

        width = pnts_1d.max()  - pnts_1d.min()
        width_dir = b3 * width / 2.0 # This tells us how to translate points

        # width_dir = b3 * pca3D.singular_values_[2] / 2

        bb3D = self.pca2D.inverse_transform(bb2D)

        bb_side1 = bb3D + width_dir
        bb_side2 = bb3D - width_dir

        bb3D = np.vstack([bb_side1, bb_side2])

        return bb3D

    def _visualize_point_cloud(self, fig, axis):
        axis.scatter(xs=self.pnts[:,0], ys=self.pnts[:,1], zs=self.pnts[:,2], c='b')

    def _visualize_bounding_box(self, fig, axis, bb, color='r'):
        ax.scatter(xs=bb[:,0], ys=bb[:,1], zs=bb[:,2], c=color, s=100)

    def plot_bb(self, n="3D", ax=None):
        """Visualize points and bounding box"""

        fig = plt.figure()

        if n is "3D":
            # TODO: Connect vertices of 3D bounding box.

            ax = fig.add_subplot(111, projection='3d')
            bb = self.mbb3D()
            ax.scatter(xs=self.pnts[:,0], ys=self.pnts[:,1], zs=self.pnts[:,2], c='b')
            ax.scatter(xs=bb[:,0], ys=bb[:,1], zs=bb[:,2], c='r', s=100)

        elif n is "2D":

            bb = self.mbb2D()
            ax = fig.add_subplot(111)

            pnts = self.pca2D.transform(self.pnts)
            ax.scatter(x=pnts[:, 0], y=pnts[:,1], c='b')
            ax.plot(bb[[0,1], 0], bb[[0,1],1],c='r')
            ax.plot(bb[[1,2], 0], bb[[1,2],1],c='r')
            ax.plot(bb[[2,3], 0], bb[[2,3],1],c='r')
            ax.plot(bb[[3,0], 0], bb[[3,0],1],c='r')


        plt.show()
