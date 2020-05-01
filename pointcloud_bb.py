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

    def _project_on_to_pcs(self, components):
        pass

    def _inverse_pc_transform(self, axis):
        # Meiying
        pass

    def main(self):
        # Meiying
        found_box = False
        current_axis = [self.pca3D.components_[0,:], self.pca3D.components_[1,:], self.pca3D.components_[2,:]]
        While not found_box:
            pass
    
    def _bounding_box_to_axis(self, bb):
        # bb is 10 by 3. The 5th and 10th one is a repetition
        pass


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

