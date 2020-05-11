#!/usr/bin/env python

import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from scipy.spatial import ConvexHull


class BoundingBox(object):
    """Creates bounding box around pointcloud data"""
    def __init__(self, pnts, eps=0.05):
        "Generates 2D or 3D bounding box around points"
        self.pnts = pnts
        self.mean = 0. # Stores the mean of the points

        self.eps = eps # Error term for deciding best bounding box

        self.pca2D = PCA(n_components=2)
        self.pca3D = PCA(n_components=3)

        self.pca2D.fit(pnts)
        self.pca3D.fit(pnts)
    
    def get_smallest_bounding_box_axis(self, pnts):
        hull_pnts           = pnts[ConvexHull(pnts).vertices]
        connected_hull_pnts = np.vstack([hull_pnts, hull_pnts[0,:]])
        # The direction of all the edges
        edges = connected_hull_pnts[1:] - connected_hull_pnts[:-1]
    
        b1 = normalize(edges, axis=1, norm='l2') # First axes
        b2 = np.vstack([-b1[:,1], b1[:, 0]]).T # Orthonomral axes
        
        #print "b1"
        #print b1
        
        #print "b2"
        #print b2
    
        print("b1 shape: {}, b2 shape: {}".format(b1.shape, b2.shape))
        # print("b1 and b2 orhtonormal?: {}".format(b1[4,:].dot(b2[4,:])))
    
        # Get extremes along each axis
        print("Hull pnts shape: ", hull_pnts.shape)
        x = np.apply_along_axis(lambda col: np.array([col.min(), col.max()]),
        arr=hull_pnts.dot(b1.T),axis=0)
        y = np.apply_along_axis(lambda col: np.array([col.min(), col.max()]),
        arr=hull_pnts.dot(b2.T),axis=0)
    
        areas      = (y[0, :] - y[1, :]) * (x[0, :] - x[1, :])
        #perimeters = (y[0, :] - y[1, :]) + (x[0, :] - x[1, :])
        #perimeters = abs(1.0 - ((y[0, :] - y[1, :]) / (x[0, :] - x[1, :])))
        # Get indices of bbs with areas within eps of smallest bb.
        #smallest_areas_idx = np.where(areas < (1.0 + self.eps) * np.min(areas))[0]
        smallest_areas_idx = np.where(areas == np.min(areas))[0]
        #smallest_perims = perimeters[smallest_areas_idx]

        print "smallest_areas_idx: ", smallest_areas_idx

        x = b1[smallest_areas_idx, :][0]
        y = b2[smallest_areas_idx, :][1]
        
        print "x: ", x
        print "y: ", y
        
        return np.array([x, y])
        

    def mbb2D(self, pnts):
        """
        Project 3D to 2D along first PCs and then analytically
        fit best minimum bounding box.

        @pnts: A 2D set of points.
        """

        pca_2D = PCA(n_components=2)
        pca_2D.fit(pnts)
        pca_axis = pca_2D.components_
        
        print "pnts shape: ", pnts.shape
        print "pca_axis shape: ", pca_axis.shape
        print "pca_axis: "
        print pca_axis
        
        pca_projection_x = np.matmul(np.linalg.inv(pca_axis)[0, :], pnts.T)
        pca_projection_y = np.matmul(np.linalg.inv(pca_axis)[1, :], pnts.T)

        pca_projection_x.sort()
        pca_projection_y.sort()

        pca_project_x_min, pca_project_x_max = np.min(pca_projection_x), np.max(pca_projection_x)
        pca_project_y_min, pca_project_y_max = np.min(pca_projection_y), np.max(pca_projection_y)

        pca_projected_boundary = np.array([[pca_project_x_min, pca_project_y_min],
                                           [pca_project_x_min, pca_project_y_max],
                                           [pca_project_x_max, pca_project_y_max],
                                           [pca_project_x_max, pca_project_y_min],
                                           [pca_project_x_min, pca_project_y_min]])

        print "pca_projected_boundary is"
        print pca_projected_boundary

        pca_boundary = np.matmul(pca_axis, pca_projected_boundary.T)
        
        pca_projected = np.matmul(np.linalg.inv(pca_axis), pnts.T)
        
        print "=======================================pca_projected"
        print pca_projected
        
        adjusted_origin = np.array([(pca_project_x_min + pca_project_x_max) / 2, pca_project_y_min])
        
        #print "pca_projected"
        #print pca_projected
        
        print "adjusted_origin"
        print adjusted_origin
        
        adjusted_pca_projected = pca_projected.copy()
        adjusted_pca_projected[0] -= adjusted_origin[0]
        adjusted_pca_projected[1] -= adjusted_origin[1]      
        
        #print "adjusted_pca_projected"
        #print adjusted_pca_projected
        # x - axis guaranteed to be the longer one
        symmetric_pca_projected = adjusted_pca_projected.copy() * -1.0
        combined_pca_projected = np.hstack([adjusted_pca_projected, symmetric_pca_projected]).T
        
        combined_min_axis = self.get_smallest_bounding_box_axis(combined_pca_projected)
        
        print "combined_min_axis"
        print combined_min_axis
        
        print "combined_pca_projected shape: ", combined_pca_projected.shape
        
        combined_pca_2D = PCA(n_components=2)
        combined_pca_2D.fit(combined_pca_projected)
        combined_pca_axis = combined_pca_2D.components_
        
        print "combined_pca_axis:"
        print combined_pca_axis
        
        #fig = plt.figure()
    
        #ax = fig.add_subplot(111)
        #ax.axis('equal')
        #ax.scatter(x=combined_pca_projected[:, 0], y=combined_pca_projected[:,1], c='b')
        
        ##x1 = [0, combined_pca_axis[0][0]] * 20
        ##y1 = [0, combined_pca_axis[1][0]] * 20
        ##x2 = [0, combined_pca_axis[0][1]] * 20
        ##y2 = [0, combined_pca_axis[1][1]] * 20
        #x1 = [0, combined_axis[0][0]] * 20
        #y1 = [0, combined_axis[1][0]] * 20
        #x2 = [0, combined_axis[0][1]] * 20
        #y2 = [0, combined_axis[1][1]] * 20        
        ##print "x: ", x
        ##print "y: ", y
        #ax.plot(np.array(x1), np.array(y1), c='r')
        #ax.plot(np.array(x2), np.array(y2), c='g')
    
        #plt.show()
        
        print "combined_pca_axis shape: ", combined_pca_axis.shape
        
        combined_pca_principle = combined_pca_axis[0, 0]
        #combined_min_principle = combined_min_axis[0, 0]
        
        combined_axis = None
        if self._close_to(combined_pca_principle, 0, error=0.15) or self._close_to(abs(combined_pca_principle), 1, error=0.15):
            print "take original pca result"
            combined_axis = pca_axis.copy()
        else:
            print "new result"
            combined_axis = np.matmul(pca_axis, combined_min_axis)
        
        combined_projection_x = np.matmul(np.linalg.inv(combined_axis)[0, :], pnts.T)
        combined_projection_y = np.matmul(np.linalg.inv(combined_axis)[1, :], pnts.T)
    
        combined_projection_x.sort()
        combined_projection_y.sort()
    
        combined_project_x_min, combined_project_x_max = np.min(combined_projection_x), np.max(combined_projection_x)
        combined_project_y_min, combined_project_y_max = np.min(combined_projection_y), np.max(combined_projection_y)
    
        combined_projected_boundary = np.array([[combined_project_x_min, combined_project_y_min],
                                                [combined_project_x_min, combined_project_y_max],
                                                [combined_project_x_max, combined_project_y_max],
                                                [combined_project_x_max, combined_project_y_min],
                                                [combined_project_x_min, combined_project_y_min]])
    
        print "combined_projected_boundary is"
        print combined_projected_boundary
    
        combined_boundary = np.matmul(combined_axis, combined_projected_boundary.T)
        
        print "combined boundary is"
        print combined_boundary

        all_axis = [np.array([[np.cos(i * np.pi / 180.0), -np.sin(i * np.pi / 180.0)], [np.sin(i * np.pi / 180.0), np.cos(i * np.pi / 180.0)]]) for i in range(181)]

        converted_pnts = [np.matmul(np.linalg.inv(i)[0, :], pnts.T) for i in all_axis]
        ##converted_pnts = [np.matmul(np.linalg.inv(i), pnts.T) for i in all_axis]
        length = [np.ptp(i) for i in converted_pnts]

        dtype = [('index', 'i4'), ('length', 'f8')]

        data = np.array([(i, length[i]) for i in range(len(all_axis))], dtype = dtype)

        # for i in range(len(all_axis)):
        #     print i, ": ", length[i]

        data.sort(order='length')

        chosen_indices = data[data['length'] < np.min(data['length']) * 1.05]
        chosen_indices = chosen_indices['index']

        # print "chosen_indices: "
        # print chosen_indices

        avg_indices = np.mean(chosen_indices)

        # print "avg_indices: ", avg_indices

        chosen_index = data['index'][0]

        # print "chosen_index: ", chosen_index

        axis = all_axis[chosen_index]
        ##axis = [all_axis[chosen_index] * 100]

        # print "axis is: ", axis

        projection_x = np.matmul(np.linalg.inv(axis)[0, :], pnts.T)
        projection_y = np.matmul(np.linalg.inv(axis)[1, :], pnts.T)

        a = pnts.T[0].copy()
        a.sort()

        b = pnts.T[1].copy()
        b.sort()

        #print "pnt x"
        #print a

        #print "pnt y"
        #print b

        projection_x.sort()
        projection_y.sort()

        #print "projection_x"
        #print projection_x

        #print "projection_y"
        #print projection_y

        project_x_min, project_x_max = np.min(projection_x), np.max(projection_x)
        project_y_min, project_y_max = np.min(projection_y), np.max(projection_y)

        projected_boundary = np.array([[project_x_min, project_y_min],
                                       [project_x_min, project_y_max],
                                       [project_x_max, project_y_max],
                                       [project_x_max, project_y_min],
                                       [project_x_min, project_y_min]])

        # print "projected_boundary is"
        # print projected_boundary

        boundary = np.matmul(axis, projected_boundary.T)

        axis_2 = all_axis[int(avg_indices)]
        ##axis = [all_axis[chosen_index] * 100]

        # print "axis_2 is: ", axis_2

        avg_projection_x = np.matmul(np.linalg.inv(axis_2)[0, :], pnts.T)
        avg_projection_y = np.matmul(np.linalg.inv(axis_2)[1, :], pnts.T)

        avg_projection_x.sort()
        avg_projection_y.sort()

        avg_project_x_min, avg_project_x_max = np.min(avg_projection_x), np.max(avg_projection_x)
        avg_project_y_min, avg_project_y_max = np.min(avg_projection_y), np.max(avg_projection_y)

        avg_projected_boundary = np.array([[avg_project_x_min, avg_project_y_min],
                                           [avg_project_x_min, avg_project_y_max],
                                           [avg_project_x_max, avg_project_y_max],
                                           [avg_project_x_max, avg_project_y_min],
                                           [avg_project_x_min, avg_project_y_min]])

        # print "avg_projected_boundary is"
        # print avg_projected_boundary

        avg_boundary = np.matmul(axis_2, avg_projected_boundary.T)

        # print "avg_boundary is"
        # print avg_boundary

        # print "======================================================="

        #hull_pnts           = pnts[ConvexHull(pnts).vertices]
        #connected_hull_pnts = np.vstack([hull_pnts, hull_pnts[0,:]])
        ## The direction of all the edges
        #edges = connected_hull_pnts[1:] - connected_hull_pnts[:-1]

        #b1 = normalize(edges, axis=1, norm='l2') # First axes
        #b2 = np.vstack([-b1[:,1], b1[:, 0]]).T # Orthonomral axes

        #print("b1 shape: {}, b2 shape: {}".format(b1.shape, b2.shape))
        ## print("b1 and b2 orhtonormal?: {}".format(b1[4,:].dot(b2[4,:])))

        ## Get extremes along each axis
        #print("Hull pnts shape: ", hull_pnts.shape)
        #x = np.apply_along_axis(lambda col: np.array([col.min(), col.max()]),
                                #arr=hull_pnts.dot(b1.T),axis=0)
        #y = np.apply_along_axis(lambda col: np.array([col.min(), col.max()]),
                                #arr=hull_pnts.dot(b2.T),axis=0)

        #areas      = (y[0, :] - y[1, :]) * (x[0, :] - x[1, :])
        ##perimeters = (y[0, :] - y[1, :]) + (x[0, :] - x[1, :])
        #perimeters = abs(1.0 - ((y[0, :] - y[1, :]) / (x[0, :] - x[1, :])))
        ## Get indices of bbs with areas within eps of smallest bb.
        #smallest_areas_idx = np.where(areas < (1.0 + self.eps) * np.min(areas))[0]
        #smallest_perims = perimeters[smallest_areas_idx]

        #k_p = np.argmin(smallest_perims)
        #k   = smallest_areas_idx[k_p]

        #print "all areas: "
        #print areas
        #print "all perimeters: "
        #print perimeters
        #print "k = ", k
        #print "chosen perimeters: "
        #print smallest_perims
        #print "chosen perimeter: ", k_p, "with the value: ", smallest_perims[k_p]

        #print("k_p:{}".format(k_p))
        #print("areas: {}".format(smallest_areas_idx))

        ## print("k_a: {}".format(k_a))
        #print("x shape: {} y shape: {}".format(x.shape, y.shape))

        #print(x[:, k])
        #print(y[:,k])

        #inverse_rot = np.vstack([b1[k,:], b2[k,:]]) # rotate back to orig coords

        #rot_bb = np.array([x[[0,1,1,0,0],k],
                        #y[[0,0,1,1,0],k]])

        #print(rot_bb)


        #bb = inverse_rot.T.dot(rot_bb) # rotate points back to orig frame

        #all_contours = []
        #n = b1.shape[0]
        #for i in range(n):
            #inverse_rot = np.vstack([b1[i,:], b2[i,:]])
            #rot_bb = np.array([x[[0,1,1,0,0],i],
                            #y[[0,0,1,1,0],i]])
            #box = inverse_rot.T.dot(rot_bb)
            #all_contours.append(box.T.copy())

        #smallest_areas = []
        #n = smallest_areas_idx.shape[0]
        #for j in range(n):
            #i = smallest_areas_idx[j]
            #inverse_rot = np.vstack([b1[i,:], b2[i,:]])
            #rot_bb = np.array([x[[0,1,1,0,0],i],
                            #y[[0,0,1,1,0],i]])
            #box = inverse_rot.T.dot(rot_bb)
            #smallest_areas.append(box.T.copy())

        return boundary, avg_boundary, pca_boundary, combined_boundary, pnts
        #return bb.T, hull_pnts, all_contours, smallest_areas, [axis] * 1000, length, boundary, pnts

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
            self.mean = pnts.mean(axis=0)
            pnts -= self.mean

        return np.dot(pnts, components.T)

    def _inverse_transform(self, pnts):

        return np.dot(pnts.T, self.bases) + self.mean

    def _inverse_pc_transform(self, bb2D, b3):
        # Meiying
        b3 = b3 / np.linalg.norm(b3) # normalize

        pnts_1d = self.pnts.dot(b3)

        # initial 2D bb is located in the center of the point cloud.
        # We want to copy it and shift both in the directions of 3rd PC

        width = pnts_1d.max()  - pnts_1d.min()
        width_dir = b3 * width / 2.0 # This tells us how to translate points

        # width_dir = b3 * pca3D.singular_values_[2] / 2

        bb3D = self.pca2D.inverse_transform(bb2D) # to be updated?

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

    def _get_bounding_box(self, pnts, axis, projection_indices, other_index):
        # Meiying
        pnts = self._transform(self.pca3D.components_[indices[(projection_indices, indices[projection_indices]), :]]) # test and fix this
        bb2D = self.mbb2D(pnts)
        bb3D = self._inverse_pc_transform(bb2D, self.pca3D.components_[other_index, :])
        return bb3D

    def _get_bounding_box_area(self, bounding_box):
        # Meiying
        axes = self._bounding_box_to_axis(bounding_box)
        return np.linalg.norm(axes[:, 0]) * np.linalg.norm(axes[:, 1]) * np.linalg.norm(axes[:, 2])

    def _close_to(self, m, n, error=1e-6):
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

    def get_3d_axis(self):
        
        return None

    def mbb3D(self):
        """
        Estimate vertices of 3d bounding box using 2d bb and 3rd P.C.
        """

        print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!PCA 3d"
        print self.pca3D.components_
        x = self.pca3D.components_[0]
        y = self.pca3D.components_[1]
        z = self.pca3D.components_[2]
        print "chosen: ", self.pca3D.components_[[0,1], :]
        print "x, y", np.dot(x, y)
        print "x, z", np.dot(x, z)
        print "y, z", np.dot(y, z)

        #axis_3d = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
        axis_3d = self.pca3D.components_[[0,1], :]

        pnts_2d = self._transform(self.pnts, axis_3d)
        _,_,_,bb2D, _ = self.mbb2D(pnts_2d)

        b3 = self.pca3D.components_[2,:] # 3rd PC
        b3 = b3 / np.linalg.norm(b3) # normalize

        pnts_1d = self.pnts.dot(b3)

        # initial 2D bb is located in the center of the point cloud.
        # We want to copy it and shift both in the directions of 3rd PC

        width = pnts_1d.max()  - pnts_1d.min()
        width_dir = b3 * width / 2.0 # This tells us how to translate points

        # width_dir = b3 * pca3D.singular_values_[2] / 2

        # bb3D = self.pca2D.inverse_transform(bb2D)
        print(bb2D)
        bb3D = self._inverse_transform(bb2D)

        bb_side1 = bb3D + width_dir
        bb_side2 = bb3D - width_dir

        bb3D = np.vstack([bb_side1, bb_side2])

        return bb3D

    def _visualize_point_cloud(self, fig, axis):
        axis.scatter(xs=self.pnts[:,0], ys=self.pnts[:,1], zs=self.pnts[:,2], c='b')

    def _visualize_bounding_box(self, fig, axis, bb, color='r'):
        ax.scatter(xs=bb[:,0], ys=bb[:,1], zs=bb[:,2], c=color, s=100)

    def plot_bb(self, n="3D", components=None):
        """Visualize points and bounding box"""


        fig = plt.figure()

        if n is "3D":
            # TODO: Connect vertices of 3D bounding box.         
            
            ax = fig.add_subplot(111, projection='3d')
            ax.axis('equal')
            bb = self.mbb3D()
            print "bb3D"
            print bb
            print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!axis"
            x = bb[0] - bb[1]
            y = bb[0] - bb[3]
            z = bb[0] - bb[5]
            print x / np.linalg.norm(x)
            print y / np.linalg.norm(y)
            print z / np.linalg.norm(z)
            print "x, y", np.dot(x, y)
            print "x, z", np.dot(x, z)
            print "y, z", np.dot(y, z)            
            ax.scatter(xs=self.pnts[:,0], ys=self.pnts[:,1], zs=self.pnts[:,2], c='b')
            ax.scatter(xs=bb[:,0], ys=bb[:,1], zs=bb[:,2], c='r', s=100)
            
            ax.plot(bb.T[0], bb.T[1], bb.T[2], c='r')
            ax.plot(bb[(1, 6), :].T[0], bb[(1, 6), :].T[1], bb[(1, 6), :].T[2], c='r')
            ax.plot(bb[(2, 7), :].T[0], bb[(2, 7), :].T[1], bb[(2, 7), :].T[2], c='r')
            ax.plot(bb[(3, 8), :].T[0], bb[(3, 8), :].T[1], bb[(3, 8), :].T[2], c='r')
            

        elif n is "2D":

            if components is None:
                components = self.pca2D.components_[[0,1], :]

            pnts = self._transform(self.pnts, components)
            #bb, hull_pnts, all_contours, smallest_areas, axis, length, boundary, pnt = self.mbb2D(pnts)
            boundary, avg_boundary, pca_boundary, combined_boundary, pnt = self.mbb2D(pnts)
            ax = fig.add_subplot(111)
            ax.axis('equal')

            pnts = self.pca2D.transform(self.pnts)
            ax.scatter(x=pnt[:, 0], y=pnt[:,1], c='b')
            #ax.scatter(x=hull_pnts[:,0], y=hull_pnts[:,1], c='r')

            #for axi in axis:
                #ax.scatter(x=axi[0, 0], y=axi[1, 0], c='r')
                #ax.scatter(x=axi[0, 1], y=axi[1, 1], c='r')

            boundary = boundary.T
            # print "boundary is", boundary

            avg_boundary = avg_boundary.T
            # print "avg_boundary is", avg_boundary
            
            pca_boundary = pca_boundary.T
            print "pca_boundary: "
            print pca_boundary
            
            combined_boundary = combined_boundary.T
            print "combined_boundary:"
            print combined_boundary

            #ax.scatter(x=boundary[0, 0], y=boundary[0, 1], c='y')
            #ax.scatter(x=boundary[1, 0], y=boundary[1, 1], c='y')
            #ax.scatter(x=boundary[2, 0], y=boundary[2, 1], c='y')
            #ax.scatter(x=boundary[3, 0], y=boundary[3, 1], c='y')

            #ax.plot(boundary.T[0], boundary.T[1], c='y')
            
            #ax.scatter(x=pca_boundary[0, 0], y=pca_boundary[0, 1], c='r')
            #ax.scatter(x=pca_boundary[1, 0], y=pca_boundary[1, 1], c='r')
            #ax.scatter(x=pca_boundary[2, 0], y=pca_boundary[2, 1], c='r')
            #ax.scatter(x=pca_boundary[3, 0], y=pca_boundary[3, 1], c='r')

            #ax.plot(pca_boundary.T[0], pca_boundary.T[1], c='r')             

            #ax.scatter(x=avg_boundary[0, 0], y=avg_boundary[0, 1], c='g')
            #ax.scatter(x=avg_boundary[1, 0], y=avg_boundary[1, 1], c='g')
            #ax.scatter(x=avg_boundary[2, 0], y=avg_boundary[2, 1], c='g')
            #ax.scatter(x=avg_boundary[3, 0], y=avg_boundary[3, 1], c='g')

            #ax.plot(avg_boundary.T[0], avg_boundary.T[1], c='g')
            
            ax.scatter(x=combined_boundary[0, 0], y=combined_boundary[0, 1], c='g')
            ax.scatter(x=combined_boundary[1, 0], y=combined_boundary[1, 1], c='g')
            ax.scatter(x=combined_boundary[2, 0], y=combined_boundary[2, 1], c='g')
            ax.scatter(x=combined_boundary[3, 0], y=combined_boundary[3, 1], c='g')

            ax.plot(combined_boundary.T[0], combined_boundary.T[1], c='g')            
                       
            #for i in range(len(length)):
                #ax.scatter(x = i, y = length[i], c='r')

            #for contour in all_contours:
                #ax.plot(contour[[0,1], 0], contour[[0,1],1],c='g')
                #ax.plot(contour[[1,2], 0], contour[[1,2],1],c='g')
                #ax.plot(contour[[2,3], 0], contour[[2,3],1],c='g')
                #ax.plot(contour[[3,0], 0], contour[[3,0],1],c='g')

            #for contour in smallest_areas:
                #ax.plot(contour[[0,1], 0], contour[[0,1],1],c='b')
                #ax.plot(contour[[1,2], 0], contour[[1,2],1],c='b')
                #ax.plot(contour[[2,3], 0], contour[[2,3],1],c='b')
                #ax.plot(contour[[3,0], 0], contour[[3,0],1],c='b')

            #ax.plot(bb[[0,1], 0], bb[[0,1],1],c='r')
            #ax.plot(bb[[1,2], 0], bb[[1,2],1],c='r')
            #ax.plot(bb[[2,3], 0], bb[[2,3],1],c='r')
            #ax.plot(bb[[3,0], 0], bb[[3,0],1],c='r')

        plt.show()
