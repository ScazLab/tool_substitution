#!/usr/bin/env python

import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from scipy.spatial import ConvexHull



FIGS_PATH = "./figs/"



def pnts_to_img(pnts, fn=None, dims=(3,3)):
    fig, ax = plt.subplots(figsize=dims)

    print(pnts.shape)
    ax.scatter(x=pnts[:, 0], y=pnts[:, 1], c='b', s=50)
    plt.axis('off')
    plt.axis("tight")  # gets rid of white border
    plt.axis("image")  # square up the image instead of filling the "figure" space

    if fn is None:
        plt.show()
    else:
        plt.savefig("{}{}".format(FIGS_PATH, fn))

def bbs_to_img(bbs, fn=None, dims=(3,3)):
    fig, ax = plt.subplots(figsize=dims)

    for bb in bbs:
        ax.plot(bb[[0,1], 0], bb[[0,1],1],c='b',linewidth=5)
        ax.plot(bb[[1,2], 0], bb[[1,2],1],c='b',linewidth=5)
        ax.plot(bb[[2,3], 0], bb[[2,3],1],c='b',linewidth=5)
        ax.plot(bb[[3,0], 0], bb[[3,0],1],c='b',linewidth=5)

    plt.axis('off')
    plt.axis("tight")  # gets rid of white border
    plt.axis("image")  # square up the image instead of filling the "figure" space

    if fn is None:
        plt.show()
    else:
        plt.savefig("{}{}".format(FIGS_PATH, fn))

class CompareTools(object):

    def __init__(self, src_tool_pc, sub_tool_pc):
        "Class for comparing shapes of two tools."

        self.src_pc = src_tool_pc
        self.sub_pc = sub_tool_pc


    def choose_tool_orientation(self, metric='hamming', width=300):

        if metric == 'hamming':
            dist_measure = self.hamming_distance
        elif metric == 'moment':
            dist_measure = self.moments_distance

        fns  = ['src.png', 'sub.png']

        for pc, fn in zip([self.src_pc, self.sub_pc], fns):
            _,bb = pc.bb_2d_projection([0, 1], 2, visualize=False)
            pnts = bb.get_pc()

            pnts_to_img(pnts, fn=fn)

        print("METRIC: {}".format(metric))

        for flip_sub in [False, True]:

            sub_status = "flipped" if flip_sub else "unflipped"
            score = dist_measure(fns[0], fns[1], flip_sub, width)

            print("{} score: {}".format(sub_status, score))


    def moments_distance(self, img1_fn, img2_fn, flip_sub=False,width=300):
        """
        Uses normalized central moments (ncm) to compare 
        """

        src_shape, sub_shape = self._resize_and_norm_imgs(img1_fn,
                                                          img2_fn,
                                                          flip_sub,
                                                          width)



        cv2.imshow('tool1', src_shape)
        cv2.waitKey()
        cv2.imshow('tool2', sub_shape)
        cv2.waitKey()


        moments1 = cv2.moments(src_shape)
        moments2 = cv2.moments(sub_shape)

        #central normalized moments:
        # nu20, nu11, nu02, nu30, nu21, nu12, nu03
        ncm_keys = [m for m in moments1 if 'nu' in m]

        ncm1 = np.array([moments1[m] for m in ncm_keys])
        ncm2 = np.array([moments2[m] for m in ncm_keys])

        score = np.linalg.norm(ncm1 - ncm2)
        print("MOMENTS SCORE: {}".format(score))

        return score

    def hamming_distance(self, img1_fn, img2_fn, flip_sub, width=300):
        """
        Similarity metric for two 2D shapes. Calculates the area of non-overlap
        between two contours. The larger the overlap, the more similar shapes are.
        Smaller score is better.

        """

        src_shape, sub_shape = self._resize_and_norm_imgs(img1_fn,
                                                          img2_fn,
                                                          flip_sub,
                                                          width=width)
        # sub_shape = np.flip(sub_shape)

        cv2.imshow('tool1', src_shape)
        cv2.waitKey()
        cv2.imshow('tool2', sub_shape)
        cv2.waitKey()


        # Shapes are drawn in white ( white == 1)
        shape1_area = np.count_nonzero(src_shape)
        shape2_area = np.count_nonzero(sub_shape)
        print("shape 1 area: {} shape 2 area: {}".format(shape1_area, shape2_area))

        # Intersection area == 2, black == 0, contour area == 1
        score = np.count_nonzero((src_shape+sub_shape) == 1)
        print("Final score: {}".format(score))

        return score


    def _normalize_contour(self, img, width=300):
        "Crop shape and then resize image based on width"
        # img MUST be binary

        cnt,h = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        bounding_rect = cv2.boundingRect(cnt[0])
        img_cropped_bounding_rect = img[bounding_rect[1]:bounding_rect[1] + bounding_rect[3],
                                    bounding_rect[0]:bounding_rect[0] + bounding_rect[2]]

        new_height = int((1.0 * img.shape[0])/img.shape[1] * width)
        img_resized = cv2.resize(img_cropped_bounding_rect, (width, new_height))

        return img_resized


    def _resize_and_norm_imgs(self, img1_fn, img2_fn, flip_sub, width):

        img1 = cv2.imread("{}{}".format(FIGS_PATH,img1_fn), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread("{}{}".format(FIGS_PATH,img2_fn), cv2.IMREAD_GRAYSCALE)

        # convert grayscale imgs to binary
        (thresh, img_bw1) = cv2.threshold(img1, 128, 255,
                                          cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        (thresh, img_bw2) = cv2.threshold(img2, 128, 255,
                                          cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        resized_img1 = self._normalize_contour(img_bw1, width)
        resized_img2 = self._normalize_contour(img_bw2, width)



        cv2.imshow('tool1', resized_img1)
        cv2.waitKey()
        cv2.imshow('tool2', resized_img2)
        cv2.waitKey()

        assert(resized_img1.shape == resized_img2.shape)

        # Get contours of the bounding boxes for resized imgs.
        cs1, h1   = cv2.findContours( resized_img1.copy(), cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE )
        cs2, h2   = cv2.findContours( resized_img2.copy(), cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE)

        blank = np.zeros(img1.shape[0:2])

        # Hierarchy keeps track of nested contours
        h1 = h1[0] # Removes unnecessary outer list
        h2 = h2[0]

        # Removes external contour so only the tool shape is whited.
        cs1 = [cs1[i] for i in range(len(cs1)) if h1[i][3] == 0]
        cs2 = [cs2[i] for i in range(len(cs2)) if h2[i][3] == 0]


        # -1 for all contours.
        num_contours = -1
        src_shape = cv2.drawContours( blank.copy(), cs1,  num_contours,
                                   1, thickness=-1)
        sub_shape = cv2.drawContours( blank.copy(), cs2, num_contours,
                                   color=1, thickness=-1)


        if flip_sub:
            sub_shape = np.flip(sub_shape)

        return src_shape, sub_shape
