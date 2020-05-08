#!/usr/bin/env python

import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from scipy.spatial import ConvexHull



FIGS_PATH = "./figs/"


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

    def __init__(self):
        "Class for comparing shapes of two tools."


    def hamming_distance(self, img1_fn, img2_fn, width=300):
        """
        Similarity metric for two 2D shapes. Calculates the area of non-overlap
        between two contours. The larger the overlap, the more similar shapes are.
        Smaller score is better.

        """

        # Read in image as grayscale
        img1 = cv2.imread("{}{}".format(FIGS_PATH,img1_fn), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread("{}{}".format(FIGS_PATH,img2_fn), cv2.IMREAD_GRAYSCALE)

        # convert grayscale imgs to binary
        (thresh, img_bw1) = cv2.threshold(img1, 128, 255,
                                          cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        (thresh, img_bw2) = cv2.threshold(img2, 128, 255,
                                          cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        resized_img1, cs1, h1 = self._normalize_contour(img_bw1, width)
        resized_img2, cs2, h2 = self._normalize_contour(img_bw2, width)

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
        shape1 = cv2.drawContours( blank.copy(), cs1,  num_contours,
                                   1, thickness=-1)
        shape2 = cv2.drawContours( blank.copy(), cs2, num_contours,
                                   color=1, thickness=-1)

        cv2.imshow('tool1', shape1)
        cv2.waitKey()
        cv2.imshow('tool2', shape2)
        cv2.waitKey()


        # Shapes are drawn in white ( white == 1)
        shape1_area = np.count_nonzero(shape1)
        shape2_area = np.count_nonzero(shape2)
        print("shape 1 area: {} shape 2 area: {}".format(shape1_area, shape2_area))

        # Intersection area == 2, black == 0, contour area == 1
        score = np.count_nonzero((shape1+shape2) == 1)
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

        return img_resized, cnt, h
