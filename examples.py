#!/usr/bin/env python

import argparse
import cv2
from compare_tools import bbs_to_img, CompareTools, FIGS_PATH
from pointcloud_bb import BoundingBox

from sample_points_from_stl import (get_guitar_points, get_man_points,
                                    get_hammer_points, get_saw_points)

parser = argparse.ArgumentParser()
parser.add_argument("tool", type=str, default='hammer',
                    help="Tool to create bounding box for. Options: 'hammer', 'guitar', 'saw'")
args = parser.parse_args()



def guitar_bb():
    pnts = get_guitar_points(7000)
    bb = BoundingBox(pnts)
    bb.plot_bb("2D")
    bb.plot_bb("3D")

def hammer_bb():
    pnts = get_hammer_points(7000)
    bb = BoundingBox(pnts)
    bb.plot_bb("2D")
    bb.plot_bb("3D")

def saw_bb():
    pnts = get_saw_points(7000)
    bb = BoundingBox(pnts)
    bb.plot_bb("2D")
    bb.plot_bb("3D")

def test_hamming_dist():
    ct = CompareTools()
    img1 = cv2.imread("{}{}".format(FIGS_PATH,'test_box.png'), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("{}{}".format(FIGS_PATH,'test_box_rot.png'), cv2.IMREAD_GRAYSCALE)
    cv2.imshow('test', mat=img1)
    cv2.waitKey()
    cv2.imshow('test', mat=img2)
    cv2.waitKey()

    print("Score should be high due to orientation:")
    ct.hamming_distance('test_box.png', 'test_box_rot.png', width=1000)

    img2 = img1.copy()
    cv2.imshow('test', mat=img1)
    cv2.waitKey()
    cv2.imshow('test', mat=img2)
    cv2.waitKey()
    
    print("Score should 0 due to same shape and same orientation.")
    ct.hamming_distance('test_box.png', 'test_box.png', width=1000)

if __name__ == '__main__':
    if args.tool == 'hammer':
        hammer_bb()
    elif args.tool == 'saw':
        saw_bb()
    elif args.tool == 'guitar':
        guitar_bb()
