#!/usr/bin/env python

import argparse
import cv2
from compare_tools import bbs_to_img, CompareTools, FIGS_PATH

from sample_points_from_stl import (get_guitar_points, get_man_points,
                                    get_hammer_points, get_saw_points,
                                    get_rake_points, get_l_points)

from tool_segmentation_example import compare_two_tools
from tool_pointcloud import ToolPointCloud


parser = argparse.ArgumentParser()
parser.add_argument("-t", type=str, default='hammer',
                    help="Tool to create bounding box for. \
                    Options: 'hammer', 'guitar', 'saw', 'rake', 'L' ")
parser.add_argument("-H", type=int, default=3,
                    help="Calc Hamming dist of two segmented tools. Takes num segments as arg. ")

args = parser.parse_args()


def guitar_pc():
    return get_guitar_points(7000)

def hammer_pc():
    return get_hammer_points(2000)

def saw_pc():
    return get_saw_points(3000)

def rake_pc():
    return get_rake_points(7000)

def l_pc():
    return get_l_points(7000)

def plot_l_PC():
    pnts = get_l_points(7000)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    pca = PCA(n_components=3)
    new_pnts = pca.fit_transform(pnts)
    ax.scatter(xs=new_pnts[:,0], ys=new_pnts[:,1], zs=new_pnts[:,2], c='b')
    plt.show()



if __name__ == '__main__':
    tool_pc = None

    if args.t == 'hammer':
        tool_pc = hammer_pc()
    elif args.t == 'saw':
        tool_pc = saw_pc()
    elif args.t == 'guitar':
        tool_pc = guitar_pc()
    elif args.t == 'rake':
        tool_pc = rake_pc()
    elif args.t == 'L':
        tool_pc = l_pc()


    if tool_pc is not None:
        pc = ToolPointCloud(tool_pc)
        pc.visualize_bb()

    if args.H:
        compare_two_tools(k=args.H)
