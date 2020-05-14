#!/usr/bin/env python

import argparse
import cv2
from compare_tools import bbs_to_img, CompareTools, FIGS_PATH


from sample_pointcloud import GeneratePointcloud

from tool_segmentation_example import compare_two_tools
from tool_pointcloud import ToolPointCloud


parser = argparse.ArgumentParser()
parser.add_argument("-t", type=str, default='hammer',
                    help="Tool to create bounding box for. \
                    Options: 'hammer', 'guitar', 'saw', 'rake', 'L', 'knife' ")
parser.add_argument("-H", type=int, default=3,
                    help="Calc Hamming dist of two segmented tools. Takes num segments as arg. ")

args = parser.parse_args()

gp = GeneratePointcloud()

def guitar_pc():
    return gp.get_guitar_points(7000)

def hammer_pc():
    return gp.get_hammer_points(7000)

def saw_pc():
    return gp.get_saw_points(7000)

def rake_pc():
    return gp.get_rake_points(7000)

def l_pc():
    return gp.get_l_points(7000)

def knife_pc():
    return gp.get_knife_points(7000)

def plot_l_PC():
    pnts = gp.get_l_points(7000)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    pca = PCA(n_components=3)
    new_pnts = pca.fit_transform(pnts)
    ax.scatter(xs=new_pnts[:,0], ys=new_pnts[:,1], zs=new_pnts[:,2], c='b')
    plt.show()



if __name__ == '__main__':
    tool_pc = knife_pc()

    #if args.t == 'hammer':
        #tool_pc = hammer_pc()
    #elif args.t == 'saw':
        #tool_pc = saw_pc()
    #elif args.t == 'guitar':
        #tool_pc = guitar_pc()
    #elif args.t == 'rake':
        #tool_pc = rake_pc()
    #elif args.t == 'L':
        #tool_pc = l_pc()
    #elif args.t == 'knife':
        #tool_pc = knife_pc()

    print "tool_pc.shape: ", tool_pc.shape

    if tool_pc is not None:
        pc = ToolPointCloud(tool_pc)
        pc.visualize_bb()

    #if args.H:
        #compare_two_tools(k=args.H)
