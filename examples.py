#!/usr/bin/env python

import argparse
import cv2
from compare_tools import bbs_to_img, CompareTools, FIGS_PATH


from sample_pointcloud import GeneratePointcloud

from tool_segmentation_example import compare_two_tools
from tool_pointcloud import ToolPointCloud

import open3d as o3d


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
    tool_pc = rake_pc()

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

    if tool_pc is  None:

        tool_pc = gp.get_random_ply(2000)
        pc = ToolPointCloud(tool_pc)
        pc.visualize()
        centered_pc = pc.get_pc_bb_axis_frame_centered()
        centered_tpc = ToolPointCloud(centered_pc)
        centered_tpc.visualize()
        #pc.visualize_bb()

    if args.H:
        pass
        # compare_two_tools(k=args.H)

    pnts1 = gp.get_random_ply(1000)
    pnts2 = gp.get_random_ply(1000)

    pc1 = ToolPointCloud(pnts1)
    pc2 = ToolPointCloud(pnts2)

    old_norm = pc1.bb.dim_lens
    old_vol  = old_norm[0] * old_norm[1] * old_norm[2]
    print("SRC TOOL BEFORE SCALING")
    print("SRC DIMS: {}".format(old_norm))
    print("SRC VOL: {}".format(old_vol))
    print("bb: {}".format(pc1.bb))

    target_norm = pc2.bb.dim_lens
    target_vol  = target_norm[0] * target_norm[1] * target_norm[2]
    print("TARGET DIMS: {}".format(target_norm))
    print("TARGET VOL: {}".format(target_vol))

    pc1.visualize_bb()


    pc1.scale_pnts_to_target(pc2, True)
    print("SRC TOOL AFTER SCALING")
    new_norm = pc1.bb.dim_lens
    new_vol  = new_norm[0] * new_norm[1] * new_norm[2]
    print("SRC DIMS: {}".format(new_norm))
    print("SRC VOL: {}".format(new_vol))
    pc1.visualize_bb()
# p1 = o3d.geometry.PointCloud()
# p2 = o3d.geometry.PointCloud()

# p1.points = o3d.utility.Vector3dVector(pnts1)
# p2.points = o3d.utility.Vector3dVector(pnts2)
# print(o3d.registration.evaluate_registration(p1,p2, .5))
