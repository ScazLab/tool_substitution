#!/usr/bin/env python

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from pointcloud_bb import BoundingBox

from sample_points_from_stl import( get_guitar_points, get_man_points,
                                    get_hammer_points, get_saw_points,
                                    get_rake_points)

from tool_pointcloud import ToolPointCloud
from bounding_box import BoundingBox2D
from compare_tools import CompareTools, bbs_to_img

def plot_segments(bbs, pnts, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(xs=pnts[:,0], ys=pnts[:,1], zs=pnts[:,2], c=labels)

    for bb in bbs:
        ax.scatter(xs=bb[:,0], ys=bb[:,1], zs=bb[:,2], c='r', s=100)
        # ax.plot(bb[[0,1], 0], bb[[0,1],1],c='b',linewidth=5)
        # ax.plot(bb[[1,2], 0], bb[[1,2],1],c='b',linewidth=5)
        # ax.plot(bb[[2,3], 0], bb[[2,3],1],c='b',linewidth=5)
        # ax.plot(bb[[3,0], 0], bb[[3,0],1],c='b',linewidth=5)

    plt.show()

def segment_pnts(pnts, fn=None, k=2):
    """
    Segment points using kmeans and then fit BBs to each segment.
    """
    bbs = []

    kmean = KMeans(n_clusters=k)
    kmean.fit(pnts)
    labels = kmean.labels_


    pc = ToolPointCloud(pnts)

    # Project points into 2D.
    _,bb = pc.bb_2d_projection([0, 1], 2, visualize=False)
    pnts_2d = bb.get_pc()

    # Collect BBs for each cluster.
    for i in range(0, k):
        idx  = [j for j in range(len(labels)) if labels[j]==i]
        cluster_pnts = pnts_2d[idx]

        bb   = BoundingBox2D(cluster_pnts)
        bb2d = bb.get_bb()

        bbs.append(bb2d)


    if not fn is None:
        fn = "{}_{}".format(fn, k)

    # Save tool shape as outlines of BBs.
    bbs_to_img(bbs, fn=fn)


def compare_two_tools(k=2):
    pnts1 = get_rake_points(7000)
    pnts2 = get_guitar_points(7000)
    
    fn1 = "rake"
    fn2 = "guitar"

    segment_pnts(pnts1, fn=fn1, k=k)
    segment_pnts(pnts2, fn=fn2, k=k)

    fn1 = "{}_{}.png".format(fn1, k)
    fn2 = "{}_{}.png".format(fn2, k)

    ct = CompareTools()
    ct.hamming_distance(fn1, fn2)



if __name__ == '__main__':
    # pnts = get_rake_points(n=7000)
    compare_two_tools(k=1)
    # segment_pnts(pnts, fn='guitar', k=k)
