#!/usr/bin/env python

#import argparse
import cv2
from compare_tools import bbs_to_img, CompareTools, FIGS_PATH

from sample_points_from_stl import (get_guitar_points, get_man_points,
                                    get_hammer_points, get_saw_points,
                                    get_rake_points, get_l_points)

from tool_segmentation_example import compare_two_tools


parser = argparse.ArgumentParser()
parser.add_argument("-t", type=str, default='hammer',
                    help="Tool to create bounding box for. \
                    Options: 'hammer', 'guitar', 'saw', 'rake', 'L' ")
parser.add_argument("-H", nargs='+', type=str, default=[],
                    help="Calc Hamming dist of two tools \
                    Options: 'hammer', 'guitar', 'saw', 'rake', 'L' ")
args = parser.parse_args()

#parser = argparse.ArgumentParser()
#parser.add_argument("tool", type=str, default='hammer',
                    #help="Tool to create bounding box for. \
                    #Options: 'hammer', 'guitar', 'saw', 'rake', 'L' ")
#args = parser.parse_args()

def guitar_pc():
    return get_guitar_points(7000)

def guitar_bb(n, eps, comps=[0,1]):
    pnts = get_guitar_points(n)
    bb = BoundingBox(pnts, eps)

    c = bb.pca3D.components_[comps, :]

    bb.plot_bb("2D", c)
    bb.plot_bb("3D", c)

def hammer_bb(n, eps, comps=[0,1]):
    pnts = get_hammer_points(n)
    bb = BoundingBox(pnts, eps=.001)
    c = bb.pca3D.components_[comps, :]
    bb.plot_bb("2D", c)
    bb.plot_bb("3D", c)

def saw_bb(n, eps, comps=[0,1]):
    pnts = get_saw_points(n)
    bb = BoundingBox(pnts, eps)

    c = bb.pca3D.components_[comps, :]
    bb.plot_bb("2D", c)
    bb.plot_bb("3D", c)

def rake_bb(n, eps, comps=[0,1]):
    pnts = get_rake_points(n)
    bb = BoundingBox(pnts, eps, )

    c = bb.pca3D.components_[comps, :]

    bb.plot_bb("2D", c)
    bb.plot_bb("3D", c)

def l_bb(n, eps, comps=[0,1]):
    pnts = get_l_points(n)
    bb = BoundingBox(pnts, eps)

    c = bb.pca3D.components_[comps, :]

    bb.plot_bb("2D", bb.pca3D.components_[[1,2], :])
    bb.plot_bb("3D", c)

def plot_l_PC():
    pnts = get_l_points(n)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    pca = PCA(n_components=3)
    new_pnts = pca.fit_transform(pnts, eps)
    ax.scatter(xs=new_pnts[:,0], ys=new_pnts[:,1], zs=new_pnts[:,2], c='b')
    plt.show()


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
    k = 3
    n = 9000
    eps = 0.01
    comps = [1, 2]
    if args.t == 'hammer':
        hammer_bb(n, eps, comps)
    elif args.t == 'saw':
        saw_bb(n, eps, comps)
    elif args.t == 'guitar':
        guitar_bb(n, eps, comps)
    elif args.t == 'rake':
        rake_bb(n, eps, comps)
    elif args.t == 'L':
        l_bb(n, eps, comps)
    if args.H:
        compare_two_tools(args.H[0], args.H[1], k=k)
