#!/usr/bin/env python

import math
import numpy as np

import stl
from stl import mesh
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d



def triangle_area_multi(v1,v2,v3):
    """Compute area of multiple triangles given in vertices """
    return 0.4 * np.linalg.norm(np.cross(v2 - v1,
                                          v3 - v1), axis=1)
def weighted_rand_indices(n, mesh):

    areas = triangle_area_multi(mesh.v0, mesh.v1, mesh.v2)
    probs = areas / areas.sum()

    return np.random.choice(range(len(areas)), size=n, p=probs)

def plot_mesh(mesh):
    fig = plt.figure()
    ax = mplot3d.Axes3D(fig)
    # Render the cube faces
    ax.add_collection3d(mplot3d.art3d.Poly3DCollection(mesh.vectors))

    # Auto scale to the mesh size
    # scale = np.concatenate([mesh.points]).flatten(-1)
    scale = mesh.points
    ax.auto_scale_xyz(scale, scale, scale)

    # Show the plot to the screen
    plt.show()

def stl_to_pointcloud(n, mesh):
    indx  = weighted_rand_indices(n, mesh)
    v1_xyz, v2_xyz, v3_xyz = mesh.v0[indx], mesh.v1[indx], mesh.v2[indx]

    # Get samples via barrycentric coords

    u      = np.random.rand(n, 1)
    v      = np.random.rand(n, 1)
    is_oob = u + v > 1

    u[is_oob] = 1 - u[is_oob]
    v[is_oob] = 1 - v[is_oob]
    w = 1 - (u + v)

    results = (v1_xyz * u) + (v2_xyz * v) + (w * v3_xyz)

    return results.astype(np.float32)


def get_guitar_points(n):
    m = mesh.Mesh.from_file('./tool_files/guitar.stl')
    pnts = stl_to_pointcloud(n, m)
    # This removes points for the human figure modeled in this file
    return np.array([pnt for pnt in pnts if pnt[0] > 1000])


def get_man_points(n):
    m = mesh.Mesh.from_file('./tool_files/guitar.stl')
    pnts = stl_to_pointcloud(n, m)
    # This gets only the points for the human figure modeled in this file
    return np.array([pnt for pnt in pnts if pnt[0] < 1000])

def get_saw_points(n):
    m = mesh.Mesh.from_file('./tool_files/tools.stl')
    pnts = stl_to_pointcloud(n, m)
    # This gets only the points for the human figure modeled in this file
    return np.array([pnt for pnt in pnts if pnt[0] > 2200])


def test_sampling(n, mesh):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    pnts = stl_to_pointcloud(n, mesh)
    # pnts = get_saw_points(n)

def get_hammer_points(n):
    m = mesh.Mesh.from_file('./tool_files/tools.stl')
    pnts = stl_to_pointcloud(n, m)
    # This gets only the points for the human figure modeled in this file
    return np.array([pnt for pnt in pnts if pnt[0] > 1914 and pnt[0] < 2200])

def get_rake_points(n):
    m = mesh.Mesh.from_file('./tool_files/rake.stl')
    pnts = stl_to_pointcloud(n, m)
    return pnts

def plot_pnts(pnts):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(xs=pnts[:,0], ys=pnts[:,1], zs=pnts[:,2], c='b')

    plt.show()

if __name__ == '__main__':
    # guitar_mesh = mesh.Mesh.from_file('./tool_files/guitar.stl')
    tools_mesh = mesh.Mesh.from_file('./tool_files/tools.stl')
    # mesh = gen_mesh_cube()
    # plot_mesh(mesh)
    pnts = get_hammer_points(50000)
    plot_pnts(pnts)
    test_sampling(5000, tools_mesh)
