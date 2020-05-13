#!/usr/bin/env python

import os
import math
import numpy as np
import random

import stl
from stl import mesh
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

from plyfile import PlyData, PlyElement

PLY_DIR_PATH = "./tool_files/data_demo_segmented_numbered/"


class Mesh(object):

    STL = 'stl'
    PLY = 'ply'

    def __init__(self, fn):
        "docstring"
        if ".ply" in fn:
            self._mesh = PlyData.read("{}{}".format(PLY_DIR_PATH, fn))
            self._f_type = self.PLY
            # n x 3 array of vertex indices for each triangle.
            self.vert_idx  = np.vstack(self._mesh['face'].data['vertex_indices'])
        elif ".stl" in fn:
            self._mesh = mesh.Mesh.from_file('./tool_files/tools.stl')
            self._f_type = self.STL

    @property
    def v0(self):
        if self._f_type == self.STL:
            return self._mesh.v0
        elif self._f_type == self.PLY:
            verts = self._mesh.elements[0].data[self.vert_idx[:,0]][['x','y', 'z']]
            # Transform into numpy array.
            return np.array([list(v) for v in verts])

    @property
    def v1(self):
        if self._f_type == self.STL:
            return self._mesh.v1
        elif self._f_type == self.PLY:
            verts = self._mesh.elements[0].data[self.vert_idx[:,1]][['x','y', 'z']]
            # Transform into numpy array.
            return np.array([list(v) for v in verts])

    @property
    def v2(self):
        if self._f_type == self.STL:
            return self._mesh.v2
        elif self._f_type == self.PLY:
            verts = self._mesh.elements[0].data[self.vert_idx[:,2]][['x','y', 'z']]
            # Transform into numpy array.
            return np.array([list(v) for v in verts])

    @property
    def colors(self):
        if self._f_type == self.PLY:
            return self._mesh['vertex'][['red', 'green', 'blue']]


class Mesh2Pointcloud(object):
    def __init__(self, n,  mesh):
        "docstring"
        self.mesh = mesh
        self.n    = n

    def _triangle_area_multi(self, v1,v2,v3):
        """Compute area of multiple triangles given in vertices """
        return 0.4 * np.linalg.norm(np.cross(v2 - v1,
                                            v3 - v1), axis=1)
    def _weighted_rand_indices(self):

        areas = self._triangle_area_multi(self.mesh.v0, self.mesh.v1, self.mesh.v2)
        probs = areas / areas.sum()

        return np.random.choice(range(len(areas)), size=self.n, p=probs)



    def get_pointcloud(self):
        indx  = self._weighted_rand_indices()
        v1_xyz, v2_xyz, v3_xyz = self.mesh.v0[indx], self.mesh.v1[indx], self.mesh.v2[indx]

        # Get samples via barrycentric coords

        u      = np.random.rand(self.n, 1)
        v      = np.random.rand(self.n, 1)
        is_oob = u + v > 1

        u[is_oob] = 1 - u[is_oob]
        v[is_oob] = 1 - v[is_oob]
        w = 1 - (u + v)

        results = (v1_xyz * u) + (v2_xyz * v) + (w * v3_xyz)

        return results.astype(np.float32)
        # if mesh._f_type == mesh.STL:
        #     return results.astype(np.float32)
        # else:
        #     colors = mesh.colors[indx]
        #     return results.astypr(np.float32), colors


class GeneratePointcloud(object):
    def __init__(self):
        "docstring"
        self.m2p = Mesh2Pointcloud

    def ply_to_pointcloud(self, n, fn):
        mesh = Mesh(fn)

        return self.m2p(n, mesh).get_pointcloud()

    def get_random_ply(self, n):
        tool = random.choice(os.listdir(PLY_DIR_PATH))
        k    = random.choice(os.listdir(os.path.join(PLY_DIR_PATH, tool)))
        f    = random.choice(os.listdir( os.path.join(PLY_DIR_PATH, tool, k) ))
        path = os.path.join(tool,k,f)

        return self.ply_to_pointcloud(n, path)



    def get_guitar_points(self, n):
        # m = mesh.Mesh.from_file('./tool_files/guitar.stl')
        m = Mesh('./tool_files/guitar.stl')
        pnts = self.m2p(n, m).get_pointcloud()
        # This removes points for the human figure modeled in this file
        return np.array([pnt for pnt in pnts if pnt[0] > 1000])


    def get_man_points(self, n):
        # m = mesh.Mesh.from_file('./tool_files/guitar.stl')
        m = Mesh('./tool_files/guitar.stl')
        pnts = self.m2p(n, m).get_pointcloud()
        # This gets only the points for the human figure modeled in this file
        return np.array([pnt for pnt in pnts if pnt[0] < 1000])

    def get_saw_points(self, n):
        # m = mesh.Mesh.from_file('./tool_files/tools.stl')
        m = Mesh('./tool_files/tools.stl')
        pnts = self.m2p(n, m).get_pointcloud()
        # This gets only the points for the human figure modeled in this file
        return np.array([pnt for pnt in pnts if pnt[0] > 2200])

    def test_sampling(self, n, mesh):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        pnts = self.m2p(n, mesh)
        # pnts = get_saw_points(n)

    def get_hammer_points(self, n):
        # m = mesh.Mesh.from_file('./tool_files/tools.stl')
        m = Mesh('./tool_files/tools.stl')
        pnts = self.m2p(n, m).get_pointcloud()
        # This gets only the points for the human figure modeled in this file
        return np.array([pnt for pnt in pnts if pnt[0] > 1914 and pnt[0] < 2200])

    def get_rake_points(self, n):
        # m = mesh.Mesh.from_file('./tool_files/rake.stl')
        m = Mesh('./tool_files/rake.stl')
        pnts = self.m2p(n, m).get_pointcloud()

        return np.array([pnt for pnt in pnts if pnt[2] > 4.184])


    def get_l_points(self, n):

        rect1_x = np.random.uniform(-1, 0, size=(n, ))
        rect1_y = np.random.uniform(0, 3, size=(n, ))
        rect1_z = np.random.uniform(0, 2, size=(n, ))
        rect1 = np.array([rect1_x, rect1_y, rect1_z]).T

        rect2_x = np.random.uniform(-1, 4, size=(n, ))
        rect2_y = np.random.uniform(-1, 0, size=(n, ))
        rect2_z = np.random.uniform(0, 2, size=(n, ))
        rect2 = np.array([rect2_x, rect2_y, rect2_z]).T

        return np.vstack([rect1, rect2])


if __name__ == '__main__':
    # guitar_mesh = mesh.Mesh.from_file('./tool_files/guitar.stl')
    # tools_mesh = mesh.Mesh.from_file('./tool_files/tools.stl')
    tools_mesh = Mesh('./tool_files/tools.stl')
    # print(tools_mesh.v1)
    # mesh = gen_mesh_cube()
    # plot_mesh(mesh)
    # pnts = get_hammer_points(50000)
    # pnts = get_l_points(500)
    # plot_pnts(pnts)
    # test_sampling(5000, tools_mesh)
    # fn = "hammer/3/hammer_out_4_10_fused.ply"
    # ply_to_pointcloud(100, fn)
    gc = GeneratePointcloud().get_random_ply(1000)
