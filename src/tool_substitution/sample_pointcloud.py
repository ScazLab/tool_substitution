#!/usr/bin/env python

import os
import math
import numpy as np
import random

import open3d as o3d
import stl
from stl import mesh

from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

from plyfile import PlyData, PlyElement

from util import visualize_two_pcs

PLY_DIR_PATH = "./tool_files/data_demo_segmented_numbered/"
TOOL_DIR     = "../../tool_files/point_clouds/"


class Mesh(object):

    STL = 'stl'
    PLY = 'ply'
    PCD = 'pcd'

    def __init__(self, fn):
        "docstring"
        self.is_pcd = False
        if ".ply" in fn:
            try:
                try:
                    print "READING IN PLY"
                    self._mesh = PlyData.read(fn)
                except:
                    self._mesh = PlyData.read("{}{}".format(PLY_DIR_PATH, fn))
                self._f_type = self.PLY
            # n x 3 array of vertex indices for each triangle.
                self.vert_idx  = np.vstack(self._mesh['face'].data['vertex_indices'])
                self._gen_segment_dict()
                self.from_mesh = True
            except ValueError:
                print "USING O3D PC"
                pcd = o3d.io.read_point_cloud(fn)
                print "ESTIMATING NORMALS"
                # pcd.estimate_normals()
                print "CREATING MESH"
                # self._mesh,_ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
                self._mesh = pcd
                print "CREATED MESH"
                self.is_pcd = True


        elif ".stl" in fn:
            self._mesh = mesh.Mesh.from_file(fn)
            self._f_type = self.STL
            self.from_mesh = True

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
           # print self._mesh.elements[0].data[self.vert_idx[:,[0,1,2]]][['red',
           #                                                                 'green',
           #                                                                 'blue']]
           colors = self._mesh.elements[0].data[self.vert_idx[:, 0]][['red',
                                                                 'green',
                                                                 'blue']]
           colors = np.array([self.segment_dict[(c[0], c[1], c[2])] for c in colors])

           return colors
    def _gen_segment_dict(self):
        segments = np.unique(self._mesh['vertex'][['red', 'green', 'blue']])

        self.segment_dict = {}
        for s in range(segments.shape[0]):
            self.segment_dict[tuple(segments[s])] = s


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


    def get_pointcloud(self, get_color=False):

        if not self.mesh.is_pcd:
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

            if self.mesh._f_type == self.mesh.PLY and get_color:
                colors = self.mesh.colors[indx]
                # Add colors to pointcloud matrix and return.
                return np.vstack([results.astype(np.float32).T, colors]).T
            else:
                return results.astype(np.float32)
        else:
            # pnts = self.mesh._mesh.sample_points_uniformly(self.n)
            pnts = self.mesh._mesh
            return np.asarray(pnts.points)


class GeneratePointcloud(object):
    def __init__(self):
        "docstring"
        self.m2p = Mesh2Pointcloud

    def load_pointcloud(self, fn, n=None):
         # if '.ply' in fn:
         #     pnts = PlyData.read(fn)
         #     pnts = self._mesh['vertex'][['x', 'y', 'z']]
         #     pnts = np.array([list(p) for p in pnts])

         # elif '.pcd' in fn:
        pnts = o3d.io.read_point_cloud(fn)
        pnts = np.asarray(pnts.points)

        if not n is None:
            n  = n if n < pnts.shape[0] else pnts.shape[0]
            idx = np.random.choice(n, pnts.shape[0])

            pnts = pnts[idx, :]

        return pnts


    def mesh_to_pointcloud(self, fn, n, get_color=False):
        mesh = Mesh(fn)

        return self.m2p(n, mesh).get_pointcloud(get_color)

    def get_random_pointcloud(self, n):
        paths = []
        for path, subdirs, files in os.walk(TOOL_DIR):
            name = random.choice(files)
            print "NAME: ", name
            if ".ply" in name or ".pcd" in name or '.stl' in name:
                path = os.path.join(path, name)
                paths.append(path)

        path = random.choice(paths)
        print("LOADING {}\n".format(path))
        return self.mesh_to_pointcloud(path, n, get_color=False)



    def get_random_ply(self, n, get_color=False):
        segments = 10
        k = ""
        tool = random.choice(os.listdir(PLY_DIR_PATH))
        while not segments <= 3:
            k  =  random.choice(os.listdir(os.path.join(PLY_DIR_PATH, tool)))
            segments = int(k)
            print("SEG ", segments)
        f    = random.choice(os.listdir( os.path.join(PLY_DIR_PATH, tool, k) ))
        path = os.path.join(PLY_DIR_PATH,tool,k,f)
        print("LOADING {}\n".format(path))

        return self.mesh_to_pointcloud(path, n, get_color)

    def get_knife_points(self, n):
        m = Mesh('./tool_files/knife.stl')
        pnts = self.m2p(n, m).get_pointcloud()
        return np.array(pnts)        

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
        m = Mesh('./tool_files/rake_recap.stl')
        pnts = self.m2p(n, m).get_pointcloud()

        return np.array([pnt for pnt in pnts if pnt[2] > 4.184])

    def get_both_rake_points(self, n):
        # m = mesh.Mesh.from_file('./tool_files/rake.stl')
        m = Mesh('./tool_files/rake.stl')
        pnts = self.m2p(n, m).get_pointcloud()

        return np.array([pnt for pnt in pnts if pnt[2] > 4.184]), pnts


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
    #tools_mesh = Mesh('./tool_files/tools.stl')
    # print(tools_mesh.v1)
    # mesh = gen_mesh_cube()
    # plot_mesh(mesh)
    # pnts = get_hammer_points(50000)
    # pnts = get_l_points(500)
    # plot_pnts(pnts)
    # test_sampling(5000, tools_mesh)
    # fn = "hammer/3/hammer_out_4_10_fused.ply"
    # mesh_to_pointcloud(100, fn)
    rake1 = GeneratePointcloud().mesh_to_pointcloud(1000, './tool_files/rake.stl')
    rake2 = GeneratePointcloud().get_rake_points(1000)
    visualize_two_pcs(rake1, rake2)
    
