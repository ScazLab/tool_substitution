#!/usr/bin/env python

import numpy as np

# Note: All the model frame should already by normalized/centered!

# Naming convention: T_<object>_<frame>, 4 * 4 transformation homogenous matrix
#                    Ps, many points, n * 3

# Given the perceived aruco pose, find the position of each point on the model in the world frame
def get_pnts_world_frame(T_aruco_world, T_aruco_model, Ps_pnts_model):
    # the aruco_world is perceived
    # others are saved value
    # Ps_pnts_model is a n by 3 matrix, each row is a point

    # There should be no stretching, so the unit between the world frame and the model frame must be the consistent!
    T_model_rotation_in_world = np.matmul(T_aruco_world, np.linalg.inv(T_aruco_model))

    #print "T_model_rotation_in_world"
    #print T_model_rotation_in_world

    # should be in the format of [x, y, z, 1].T
    Ps_pnts_model = np.hstack([Ps_pnts_model, np.ones((Ps_pnts_model.shape[0], 1))]).T

    Ps_pnts_world = np.matmul(T_model_rotation_in_world, Ps_pnts_model).T

    Ps_pnts_world = Ps_pnts_world[:, :-1] # get rid of the 1s in the end

    return Ps_pnts_world

# Given the pose of a point on the tool in the world point, and the point in the model frame, the aruco pose in the model frame, the T that the model that has been rotated
# Find the pose of the aruco in the world frame
def get_aruco_world_frame(T_aruco_model, Ps_pnts_model, Ps_pnts_world, R_pnts):
    # Ps_pnts_model, Ps_pnts_world: 1 by 3 matrix
    # T_aruco_model is known, get when scanning in the 3d model
    # Ps_pnts_model is from the geomatric matching result
    # Ps_pnts_world is the desired position in the world frame, which is obtained when learning the task
    # R_pnts_world is a 3 by 3 rotation matrix, which is the result of the geomatric matching
    # T_rotation = np.hstack([model_rotation, (Ps_pnts_world - Ps_pnts_model).T])
    # T_rotation = np.vstack([T_rotation, np.array([0, 0, 0, 1])])

    # T_pnts_rotation * T_pnts_initial = T_pnts_final
    # T_aruco_rotation * T_aruco_initial = T_aruco_final
    # T_pnts_rotation = T_aruco_rotation (in the world frame, not body frame)

    T_pnts_initial = get_T_from_R_p(Ps_pnts_model)
    T_pnts_final = get_T_from_R_p(Ps_pnts_world, R_pnts)

    T_pnts_rotation = np.matmul(T_pnts_final, np.linalg.inv(T_pnts_initial))

    #print "T_pnts_rotation"
    #print T_pnts_rotation

    T_aruco_rotation = T_pnts_rotation

    T_aruco_world = np.matmul(T_aruco_rotation, T_aruco_model)

    return T_aruco_world

def get_T_from_R_p(p, R = np.identity(3)):
    # p is 1 by 3
    T = np.hstack([R, p.T])
    T = np.vstack([T, np.array([0, 0, 0, 1])])

    return T

"""
T_aruco_model = np.array([[0, 1, 0, 3],
                          [1, 0, 0, 2],
                          [0, 0, -1, 0],
                          [0, 0, 0, 1]])

Ps_pnts_model = np.array([[4, 0, 0],
                          [0, 3, 0],
                          [4, 3, 0]])

T_aruco_world = np.array([[0, 0, -1, 0],
                          [1, 0, 0, 3],
                          [0, -1, 0, 1],
                          [0, 0, 0, 1]])

Ps_pnts_world = get_pnts_world_frame(T_aruco_world, T_aruco_model, Ps_pnts_model)

print "Ps_pnts_world"
print Ps_pnts_world

# This is obtained T_result_world * np.linalg.inv(T_initial_world))

print "====================================================="

# R_pnts_world * R_initial = R_final
# ==> R_pnts_world =  R_final * R_initial^-1

R_result= np.array([[ 0.,  0., -1.],
                    [ 1.,  0.,  0.],
                    [ 0., -1.,  0.]])

R_initial = np.array([[ 0.,  1.,  0.],
                      [ 1.,  0.,  0.],
                      [ 0.,  0., -1.]])

R_pnts_world = np.matmul(R_result, np.linalg.inv(R_initial))
print "R_pnts_world is"
print R_pnts_world

Ps_pnts_model = np.array([[4, 0, 0]])
Ps_pnts_world = np.array([[0, 1, 0]])

T_aruco_world_found = get_aruco_world_frame(T_aruco_model, Ps_pnts_model, Ps_pnts_world, R_pnts_world)

print "T_aruco_world_found"
print T_aruco_world_found

#model_rotation = np.array([[ 0.,  0.,  1.],
                           #[ 0.,  1.,  0.],
                           #[-1.,  0.,  0.]])

#model_rotation = np.array([[ 0.,  0., -1.,  0.],
                           #[ 0.,  1.,  0.,  1.],
                           #[-1.,  0.,  0.,  4.],
                           #[ 0.,  0.,  0.,  1.]])
"""