#!/usr/bin/env python

source_tool = get_source_tool_point_cloud() # n by 3 matrix
substitutude_tool = get_substitude_tool_point_cloud() # n by 3 matrix

source_tool_part, substitutude_tool_part = get_alignment(source_tool, substitutude_tool)

source_tool_action_part = get_tool_part_point_cloud(source_tool_part, source_tool)
substitutude_tool_action_part = get_tool_part_point_cloud(substitutude_tool_part, substitutude_tool)

# R' is a 3 by 3 rotation matrix in the body frame of the action part of the substitutude_tool!!!
# So after rotation, the pose of the substitutude_tool in the tool_world frame is
# R = R_axis * R'
# where R_axis is the axis of the action part of substitude tool, R' is the rotation required
R, source_tool_point, substitutude_tool_point = get_point_alignment(source_tool_action_part, substitutude_tool_action_part)
# rotate the substitude tool to the correct pose
substitutude_tool = (R * substitutude_tool.T).T
T_aruco_substitude_tool_initial = # how to get this one?

# The required aruco pose of the source tool in the robot world frame!
T_aruco_source_tool = get_pose_source_tool()
# Rotation of the source tool in the world frame, so basically
# T_aruco_source_tool = T * T_initial
T = T_aruco_source_tool * inverse(T_initial)

T_substitude_tool_aruco = T * T_aruco_substitude_tool_initial
