from sample_points_from_stl import get_l_points
from tool_pointcloud import ToolPointCloud
import matplotlib.pyplot as plt

if __name__ == '__main__':
    pnts = get_l_points(500)
    point_cloud = ToolPointCloud(pnts)
    
    box_1 = point_cloud.get_bounding_box()
    
    fig = plt.figure()
    axis = fig.add_subplot(111, projection='3d')
    point_cloud.visualize_point_cloud(fig, axis)
    box_1.visualize_bounding_box(fig, axis)
    plt.show()
    
    