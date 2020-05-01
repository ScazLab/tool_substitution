from sample_points_from_stl import get_l_points
from pointcloud_bb import BoundingBox

if __name__ == '__main__':
    pnts = get_l_points(500)
    boundingBox = BoundingBox(pnts)
    