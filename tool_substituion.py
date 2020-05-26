import numpy as np
import matplotlib.pyplot as plt

from tool_pointcloud import ToolPointCloud
from sample_pointcloud import GeneratePointcloud
from util import min_point_distance, r_y


def visualize(tpc, cp):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    c = tpc.segments

    pnts = tpc.pnts

    ax.axis('equal')
    ax.scatter(xs=pnts[:, 0], ys=pnts[:, 1], zs=pnts[:, 2], c=c)
    ax.scatter(xs=cp[0], ys=cp[1], zs=cp[2], c='r', s=200)
    plt.show()



class ToolSubstitution(object):
    def __init__(self, src_tool_pc, sub_tool_pc ):
        "docstring"
        self.src_tool = src_tool_pc
        self.sub_tool = sub_tool_pc

    def _calc_best_orientation(self, src_pnts, sub_pnts):

        # Average scores due to slight asymmetry in distance metric.
        no_rot_score1 = min_point_distance(src_pnts.T, sub_pnts.T)
        no_rot_score2 = min_point_distance(sub_pnts.T, src_pnts.T)

        no_rot_score = (no_rot_score1 + no_rot_score2) / 2.0

        print("NO ROT SCORE: ", no_rot_score)

        sub_pnts_rot = r_y(np.pi).dot(sub_pnts.T)

        rot_score1 = min_point_distance(src_pnts.T, sub_pnts_rot)
        rot_score2 = min_point_distance(sub_pnts_rot, src_pnts.T)

        rot_score = (rot_score1 + rot_score2) / 2.0

        print("ROT SCORE: ", rot_score)

        if no_rot_score > rot_score:
            return sub_pnts
        else:
            return sub_pnts_rot.T

    def _get_closest_pnt(self, pnt, pntcloud):
        """
        returns the point in pntcloud closest to pnt.
        """
        diffs = np.apply_along_axis(lambda row: np.linalg.norm(pnt - row),
                                    axis=1, arr=pntcloud)

        idx = np.argmin(diffs).item()
        print("{}, type:{}".format(idx, type(idx)))
        return idx

    def _align_pnts(self, src_pc, sub_pc):


        scaled_sub_pnts = sub_pc.scale_pnts_to_target(src_pc)
        scaled_sub_tool = ToolPointCloud(scaled_sub_pnts)

        aligned_sub_pnts = scaled_sub_tool.get_pc_bb_axis_frame_centered()
        aligned_src_pnts = src_pc.get_pc_bb_axis_frame_centered()


        return self._calc_best_orientation(aligned_src_pnts,
                                           aligned_sub_pnts)



    def _get_sub_tool_action_part(self):
        # Center pointclouds
        src_pnts = self.src_tool.get_pc_bb_axis_frame_centered()
        sub_pnts = self.sub_tool.get_pc_bb_axis_frame_centered()

        # Add the segment labels back in.
        src_pnts = np.vstack([src_pnts.T, self.src_tool.segments]).T
        sub_pnts = np.vstack([sub_pnts.T, self.sub_tool.segments]).T

        self.centered_src_pc = ToolPointCloud(src_pnts)
        self.centered_sub_pc = ToolPointCloud(sub_pnts)

        # Find the best alignment of the sub tool based on 3d haming distance.
        aligned_sub_pnts = self._align_pnts(self.centered_src_pc, self.centered_sub_pc)


        # aligned_sub_pnts = self._calc_best_orientation(aligned_src_pnts,
        #                                                aligned_sub_pnts)

        contact_pnt = self.src_tool.get_pnt(self.src_tool.contact_pnt_idx)
        sub_action_pnt = self._get_closest_pnt(contact_pnt, aligned_sub_pnts)

        return self.sub_tool.get_segment_from_point(sub_action_pnt)

    def _align_action_parts(self):

        # Get segment of contact point of src tool.
        src_action_seg = self.src_tool.get_segment_from_point(self.src_tool.contact_pnt_idx)
        # Determine the corresponding segment based on shape in sub tool.
        sub_action_seg = self._get_sub_tool_action_part()

        # Get pnts corresponding to these segments
        src_action_pnts = self.centered_src_pc.get_pnts_in_segment(src_action_seg)
        sub_action_pnts = self.centered_sub_pc.get_pnts_in_segment(sub_action_seg)


        src_action_pc = ToolPointCloud(src_action_pnts)
        sub_action_pc = ToolPointCloud(sub_action_pnts)

        # align action segment of sub tool to src tool
        aligned_sub_pnts = self._align_pnts(src_action_pc, sub_action_pc)

        # # src_action_contact_pnt = src_action_pc.get_pnt(self.src_tool.contact_pnt_idx)


        # Determine the contact point of src tool in action part segment point cloud
        # and find corresponding point on the sub action part segment pointcloud
        src_cntct_idx = self.centered_src_pc.idx_to_segment_idx(self.src_tool.contact_pnt_idx)
        src_action_contact_pnt = src_action_pc.get_pnt(src_cntct_idx)
        # src_cntct_pnt = self.centered_src_pc.get_pnt(self.src_tool.contact_pnt_idx)
        # src_action_contact_pnt = src_action_pc.transform(src_cntct_pnt)
        sub_contact_pnt_idx = self._get_closest_pnt(src_action_contact_pnt,
                                                    aligned_sub_pnts)

        # Get calculated sub tool contact point on original pointcloud.
        sub_contact_pnt_idx = self.centered_sub_pc.segment_idx_to_idx(sub_action_seg,
                                                                      sub_contact_pnt_idx)
        sub_cntct_pnt = self.centered_sub_pc.get_pnt(sub_contact_pnt_idx)
        src_cntct_pnt = self.centered_src_pc.get_pnt(self.src_tool.contact_pnt_idx)

        print("SRC CONTACT PNT {}".format(src_cntct_pnt))
        visualize(self.centered_src_pc, src_cntct_pnt)
        print("sub CONTACT PNT {}".format(sub_cntct_pnt))
        visualize(self.centered_sub_pc, sub_cntct_pnt)

    def main(self):
        # TODO: Make sure contact point can be transformed properly and recovered
        self._align_action_parts()



if __name__ == '__main__':
    gp = GeneratePointcloud()
    n = 3000
    get_color = True

    pnts1 = gp.get_random_ply(n, get_color)
    pnts2 = gp.get_random_ply(n, get_color)

    src = ToolPointCloud(pnts1, contact_pnt_idx=None)
    sub = ToolPointCloud(pnts2)

    cntc_pnt = src.get_pc_bb_axis_frame_centered().argmin(axis=0)[0]
    src.contact_pnt_idx = cntc_pnt

    print("SRC TOOL")
    # src.visualize()
    print("SUB TOOL")
    # sub.visualize()

    ts = ToolSubstitution(src, sub)
    ts.main()






