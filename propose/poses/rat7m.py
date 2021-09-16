from propose.poses.base import BasePose

import numpy as np


class Rat7mPose(BasePose):
    """
    Pose Class for the Rat7M dataset.
    """
    marker_names = [
        'HeadF',
        'HeadB',
        'HeadL',
        'SpineF',
        'SpineM',
        'SpineL',
        'Offset1',
        'Offset2',
        'HipL',
        'HipR',
        'ElbowL',
        'ArmL',
        'ShoulderL',
        'ShoulderR',
        'ElbowR',
        'ArmR',
        'KneeR',
        'KneeL',
        'ShinL',
        'ShinR',
    ]

    def __init__(self, pose_matrix):
        super().__init__(pose_matrix)

    @property
    def edge_groups(self):
        """
        Edge groups for plotting.
        :return: dict of edge groups
        """
        head_edges = np.array([
            self._edge('HeadF', 'HeadB'),
            self._edge('HeadF', 'HeadL'),
            self._edge('HeadF', 'SpineF'),
            self._edge('HeadL', 'SpineF'),
            self._edge('HeadL', 'HeadB'),
            self._edge('HeadB', 'SpineF'),
        ])

        spine_edges = np.array([
            self._edge('SpineF', 'SpineM'),
            self._edge('SpineM', 'SpineL'),
        ])

        leg_l_edges = np.array([
            self._edge('SpineL', 'HipL'),
            self._edge('HipL', 'KneeL'),
            self._edge('KneeL', 'ShinL'),
        ])

        leg_r_edges = np.array([
            self._edge('SpineL', 'HipR'),
            self._edge('HipR', 'KneeR'),
            self._edge('KneeR', 'ShinR'),
        ])

        arm_l_edges = np.array([
            self._edge('SpineF', 'ShoulderL'),
            self._edge('ShoulderL', 'ElbowL'),
            self._edge('ElbowL', 'ArmL'),
        ])

        arm_r_edges = np.array([
            self._edge('SpineF', 'ShoulderR'),
            self._edge('ShoulderR', 'ElbowR'),
            self._edge('ElbowR', 'ArmR'),
        ])

        return {
            'head': head_edges,
            'spine': spine_edges,
            'leg_l': leg_l_edges,
            'leg_r': leg_r_edges,
            'arm_l': arm_l_edges,
            'arm_r': arm_r_edges,
        }
