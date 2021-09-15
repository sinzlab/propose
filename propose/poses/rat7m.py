from .base import BasePose

import matplotlib.pyplot as plt
import numpy as np


class Rat7mPose(BasePose):
    def __init__(self,
                 HeadF,
                 HeadB,
                 HeadL,
                 SpineF,
                 SpineM,
                 SpineL,
                 HipL,
                 KneeL,
                 ShinL,
                 HipR,
                 KneeR,
                 ShinR,
                 ElbowL,
                 ArmL,
                 ShoulderL,
                 ElbowR,
                 ArmR,
                 ShoulderR,
                 Offset1,
                 Offset2,
                 **kwargs
                 ):

        super().__init__(
            HeadF=HeadF,
            HeadB=HeadB,
            HeadL=HeadL,
            SpineF=SpineF,
            SpineM=SpineM,
            SpineL=SpineL,
            HipL=HipL,
            KneeL=KneeL,
            ShinL=ShinL,
            HipR=HipR,
            KneeR=KneeR,
            ShinR=ShinR,
            ElbowL=ElbowL,
            ArmL=ArmL,
            ShoulderL=ShoulderL,
            ElbowR=ElbowR,
            ArmR=ArmR,
            ShoulderR=ShoulderR,
            Offset1=Offset1,
            Offset2=Offset2,
        )

    def _edge(self, marker_name_1, marker_name_2):
        n_dims = self.shape[-1]

        marker_pos_1 = getattr(self, marker_name_1)
        marker_pos_2 = getattr(self, marker_name_2)

        return [[marker_pos_1[dim], marker_pos_2[dim]] for dim in range(n_dims)]

    @property
    def edge_groups(self):
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

    @property
    def edge_groups_flat(self):
        edge_groups = self.edge_groups
        return np.array([
            *edge_groups['head'],
            *edge_groups['spine'],
            *edge_groups['leg_l'],
            *edge_groups['leg_r'],
            *edge_groups['arm_l'],
            *edge_groups['arm_r']
        ])

    def plot(self, ax, cmap=plt.get_cmap("tab10")):
        edge_groups = self.edge_groups

        line_actors = []
        for edge_group_name, c in zip(edge_groups, cmap.colors):
            for edge in edge_groups[edge_group_name]:
                line_actors.append(*ax.plot(*edge, c=c))

        return line_actors

    def animate(self, line_actors):
        updates = self.edge_groups_flat
        for line_actor, update in zip(line_actors, updates):
            line_actor.set_data(update[0], update[1])
            if len(update) == 3:
                line_actor.set_3d_properties(update[2])
