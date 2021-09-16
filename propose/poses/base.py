import numpy as np
import matplotlib.pyplot as plt


class BasePose(object):
    marker_names = []

    def __init__(self, pose_matrix):
        self.pose_matrix = pose_matrix

        self.__array_struct__ = self.pose_matrix.__array_struct__

    def __getattr__(self, item):
        idx = self.marker_names.index(item)
        return self.__class__(self.pose_matrix[..., idx, :])

    def __getitem__(self, item):
        return self.__class__(self.pose_matrix[item])

    def __str__(self):
        return f'{self.__name__}(shape={self.shape}, pose_matrix={self.pose_matrix.__str__()})'

    def __repr__(self):
        return f'{self.__name__}(shape={self.shape}, pose_matrix={self.pose_matrix.__repr__()})'

    @property
    def shape(self):
        return self.pose_matrix.shape

    def _edge(self, marker_name_1, marker_name_2):
        n_dims = self.shape[-1]

        marker_pos_1 = getattr(self, marker_name_1)
        marker_pos_2 = getattr(self, marker_name_2)

        return [[marker_pos_1[..., dim].pose_matrix, marker_pos_2[..., dim].pose_matrix] for dim in range(n_dims)]

    @property
    def edge_groups_flat(self):
        edge_groups = list(self.edge_groups.values())
        return np.array([edge for edge_group in edge_groups for edge in edge_group])

    def plot(self, ax, cmap=plt.get_cmap("tab10")):
        edge_groups = self.edge_groups

        line_actors = []
        for edge_group_name, c in zip(edge_groups, cmap.colors):
            for edge in edge_groups[edge_group_name]:
                line_actors.append(*ax.plot(*edge, c=c))

        return line_actors

    def _animate(self, line_actors):
        updates = self.edge_groups_flat
        for line_actor, update in zip(line_actors, updates):
            line_actor.set_data(update[0], update[1])
            if len(update) == 3:
                line_actor.set_3d_properties(update[2])

    def animate(self, ax):
        line_actors = self[0].plot(ax)

        def animate(i):
            pose = self[i]
            pose._animate(line_actors)

        return line_actors, animate

    @property
    def edge_groups(self):
        raise NotImplementedError('Edge groups have not been defined for this pose')
