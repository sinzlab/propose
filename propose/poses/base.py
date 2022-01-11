import numpy as np
import matplotlib.pyplot as plt

from .utils import yaml_pose_loader

from abc import ABC, abstractmethod


class BasePose(ABC):
    """
    Base class for poses. Provides and structure for storing pose information and plotting poses.
    """

    marker_names = []
    adjacency_matrix = None

    def __init__(self, pose_matrix: np.ndarray):
        """
        :param pose_matrix: A ndarray (frames, markers, positions), where frames and markers are optional dimensions.
        """
        self.pose_matrix = pose_matrix

        self.__array_struct__ = self.pose_matrix.__array_struct__

        self.set_adjacency_matrix()

    def __getattr__(self, item):
        """
        Interface for getting markers based on the name's index in the markers_names list.
        e.g.
        marker_names = ['head', 'spine', 'leg_l']
        calling self.head returns a pose with self.pose_matrix[..., 0, :]

        :param item: the marker name to be selected
        :return: new Pose constructed from self.pose_matrix[..., marker_idx, :]
        """
        if item not in self.marker_names:
            raise AttributeError(f"{item} is not a valid marker name")

        idx = self.marker_names.index(item)

        return self.__class__(pose_matrix=self.pose_matrix[..., idx, :])

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.__getattr__(item)

        return self.__class__(self.pose_matrix[item])

    def __str__(self):
        return f"{self.__class__.__name__}(shape={self.shape}, pose_matrix={self.pose_matrix.__str__()})"

    def __repr__(self):
        return f"{self.__class__.__name__}(shape={self.shape}, pose_matrix={self.pose_matrix.__repr__()})"

    def __sub__(self, other):
        return self.__class__(self.pose_matrix - other.pose_matrix)

    def __len__(self):
        return self.shape[0]

    def __eq__(self, other):
        return np.array_equal(self, other)

    @property
    def shape(self):
        return self.pose_matrix.shape

    def get_edge(self, marker_name_1: str, marker_name_2: str):
        return (
            self.marker_names.index(marker_name_1),
            self.marker_names.index(marker_name_2),
        )

    def _edge(self, marker_name_1: str, marker_name_2: str):
        """
        Function for returning a line between two nodes on the pose graph.
        :param marker_name_1: The name of the marker at which the edge starts.
        :param marker_name_2: The name of the marker at which the edge ends.
        :return: line definition between marker 1 and marker 2
        """
        n_dims = self.shape[-1]

        marker_pos_1 = getattr(self, marker_name_1)
        marker_pos_2 = getattr(self, marker_name_2)

        return [
            [marker_pos_1[..., dim].pose_matrix, marker_pos_2[..., dim].pose_matrix]
            for dim in range(n_dims)
        ]

    @property
    def edge_groups_flat(self) -> np.ndarray:
        """
        Flat alternative to the edge_groups representation
        :return: Flattened groups as ndarray
        """
        edge_groups = list(self.edge_groups.values())
        return np.array([edge for edge_group in edge_groups for edge in edge_group])

    def plot(self, ax, cmap=plt.get_cmap("tab10")):
        """
        Plotting function for displaying the pose.

        :param ax: axis on which this should be plotted.
        :param cmap: colormap for displaying the edge group colors.
        :return: list of the line actors
        """
        edge_groups = self.edge_groups

        line_actors = []
        for edge_group_name, c in zip(edge_groups, cmap.colors):
            for edge in edge_groups[edge_group_name]:
                line_actors.append(*ax.plot(*edge, c=c))

        return line_actors

    def copy(self):
        return self.__class__(self.pose_matrix)

    def save(self, path):
        np.save(path, self.pose_matrix)

    @classmethod
    def load(cls, path):
        return cls(np.load(path))

    def update(self, line_actors: list):
        """
        Animation helper function. Updates the data for the line_actors
        :param line_actors:
        """
        updates = self.edge_groups_flat
        for line_actor, update in zip(line_actors, updates):
            line_actor.set_data(update[0], update[1])
            if len(update) == 3:
                line_actor.set_3d_properties(update[2])

    def animate(self, ax):
        """
        Constructor for animating the pose.

        e.g.
        fig = plt.figure()
        ax = plt.gca()
        _, animate = pose.animate(ax)
        ani = animation.FuncAnimation(fig, animate, frames=100)
        This will animate the frames from pose[0:100]
        :param ax:
        :return:
        """
        line_actors = self[0].plot(ax)

        def animate(i):
            pose = self[i]
            pose.update(line_actors)

        return line_actors, animate

    @property
    @abstractmethod
    def edge_groups(self):
        raise NotImplementedError("Edge groups have not been defined for this pose")

    @abstractmethod
    def set_adjacency_matrix(self):
        raise NotImplementedError("Adjacency matrix has not been setup")


class YamlPose(BasePose):
    def __init__(self, pose_matrix, path):
        marker_names, named_edges, named_group_edges = yaml_pose_loader(path)

        self.marker_names = marker_names
        self.__named_edges = named_edges
        self.__named_group_edges = named_group_edges
        self.__path = path

        super().__init__(pose_matrix)

    def set_adjacency_matrix(self):
        self.adjacency_matrix = np.eye(len(self.marker_names))

        edges = [self.get_edge(src, dst) for src, dst in self.__named_edges]

        for edge in edges:
            self.adjacency_matrix[edge] = 1
            self.adjacency_matrix[edge[::-1]] = 1

    @property
    def edge_groups(self):
        """
        Edge groups for plotting.
        :return: dict of edge groups
        """
        groups = {
            group: np.array(
                [self._edge(src, dst) for src, dst in self.__named_group_edges[group]]
            )
            for group in self.__named_group_edges.keys()
        }
        return groups

    def __getattr__(self, item):
        """
        Interface for getting markers based on the name's index in the markers_names list.
        e.g.
        marker_names = ['head', 'spine', 'leg_l']
        calling self.head returns a pose with self.pose_matrix[..., 0, :]

        :param item: the marker name to be selected
        :return: new Pose constructed from self.pose_matrix[..., marker_idx, :]
        """
        if item not in self.marker_names:
            raise AttributeError(f"{item} is not a valid marker name")

        idx = self.marker_names.index(item)
        return self.__class__(
            path=self.__path, pose_matrix=self.pose_matrix[..., idx, :]
        )

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.__getattr__(item)

        return self.__class__(path=self.__path, pose_matrix=self.pose_matrix[item])
