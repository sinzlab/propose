
def plot_pose(ax, pose):
    n_dims = pose.shape[-1]

    def edge(node_1, node_2):
        return [[pose[node_1, dim], pose[node_2, dim]] for dim in range(n_dims)]

    # Head
    ax.plot(*edge(0, 1), c='tab:blue')
    ax.plot(*edge(0, 2), c='tab:blue')
    ax.plot(*edge(0, 3), c='tab:blue')
    ax.plot(*edge(1, 2), c='tab:blue')
    ax.plot(*edge(1, 3), c='tab:blue')
    ax.plot(*edge(2, 3), c='tab:blue')

    # Spine
    ax.plot(*edge(3, 4), c='tab:orange')
    ax.plot(*edge(4, 5), c='tab:orange')

    # Leg L
    ax.plot(*edge(5, 8), c='tab:olive')
    ax.plot(*edge(8, 17), c='tab:olive')
    ax.plot(*edge(17, 18), c='tab:olive')

    # Leg R
    ax.plot(*edge(5, 9), c='tab:cyan')
    ax.plot(*edge(9, 16), c='tab:cyan')
    ax.plot(*edge(16, 19), c='tab:cyan')

    # Arm L
    ax.plot(*edge(3, 12), c='tab:purple')
    ax.plot(*edge(12, 10), c='tab:purple')
    ax.plot(*edge(10, 11), c='tab:purple')

    # Arm R
    ax.plot(*edge(3, 13), c='tab:green')
    ax.plot(*edge(13, 14), c='tab:green')
    ax.plot(*edge(14, 15), c='tab:green')
