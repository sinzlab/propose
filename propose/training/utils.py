from torch_geometric.data import HeteroData


def get_x_graph(data_dict: dict) -> HeteroData:
    """
    Get the graph with only x nodes from the data dictionary.
    :param data_dict: data dictionary
    :return: Graph with only x nodes
    """
    M_only_graph = {
        'x': {**data_dict['x']}
    }

    try:
        M_only_graph[('x', '->', 'x')] = {**data_dict[('x', '->', 'x')]}
    except KeyError:
        pass

    try:
        M_only_graph[('x', '<-', 'x')] = {**data_dict[('x', '<-', 'x')]}
    except KeyError:
        pass

    return HeteroData(M_only_graph)
