import itertools
import os
import re

import networkx as nx
import numpy as np

INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
OUTPUT_NODE = 6


def get_top_k(array, k):
    return list(np.argpartition(array[0], -k)[-k:])


def parent_combinations(adjacency_matrix, node, n_parents=2):
    """Get all possible parent combinations for the current node."""
    if node != 1:
        # Parents can only be nodes which have an index that is lower than the current index,
        # because of the upper triangular adjacency matrix and because the index is also a
        # topological ordering in our case.
        return itertools.combinations(np.argwhere(adjacency_matrix[:node, node] == 0).flatten(),
                                      n_parents)  # (e.g. (0, 1), (0, 2), (1, 2), ...
    else:
        return [[0]]


def draw_graph_to_adjacency_matrix(graph):
    """
    Draws the graph in circular format for easier debugging
    :param graph:
    :return:
    """
    dag = nx.DiGraph(graph)
    nx.draw_circular(dag, with_labels=True)


def upscale_to_nasbench_format(adjacency_matrix):
    """
    The search space uses only 4 intermediate nodes, rather than 5 as used in nasbench
    This method adds a dummy node to the graph which is never used to be compatible with nasbench.
    :param adjacency_matrix:
    :return:
    """
    return np.insert(
        np.insert(adjacency_matrix,
                  5, [0, 0, 0, 0, 0, 0], axis=1),
        5, [0, 0, 0, 0, 0, 0, 0], axis=0)


def parse_log(path):
    f = open(os.path.join(path, 'log.txt'), 'r')
    # Read in the relevant information
    train_accuracies = []
    valid_accuracies = []
    for line in f:
        if 'train_acc' in line:
            train_accuracies.append(line)
        elif 'valid_acc' in line:
            valid_accuracies.append(line)

    valid_error = [[1 - 1 / 100 * float(re.search('valid_acc ([-+]?[0-9]*\.?[0-9]+)', line).group(1))] for line in
                   valid_accuracies]
    train_error = [[1 - 1 / 100 * float(re.search('train_acc ([-+]?[0-9]*\.?[0-9]+)', line).group(1))] for line in
                   train_accuracies]

    return valid_error, train_error

# https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]
