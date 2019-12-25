import itertools

import copy
import numpy as np
import random
from abc import abstractmethod
from nasbench import api

from nasbench_analysis.utils import CONV1X1, CONV3X3, MAXPOOL3X3, INPUT, \
    OUTPUT
from nasbench_analysis.utils import parent_combinations


class SearchSpace:
    def __init__(self, search_space_number, num_intermediate_nodes):
        self.search_space_number = search_space_number
        self.num_intermediate_nodes = num_intermediate_nodes
        self.num_parents_per_node = {}

    @abstractmethod
    def create_nasbench_adjacency_matrix(self, parents, **kwargs):
        """Based on given connectivity pattern create the corresponding adjacency matrix."""
        pass

    @abstractmethod
    def sample(self, **kwargs):
        """Sample a valid adjacency matrix from the search space without loose ends."""
        pass

    @abstractmethod
    def sample_with_loose_ends(self, **kwargs):
        """Sample a valid adjacency matrix from the search space with loose ends."""
        pass

    @abstractmethod
    def generate_adjacency_matrix_without_loose_ends(self, **kwargs):
        """Returns every adjacency matrix in the search space without loose ends."""
        pass

    def generate_search_space_without_loose_ends(self):
        # Create all possible connectivity patterns
        for iter, adjacency_matrix in enumerate(self.generate_adjacency_matrix_without_loose_ends()):
            print(iter)
            # Print graph
            # Evaluate every possible combination of node ops.
            n_repeats = int(np.sum(np.sum(adjacency_matrix, axis=1)[1:-1] > 0))
            for combination in itertools.product([CONV1X1, CONV3X3, MAXPOOL3X3], repeat=n_repeats):
                # Create node labels
                # Add some op as node 6 which isn't used, here conv1x1
                ops = [INPUT]
                combination = list(combination)
                for i in range(5):
                    if np.sum(adjacency_matrix, axis=1)[i + 1] > 0:
                        ops.append(combination.pop())
                    else:
                        ops.append(CONV1X1)
                assert (len(combination) == 0, 'Something is wrong')
                ops.append(OUTPUT)

                # Create nested list from numpy matrix
                nasbench_adjacency_matrix = adjacency_matrix.astype(np.int).tolist()

                # Assemble the model spec
                model_spec = api.ModelSpec(
                    # Adjacency matrix of the module
                    matrix=nasbench_adjacency_matrix,
                    # Operations at the vertices of the module, matches order of matrix
                    ops=ops)

                yield adjacency_matrix, ops, model_spec

    def _generate_adjacency_matrix(self, adjacency_matrix, node):
        if self._check_validity_of_adjacency_matrix(adjacency_matrix):
            # If graph from search space then yield.
            yield adjacency_matrix
        else:
            req_num_parents = self.num_parents_per_node[str(node)]
            current_num_parents = np.sum(adjacency_matrix[:, node], dtype=np.int)
            num_parents_left = req_num_parents - current_num_parents

            for parents in parent_combinations(adjacency_matrix, node, n_parents=num_parents_left):
                # Make copy of adjacency matrix so that when it returns to this stack
                # it can continue with the unmodified adjacency matrix
                adjacency_matrix_copy = copy.copy(adjacency_matrix)
                for parent in parents:
                    adjacency_matrix_copy[parent, node] = 1
                    for graph in self._generate_adjacency_matrix(adjacency_matrix=adjacency_matrix_copy, node=parent):
                        yield graph

    def _create_adjacency_matrix(self, parents, adjacency_matrix, node):
        if self._check_validity_of_adjacency_matrix(adjacency_matrix):
            # If graph from search space then yield.
            return adjacency_matrix
        else:
            for parent in parents[str(node)]:
                adjacency_matrix[parent, node] = 1
                if parent != 0:
                    adjacency_matrix = self._create_adjacency_matrix(parents=parents, adjacency_matrix=adjacency_matrix,
                                                                     node=parent)
            return adjacency_matrix

    def _create_adjacency_matrix_with_loose_ends(self, parents):
        # Create the adjacency_matrix on a per node basis
        adjacency_matrix = np.zeros([len(parents), len(parents)])
        for node, node_parents in parents.items():
            for parent in node_parents:
                adjacency_matrix[parent, int(node)] = 1
        return adjacency_matrix

    def _check_validity_of_adjacency_matrix(self, adjacency_matrix):
        """
        Checks whether a graph is a valid graph in the search space.
        1. Checks that the graph is non empty
        2. Checks that every node has the correct number of inputs
        3. Checks that if a node has outgoing edges then it should also have incoming edges
        4. Checks that input node is connected
        5. Checks that the graph has no more than 9 edges
        :param adjacency_matrix:
        :return:
        """
        # Check that the graph contains nodes
        num_intermediate_nodes = sum(np.array(np.sum(adjacency_matrix, axis=1) > 0, dtype=int)[1:-1])
        if num_intermediate_nodes == 0:
            return False

        # Check that every node has exactly the right number of inputs
        col_sums = np.sum(adjacency_matrix[:, :], axis=0)
        for col_idx, col_sum in enumerate(col_sums):
            if col_sum > 0:
                if col_sum != self.num_parents_per_node[str(col_idx)]:
                    return False

        # Check that if a node has outputs then it should also have incoming edges (apart from zero)
        col_sums = np.sum(np.sum(adjacency_matrix, axis=0) > 0)
        row_sums = np.sum(np.sum(adjacency_matrix, axis=1) > 0)
        if col_sums != row_sums:
            return False

        # Check that the input node is always connected. Otherwise the graph is disconnected.
        row_sum = np.sum(adjacency_matrix, axis=1)
        if row_sum[0] == 0:
            return False

        # Check that the graph returned has no more than 9 edges.
        num_edges = np.sum(adjacency_matrix.flatten())
        if num_edges > 9:
            return False

        return True

    def _sample_architecture(self, adjacency_matrix, node):
        req_num_parents = self.num_parents_per_node[str(node)]
        current_num_parents = np.sum(adjacency_matrix[:, node], dtype=np.int)
        num_parents_left = req_num_parents - current_num_parents
        sampled_parents = \
            random.sample(list(parent_combinations(adjacency_matrix, node, n_parents=num_parents_left)), 1)[0]
        for parent in sampled_parents:
            adjacency_matrix[parent, node] = 1
            adjacency_matrix = self._sample_architecture(adjacency_matrix, parent)
        return adjacency_matrix
