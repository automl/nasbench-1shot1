import glob
import json
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from nasbench import api

from nasbench_analysis.search_spaces.search_space_1 import SearchSpace1
from nasbench_analysis.search_spaces.search_space_2 import SearchSpace2
from nasbench_analysis.search_spaces.search_space_3 import SearchSpace3
from nasbench_analysis.utils import get_top_k, INPUT, OUTPUT, CONV1X1, NasbenchWrapper, PRIMITIVES


def softmax(weights, axis=-1):
    return F.softmax(torch.Tensor(weights), axis).data.cpu().numpy()

def eval_one_shot_model(config, model, nasbench):
    model_list = pickle.load(open(model, 'rb'))

    alphas_mixed_op = model_list[0]
    chosen_node_ops = softmax(alphas_mixed_op, axis=-1).argmax(-1)

    node_list = [PRIMITIVES[i] for i in chosen_node_ops]
    alphas_output = model_list[1]
    alphas_inputs = model_list[2:]

    if config['search_space'] == '1':
        search_space = SearchSpace1()
        num_inputs = list(search_space.num_parents_per_node.values())[3:-1]
        parents_node_3, parents_node_4 = \
            [get_top_k(softmax(alpha, axis=1), num_input) for num_input, alpha in zip(num_inputs, alphas_inputs)]
        output_parents = get_top_k(softmax(alphas_output), num_inputs[-1])
        parents = {
            '0': [],
            '1': [0],
            '2': [0, 1],
            '3': parents_node_3,
            '4': parents_node_4,
            '5': output_parents
        }
        node_list = [INPUT, *node_list, CONV1X1, OUTPUT]

    elif config['search_space'] == '2':
        search_space = SearchSpace2()
        num_inputs = list(search_space.num_parents_per_node.values())[2:]
        parents_node_2, parents_node_3, parents_node_4 = \
            [get_top_k(softmax(alpha, axis=1), num_input) for num_input, alpha in zip(num_inputs[:-1], alphas_inputs)]
        output_parents = get_top_k(softmax(alphas_output), num_inputs[-1])
        parents = {
            '0': [],
            '1': [0],
            '2': parents_node_2,
            '3': parents_node_3,
            '4': parents_node_4,
            '5': output_parents
        }
        node_list = [INPUT, *node_list, CONV1X1, OUTPUT]

    elif config['search_space'] == '3':
        search_space = SearchSpace3()
        num_inputs = list(search_space.num_parents_per_node.values())[2:]
        parents_node_2, parents_node_3, parents_node_4, parents_node_5 = \
            [get_top_k(softmax(alpha, axis=1), num_input) for num_input, alpha in zip(num_inputs[:-1], alphas_inputs)]
        output_parents = get_top_k(softmax(alphas_output), num_inputs[-1])
        parents = {
            '0': [],
            '1': [0],
            '2': parents_node_2,
            '3': parents_node_3,
            '4': parents_node_4,
            '5': parents_node_5,
            '6': output_parents
        }
        node_list = [INPUT, *node_list, OUTPUT]

    else:
        raise ValueError('Unknown search space')

    adjacency_matrix = search_space.create_nasbench_adjacency_matrix(parents)
    # Convert the adjacency matrix in format for nasbench
    adjacency_list = adjacency_matrix.astype(np.int).tolist()
    model_spec = api.ModelSpec(matrix=adjacency_list, ops=node_list)
    # Query nasbench
    data = nasbench.query(model_spec)
    valid_error, test_error, runtime, params = [], [], [], []
    for item in data:
        test_error.append(1 - item['test_accuracy'])
        valid_error.append(1 - item['validation_accuracy'])
        runtime.append(item['training_time'])
        params.append(item['trainable_parameters'])
    return test_error, valid_error, runtime, params


