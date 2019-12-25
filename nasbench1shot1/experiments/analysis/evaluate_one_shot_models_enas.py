import argparse
import json
import os
import pickle

import numpy as np
import torch
from nasbench import api

from nasbench_analysis.search_spaces.search_space_1 import SearchSpace1
from nasbench_analysis.search_spaces.search_space_2 import SearchSpace2
from nasbench_analysis.search_spaces.search_space_3 import SearchSpace3
from nasbench_analysis.utils import CONV1X1, INPUT, OUTPUT, upscale_to_nasbench_format
from nasbench_analysis.utils import NasbenchWrapper
from optimizers.random_search_with_weight_sharing.darts_wrapper_discrete import DartsWrapper


def correlation_between_one_shot_nb(model_path, config, epoch):
    if config['search_space'] == '1':
        search_space = SearchSpace1()
    elif config['search_space'] == '2':
        search_space = SearchSpace2()
    elif config['search_space'] == '3':
        search_space = SearchSpace3()
    else:
        raise ValueError('Unknown search space')
    model = DartsWrapper(save_path=model_path, seed=0, batch_size=128, grad_clip=5, epochs=200,
                         num_intermediate_nodes=search_space.num_intermediate_nodes, search_space=search_space,
                         cutout=False)
    discrete = True
    normalize = False

    model.load(epoch=epoch)
    controller = torch.load(os.path.join(model_path, 'controller_epoch_{}.pt'.format(epoch)))

    nb_test_errors = {'4': [], '12': [], '36': [], '108': []}
    nb_valid_errors = {'4': [], '12': [], '36': [], '108': []}
    one_shot_test_errors = []

    for idx in range(100):
        (adjacency_matrix_ss, ops_ss), _, _ = controller()

        print(adjacency_matrix_ss, ops_ss)

        one_shot_test_error = model.evaluate_test((adjacency_matrix_ss, ops_ss), split='test', discrete=discrete,
                                                  normalize=normalize)
        one_shot_test_errors.extend(np.repeat(one_shot_test_error, 3))

        # # Query NASBench
        # Create nested list from numpy matrix
        if str(config['search_space']) == '1' or str(config['search_space']) == '2':
            adjacency_matrix_ss = upscale_to_nasbench_format(adjacency_matrix_ss)
            # Remove input, output and 5th node
            ops_ss.append(CONV1X1)

        nasbench_adjacency_matrix = adjacency_matrix_ss.astype(np.int).tolist()

        ops_ss.insert(0, INPUT)
        ops_ss.append(OUTPUT)

        # Assemble the model spec
        model_spec = api.ModelSpec(
            # Adjacency matrix of the module
            matrix=nasbench_adjacency_matrix,
            # Operations at the vertices of the module, matches order of matrix
            ops=ops_ss)
        for nb_epoch_budget in [4, 12, 36, 108]:
            data = nasbench.query(model_spec=model_spec, epochs=nb_epoch_budget)
            nb_test_errors[str(nb_epoch_budget)].extend([1 - item['test_accuracy'] for item in data])
            nb_valid_errors[str(nb_epoch_budget)].extend([1 - item['validation_accuracy'] for item in data])
        # print('NB', nb_test_errors[-1], 'OS', one_shot_test_errors[-1], 'weights', model.model.arch_parameters())

    # correlation = np.corrcoef(one_shot_test_errors, nb_test_errors)[0, -1]
    return None, nb_test_errors, nb_valid_errors, one_shot_test_errors


def eval_directory_on_epoch(path, epoch):
    """Evaluates all one-shot architecture methods in the directory."""
    # Read in config
    with open(os.path.join(path, 'config.json')) as fp:
        config = json.load(fp)
    correlations, nb_test_errors, nb_valid_errors, one_shot_test_errors = \
        correlation_between_one_shot_nb(model_path=path,
                                        config=config,
                                        epoch=epoch)

    with open(os.path.join(path, 'correlation_{}.obj'.format(epoch)), 'wb') as fp:
        pickle.dump(correlations, fp)
        print(os.path.join(path, 'correlation_{}.obj'.format(epoch)))

    with open(os.path.join(path, 'nb_test_errors_{}.obj'.format(epoch)), 'wb') as fp:
        pickle.dump(nb_test_errors, fp)

    with open(os.path.join(path, 'nb_valid_errors_{}.obj'.format(epoch)), 'wb') as fp:
        pickle.dump(nb_valid_errors, fp)

    with open(os.path.join(path, 'one_shot_test_errors_{}.obj'.format(epoch)), 'wb') as fp:
        pickle.dump(one_shot_test_errors, fp)


def main():
    # Load NASBench
    eval_directory_on_epoch(args.model_path, args.epoch)


parser = argparse.ArgumentParser("correlation_analysis")
parser.add_argument('--data', type=str, default='../data', help='location of the darts corpus')
parser.add_argument('--model_path', default="experiments/darts/search_space_1/search-baseline-20190821-171946-0-1",
                    help='Path to where the models are stored.')
parser.add_argument('--epoch', type=int, help='Epoch')
args = parser.parse_args()

if __name__ == '__main__':
    nasbench = NasbenchWrapper('/home/siemsj/projects/darts_weight_sharing_analysis/nasbench_full.tfrecord')
    main()
