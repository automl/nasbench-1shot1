import glob
import json
import os
import pickle

import numpy as np
from nasbench import api

from nasbench_analysis.utils import INPUT, OUTPUT, CONV1X1, NasbenchWrapper, upscale_to_nasbench_format, natural_keys


def get_directory_list(path):
    """Find directory containing config.json files"""
    directory_list = []
    # return nothing if path is a file
    if os.path.isfile(path):
        return []
    # add dir to directorylist if it contains .json files
    if len([f for f in os.listdir(path) if f == 'config.json' or 'sample_val_architecture' in f]) > 0:
        directory_list.append(path)
    for d in os.listdir(path):
        new_path = os.path.join(path, d)
        if os.path.isdir(new_path):
            directory_list += get_directory_list(new_path)
    return directory_list


def eval_random_ws_model(config, model):
    model_list = pickle.load(open(model, 'rb'))
    adjacency_matrix, node_list = model_list[0][0]
    if int(config['search_space']) == int('1'):
        adjacency_matrix = upscale_to_nasbench_format(adjacency_matrix)
        node_list = [INPUT, *node_list, CONV1X1, OUTPUT]
    elif int(config['search_space']) == int('2'):
        adjacency_matrix = upscale_to_nasbench_format(adjacency_matrix)
        node_list = [INPUT, *node_list, CONV1X1, OUTPUT]
    elif int(config['search_space']) == int('3'):
        node_list = [INPUT, *node_list, OUTPUT]
    else:
        raise ValueError('Unknown search space')

    # Convert the adjacency matrix in format for nasbench
    adjacency_list = adjacency_matrix.astype(np.int).tolist()
    model_spec = api.ModelSpec(matrix=adjacency_list, ops=node_list)
    # Query nasbench
    data = nasbench.query(model_spec)
    valid_error, test_error = [], []
    for item in data:
        test_error.append(1 - item['test_accuracy'])
        valid_error.append(1 - item['validation_accuracy'])
    return test_error, valid_error


def eval_directory(path):
    """Evaluates all one-shot architecture methods in the directory."""
    # Read in config
    with open(os.path.join(path, 'config.json')) as fp:
        config = json.load(fp)
    # Accumulate all one-shot models
    random_ws_archs = glob.glob(os.path.join(path, 'full_val_architecture_epoch_*.obj'))
    # Sort them by date
    random_ws_archs.sort(key=natural_keys)
    # Eval all models on nasbench
    test_errors = []
    valid_errors = []
    for model in random_ws_archs:
        test, valid = eval_random_ws_model(config=config, model=model)
        test_errors.append(test)
        valid_errors.append(valid)

    with open(os.path.join(path, 'one_shot_validation_errors.obj'), 'wb') as fp:
        pickle.dump(valid_errors, fp)

    with open(os.path.join(path, 'one_shot_test_errors.obj'), 'wb') as fp:
        pickle.dump(test_errors, fp)


def main():
    for directory in get_directory_list("experiments/enas/"):
        try:
            eval_directory(directory)
        except Exception as e:
            print('error', e, directory)


if __name__ == '__main__':
    nasbench = NasbenchWrapper(
        dataset_file='/home/siemsj/projects/darts_weight_sharing_analysis/nasbench_analysis/nasbench_data/108_e/nasbench_full.tfrecord')
    main()
