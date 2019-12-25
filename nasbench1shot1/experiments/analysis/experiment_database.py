import json
import os
import pickle

import numpy as np

from experiments.analysis.utils import parse_log, read_in_correlation
from nasbench_analysis.search_spaces.search_space_1 import SearchSpace1
from nasbench_analysis.search_spaces.search_space_2 import SearchSpace2
from nasbench_analysis.search_spaces.search_space_3 import SearchSpace3


def search_space_id_to_obj(id):
    if int(id) == 1:
        return SearchSpace1()
    elif int(id) == 2:
        return SearchSpace2()
    elif int(id) == 3:
        return SearchSpace3()
    else:
        raise ValueError('Search space unknown.')


def get_directory_list(path):
    """Find directory containing config.json files"""
    directory_list = []
    # return nothing if path is a file
    if os.path.isfile(path):
        return []
    # add dir to directorylist if it contains .json files
    if len([f for f in os.listdir(path) if f == 'config.json']) > 0:
        directory_list.append(path)
    for d in os.listdir(path):
        new_path = os.path.join(path, d)
        if os.path.isdir(new_path):
            directory_list += get_directory_list(new_path)
    return directory_list


def get_key_from_scalar_configs(configs, key):
    metrics_to_stack = [list(config['scalars'][key]) for config in configs]
    shortest_metric = min([len(m) for m in metrics_to_stack])

    if 'validation_errors' == key or 'test_errors' == key:
        search_space = search_space_id_to_obj(configs[0]['search_space'])
        if 'test' in key:
            minimum = search_space.test_min_error
        elif 'valid' in key:
            minimum = search_space.valid_min_error
        else:
            raise ValueError('incorrect name in key')
    else:
        minimum = 0

    return np.mean(np.stack([metric[:shortest_metric] for metric in metrics_to_stack], axis=0), axis=-1) - minimum


class ExperimentDatabase:
    def __init__(self, root_dir):
        """Load all directories with trainings."""
        self._load(root_dir=root_dir)

    def query(self, conditions):
        searched_config = []
        for config in self._database:
            # Only select config if all conditions are satisfied
            conds_satisfied = [config.get(cond_key, None) == cond_val for cond_key, cond_val in conditions.items()]
            if all(conds_satisfied):
                searched_config.append(config)

        return searched_config

    def query_correlation(self, conditions):
        searched_config = []
        for config in self._database:
            # Only select config if all conditions are satisfied
            conds_satisfied = [config.get(cond_key, None) == cond_val for cond_key, cond_val in conditions.items()]
            if all(conds_satisfied):
                if config['scalars']['correlation_total'] is not None:
                    searched_config.append(config)

        return searched_config

    def _load(self, root_dir):
        self._database = []
        for directory in get_directory_list(root_dir):
            try:
                self._database.append(self._get_run_dictionary(directory))
            except Exception as e:
                print('Error occurred in loading', directory, e)

    def _get_run_dictionary(self, path):
        with open(os.path.join(path, 'config.json')) as fp:
            config = json.load(fp)

        with open(os.path.join(path, 'one_shot_validation_errors.obj'), 'rb') as fp:
            validation_errors = pickle.load(fp)

        with open(os.path.join(path, 'one_shot_test_errors.obj'), 'rb') as fp:
            test_errors = pickle.load(fp)

        one_shot_validation_errors, one_shot_training_errors = parse_log(path)
        correlation_total, correlation_top = read_in_correlation(path, config)

        config['scalars'] = {}
        config['scalars']['validation_errors'] = validation_errors
        config['scalars']['test_errors'] = test_errors
        config['scalars']['one_shot_validation_errors'] = one_shot_validation_errors
        config['scalars']['one_shot_training_errors'] = one_shot_training_errors
        config['scalars']['correlation_total'] = correlation_total
        config['scalars']['correlation_top'] = correlation_top

        return config


def main():
    experiment_database = ExperimentDatabase(root_dir='experiments/darts')
    pass


if __name__ == '__main__':
    main()
