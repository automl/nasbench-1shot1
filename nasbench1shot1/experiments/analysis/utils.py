import collections
import glob
import os
import pickle
import re

import numpy as np
import scipy.stats as stats

from nasbench_analysis.eval_darts_one_shot_model_in_nasbench import natural_keys


def parse_log(path):
    f = open(os.path.join(path, 'log.txt'), 'r')
    # Read in the relevant information
    train_accuracies = []
    valid_accuracies = []
    for line in f:
        if 'train_acc' in line:
            train_accuracies.append(line)
        if 'valid_acc' in line:
            valid_accuracies.append(line)

    valid_error = [[1 - 1 / 100 * float(re.search('valid_acc ([-+]?[0-9]*\.?[0-9]+)', line).group(1))] for line in
                   valid_accuracies]
    train_error = [[1 - 1 / 100 * float(re.search('train_acc ([-+]?[0-9]*\.?[0-9]+)', line).group(1))] for line in
                   train_accuracies]

    return valid_error, train_error


def compute_spearman_correlation_top_1000(one_shot_test_error, nb_test_error):
    sort_by_one_shot = lambda os, nb: [[y, x] for (y, x) in sorted(zip(os, nb), key=lambda pair: pair[0])]
    correlation_at_epoch = []
    for one_shot_test_error_on_epoch in one_shot_test_error:
        sorted_by_os_error = np.array(sort_by_one_shot(one_shot_test_error_on_epoch[0], nb_test_error))
        correlation_at_epoch.append(
            stats.spearmanr(sorted_by_os_error[:, 0][:1000], sorted_by_os_error[:, 1][:1000]).correlation)
    return correlation_at_epoch


def compute_spearman_correlation(one_shot_test_error, nb_test_error):
    correlation_at_epoch = []
    for one_shot_test_error_on_epoch in one_shot_test_error:
        correlation_at_epoch.append(stats.spearmanr(one_shot_test_error_on_epoch[0], nb_test_error).correlation)
    return correlation_at_epoch


def read_in_correlation(path, config):
    correlation_files = glob.glob(os.path.join(path, 'correlation_*.obj'))
    # If no correlation files available
    if len(correlation_files) == 0:
        return None, None
    else:
        read_file_list_with_pickle = lambda file_list: [pickle.load(open(file, 'rb')) for file in file_list]
        correlation_files.sort(key=natural_keys)

        one_shot_test_errors = glob.glob(os.path.join(path, 'one_shot_test_errors_*'))
        one_shot_test_errors.sort(key=natural_keys)
        one_shot_test_errors = read_file_list_with_pickle(one_shot_test_errors)

        if config['search_space'] == '1':
            nb_test_errors_per_epoch = pickle.load(
                open('experiments/analysis/data/test_errors_per_epoch_ss1.obj', 'rb'))
        elif config['search_space'] == '2':
            nb_test_errors_per_epoch = pickle.load(
                open('experiments/analysis/data/test_errors_per_epoch_ss2.obj', 'rb'))
        elif config['search_space'] == '3':
            nb_test_errors_per_epoch = pickle.load(
                open('experiments/analysis/data/test_errors_per_epoch_ss3.obj', 'rb'))
        else:
            raise ValueError('Unknown search space')
        correlation_per_epoch_total = {
            epoch: compute_spearman_correlation(one_shot_test_errors, nb_test_errors_at_epoch) for
            epoch, nb_test_errors_at_epoch in nb_test_errors_per_epoch.items()}

        correlation_per_epoch_top = {
            epoch: compute_spearman_correlation_top_1000(one_shot_test_errors, nb_test_errors_at_epoch) for
            epoch, nb_test_errors_at_epoch in nb_test_errors_per_epoch.items()}

        return collections.OrderedDict(sorted(correlation_per_epoch_total.items())), collections.OrderedDict(
            sorted(correlation_per_epoch_top.items()))

