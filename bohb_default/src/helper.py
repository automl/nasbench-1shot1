import os
import sys
import subprocess
import json
import numpy as np
import ConfigSpace as CS

from nasbench_analysis import nasbench_eval as naseval


def load_data(model_path, search_space, nasbench_data):
    '''
    Test error, validation error, runtime and n parameters queried from
    NASBench-101
    '''
    info = {}
    index = np.random.choice(list(range(3)))
    test, valid, runtime, params = naseval.eval_one_shot_model(
        {'search_space': str(search_space)},
        model_path,
        nasbench_data
    )

    info['val_error'] = [valid[index]]
    info['test_error'] = [test[index]]
    info['runtime'] = [runtime[index]]
    info['params'] = [params[index]]

    return info


def configuration_darts(config, budget, min_budget, eta, config_id, search_space,
                        algorithm, nasbench_data, seed, directory,
                        darts_source='', unrolled=True):
    '''
    Run DARTS for the given config
    '''
    dest_dir = os.path.join(directory, "_".join(map(str, config_id)))
    ret_dict =  { 'loss': float('inf'), 'info': None}

    try:
        bash_strings = [
            "PYTHONPATH=%s python ../optimizers/%s/train_search_bohb.py"%(
                darts_source, algorithm
            ),
            "--save %s --epochs %d"%(dest_dir, int(budget)),
            "--data ../data",
            #"--unrolled", ##### remove this for first order DARTS
            "--seed {}".format(seed),
            "--search_space {}".format(str(search_space)),
            "--batch_size {batch_size}".format(**config),
            "--weight_decay {weight_decay}".format(**config),
            "--learning_rate {learning_rate}".format(**config),
            "--momentum {momentum}".format(**config),
            #"--debug",
            #"--report_freq 2",
            "--cutout_prob {cutout_prob}".format(**config)
        ]

        if unrolled:
            bash_strings.append('--unrolled')

        subprocess.check_call( " ".join(bash_strings), shell=True)
        info = load_data(
            os.path.join(dest_dir, 'one_shot_architecture_{}.obj'.format(int(budget))),
            search_space,
            nasbench_data
        )

        with open(os.path.join(dest_dir,'config.json'), 'r') as fh:
            info['config'] = '\n'.join(fh.readlines())

        ret_dict = {'loss': info['val_error'][-1], 'info': info}

    except:
        print("Entering exception!!")
        raise

    return ret_dict

