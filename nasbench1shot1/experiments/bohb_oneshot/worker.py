import time
import os
import sys
import subprocess
import json
import argparse
import numpy as np

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from hpbandster.core.worker import Worker

from nasbench1shot1.core.evaluation import eval_one_shot_model as naseval


class BOHB_Worker(Worker):
    def __init__(self, eta, min_budget, max_budget, search_space,
                 algorithm, nasbench_data, seed, unrolled, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mainsourcepath = 'nasbench1shot1'
        self.eta = eta
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.nasbench_data = nasbench_data
        self.search_space = search_space
        self.arlgorithm = algorithm
        self.seed = seed
        self.unrolled = unrolled

    @staticmethod
    def complete_config(config):
        config['batch_size'] = 96
        config['momentum'] = 0.9
        #config['learning_rate'] = 0.025
        return config

    @staticmethod
    def get_config_space():
        config_space=CS.ConfigurationSpace()

        config_space.add_hyperparameter(CSH.UniformFloatHyperparameter('learning_rate',
                                                                       lower=1e-3,
                                                                       upper=1,
                                                                       default_value=0.025,
                                                                       log=True))
        config_space.add_hyperparameter(CSH.UniformFloatHyperparameter('weight_decay',
                                                                       lower=1e-5,
                                                                       upper=1e-3,
                                                                       default_value=3e-4,
                                                                       log=True))
        config_space.add_hyperparameter(CSH.UniformFloatHyperparameter('cutout_prob',
                                                                       lower=0,
                                                                       upper=1,
                                                                       default_value=0,
                                                                       log=False))

        return config_space


    @staticmethod
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


    def compute(self, config, budget, config_id, working_directory):
        return run_configuration(
            config=BOHB_Worker.complete_config(config),
            budget=int(budget),
            config_id=config_id,
            directory=working_directory
        )


    def run_configuration(config, budget, config_id, directory):
        '''
        Run DARTS for the given config
        '''
        dest_dir = os.path.join(directory, "_".join(map(str, config_id)))
        ret_dict =  { 'loss': float('inf'), 'info': None}

        try:
            bash_strings = [
                "python %s/optimizers/%s/train_search_bohb.py"%(
                    self.darts_source, self.algorithm
                ),
                "--save %s --epochs %d"%(dest_dir, int(budget)),
                "--data nasbench1shot1/data",
                "--seed {}".format(self.seed),
                "--search_space {}".format(str(self.search_space)),
                "--batch_size {batch_size}".format(**config),
                "--weight_decay {weight_decay}".format(**config),
                "--learning_rate {learning_rate}".format(**config),
                "--momentum {momentum}".format(**config),
                #"--debug",
                #"--report_freq 2",
                "--cutout_prob {cutout_prob}".format(**config)
            ]

            if self.unrolled:
                bash_strings.append('--unrolled')

            subprocess.check_call( " ".join(bash_strings), shell=True)
            info = BOHB_Worker.load_data(
                os.path.join(dest_dir,
                             'one_shot_architecture_{}.obj'.format(int(budget))),
                self.search_space,
                self.nasbench_data
            )

            with open(os.path.join(dest_dir,'config.json'), 'r') as fh:
                info['config'] = '\n'.join(fh.readlines())

            ret_dict = {'loss': info['val_error'][-1], 'info': info}

        except:
            print("Entering exception!!")
            raise

        return ret_dict

