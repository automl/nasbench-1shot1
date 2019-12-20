import time
import os, sys
import argparse

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from helper import configuration_darts

from hpbandster.core.worker import Worker

class darts_base(Worker):
    def __init__(self, eta, min_budget, max_budget, search_space,
                 algorithm, nasbench_data, seed, unrolled, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mainsourcepath = '/home/zelaa/ICLR19/darts_weight_sharing_analysis'
        self.path = os.path.join(self.mainsourcepath, 'optimizers/darts')
        self.eta = eta
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.nasbench_data = nasbench_data
        self.search_space = search_space
        self.arlgorithm = algorithm
        self.seed = seed
        self.unrolled = unrolled

    def compute(self, config, budget, config_id, working_directory):
        return(configuration_darts(config=darts_base.complete_config(config),
                                   budget=int(budget),
                                   min_budget=int(self.min_budget),
                                   eta=self.eta,
                                   config_id=config_id,
                                   search_space=self.search_space,
                                   algorithm=self.arlgorithm,
                                   nasbench_data=self.nasbench_data,
                                   seed=self.seed,
                                   directory=working_directory,
                                   darts_source=self.mainsourcepath,
                                   unrolled=self.unrolled))

    @staticmethod
    def complete_config(config):
        config['batch_size'] = 96
        config['momentum'] = 0.9
        config['learning_rate'] = 0.025
        return config

    @staticmethod
    def get_config_space():
        config_space=CS.ConfigurationSpace()

        #config_space.add_hyperparameter(CSH.UniformFloatHyperparameter('learning_rate',
        #                                                               lower=1e-3,
        #                                                               upper=1,
        #                                                               default_value=0.025,
        #                                                               log=True))
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

    @classmethod
    def data_subdir(cls):
        return 'DARTS'

