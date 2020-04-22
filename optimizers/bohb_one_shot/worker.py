import os
import subprocess
import numpy as np

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from hpbandster.core.worker import Worker

from nasbench_analysis import eval_darts_one_shot_model_in_nasbench as naseval


class one_shot_worker(Worker):
    def __init__(self, eta, min_budget, max_budget, search_space,
                 algorithm, nasbench_data, seed, unrolled, cs, *args, **kwargs):
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
        self.cs = cs

    def compute(self, config, budget, config_id, working_directory):
        return one_shot_worker._run_configuration(
            config=one_shot_worker.complete_config(config, self.cs),
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
            unrolled=self.unrolled
        )

    @staticmethod
    def complete_config(config, cs=2):
        config['init_channels'] = 16
        config['layers'] = 9
        if cs == 1:
            config['batch_size'] = 96
            config['momentum'] = 0.9
            config['learning_rate'] = 0.025
        elif cs == 2:
            config['batch_size'] = 96
            config['momentum'] = 0.9
        return config

    def get_config_space(self):
        config_space=CS.ConfigurationSpace()

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
        if self.cs == 2:
            config_space.add_hyperparameter(CSH.UniformFloatHyperparameter('learning_rate',
                                                                           lower=1e-3,
                                                                           upper=1,
                                                                           default_value=0.025,
                                                                           log=True))
        elif self.cs == 3:
            config_space.add_hyperparameter(CSH.UniformFloatHyperparameter('learning_rate',
                                                                           lower=1e-3,
                                                                           upper=1,
                                                                           default_value=0.025,
                                                                           log=True))
            config_space.add_hyperparameter(CSH.UniformFloatHyperparameter('arch_learning_rate',
                                                                           lower=1e-5,
                                                                           upper=1e-2,
                                                                           log=True))
            config_space.add_hyperparameter(CSH.UniformFloatHyperparameter('arch_weight_decay',
                                                                           lower=1e-4,
                                                                           upper=1e-1,
                                                                           log=True))
            config_space.add_hyperparameter(CSH.UniformFloatHyperparameter('momentum',
                                                                           lower=0,
                                                                           upper=0.99,
                                                                           log=False))
            config_space.add_hyperparameter(CSH.UniformIntegerHyperparameter('cutout_length',
                                                                             lower=2,
                                                                             upper=32,
                                                                             log=True))
            config_space.add_hyperparameter(CSH.UniformIntegerHyperparameter('batch_size',
                                                                             lower=16,
                                                                             upper=128,
                                                                             log=False))
            config_space.add_hyperparameter(CSH.UniformIntegerHyperparameter('grad_clip',
                                                                             lower=1,
                                                                             upper=9,
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


    @staticmethod
    def _run_configuration(config, budget, min_budget, eta, config_id, search_space,
                            algorithm, nasbench_data, seed, directory,
                            darts_source='', unrolled=True, cs=1):
        '''
        Run search for the given config
        '''
        dest_dir = os.path.join(directory, "_".join(map(str, config_id)))
        ret_dict =  { 'loss': float('inf'), 'info': None}

        bash_strings = [
            "python optimizers/%s/train_search_bohb.py"%(
                algorithm
            ),
            "--save %s --epochs %d"%(dest_dir, int(budget)),
            "--data ./data",
            "--seed {}".format(seed),
            "--search_space {}".format(str(search_space)),
            "--init_channels {init_channels}".format(**config),
            "--layers {layers}".format(**config),
            "--batch_size {batch_size}".format(**config),
            "--weight_decay {weight_decay}".format(**config),
            "--learning_rate {learning_rate}".format(**config),
            "--momentum {momentum}".format(**config),
            "--cutout_prob {cutout_prob}".format(**config)
        ]

        if cs == 3:
            bash_strings += [
                "--grad_clip {grad_clip}".format(**config),
                "--cutout_length {cutout_length}".format(**config),
                "--arch_learning_rate {arch_learning_rate}".format(**config),
                "--arch_weight_decay {arch_weight_decay}".format(**config)
            ]

        if unrolled:
            bash_strings.append('--unrolled')

        subprocess.check_call( " ".join(bash_strings), shell=True)
        info = one_shot_worker.load_data(
            os.path.join(dest_dir, 'one_shot_architecture_{}.obj'.format(int(budget))),
            search_space,
            nasbench_data
        )

        with open(os.path.join(dest_dir,'config.json'), 'r') as fh:
            info['config'] = '\n'.join(fh.readlines())

        ret_dict = {'loss': info['val_error'][-1], 'info': info}
        return ret_dict


