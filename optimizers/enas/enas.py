import argparse
import inspect
import json
import logging
import os
import pickle
import sys
import time

import numpy as np
import torch

from nasbench_analysis.search_spaces.search_space_1 import SearchSpace1
from nasbench_analysis.search_spaces.search_space_2 import SearchSpace2
from nasbench_analysis.search_spaces.search_space_3 import SearchSpace3
from optimizers.enas.enas_child import ENASChild
from optimizers.enas.micro_controller import Controller

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)


class ENAS:
    def __init__(self, args, model, controller, seed, save_dir, search_space):
        self.save_dir = save_dir

        self.model = model
        self.controller = controller
        self.seed = seed

        self.iters = 0

    def save(self, epoch):
        # Save one-shot model
        self.model.save(epoch=epoch)

        # Save controller
        torch.save(self.controller, os.path.join(self.save_dir, 'controller_epoch_{}.pt'.format(epoch)))

    def run(self):
        for epoch in range(args.epochs):
            # 1. Train the one-shot model
            self.model.train_model(epoch=epoch)

            # 2. Train the controller
            self.model.train_controller()

            # Evaluate the model
            self.get_eval_arch(epoch=epoch)

            # Save the one-shot model
            self.save(epoch)

    def get_eval_arch(self, epoch, rounds=None):
        best_rounds = []
        sample_vals = []
        for _ in range(1000):
            # Sample an architecture from the controller
            arch, _, _ = self.model.controller()
            try:
                ppl = self.model.evaluate(arch)
            except Exception as e:
                ppl = 1000000
            logging.info(arch)
            logging.info('objective_val: %.3f' % ppl)
            sample_vals.append((arch, ppl))

        sample_vals = sorted(sample_vals, key=lambda x: x[1])

        # Save sample validations
        with open(os.path.join(self.save_dir, 'sample_val_architecture_epoch_{}.obj'.format(epoch)),
                  'wb') as f:
            pickle.dump(sample_vals, f)

        full_vals = []
        if 'split' in inspect.getfullargspec(self.model.evaluate).args:
            for i in range(5):
                arch = sample_vals[i][0]
                try:
                    ppl = self.model.evaluate(arch, split='valid')
                except Exception as e:
                    print(e)
                    ppl = 1000000
                full_vals.append((arch, ppl))
            full_vals = sorted(full_vals, key=lambda x: x[1])
            logging.info('best arch: %s, best arch valid performance: %.3f' % (
                ' '.join([str(i) for i in full_vals[0][0]]), full_vals[0][1]))
            best_rounds.append(full_vals[0])
        else:
            best_rounds.append(sample_vals[0])

        # Save the fully evaluated architectures
        with open(os.path.join(self.save_dir,
                               'full_val_architecture_epoch_{}.obj'.format(epoch)), 'wb') as f:
            pickle.dump(full_vals, f)
        return best_rounds


def main(args):
    # Fill in with root output path
    root_dir = os.getcwd()
    print('root_dir', root_dir)
    if args.save_dir is None:
        save_dir = os.path.join(root_dir, 'experiments/enas/ss_{}_{}_{}'.format(time.strftime("%Y%m%d-%H%M%S"),
                                                                                args.search_space, args.seed))
    else:
        save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if args.eval_only:
        assert args.save_dir is not None

    # Dump the config of the run folder
    with open(os.path.join(save_dir, 'config.json'), 'w') as fp:
        json.dump(args.__dict__, fp)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(save_dir, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    logging.info(args)

    if args.search_space == '1':
        search_space = SearchSpace1()
    elif args.search_space == '2':
        search_space = SearchSpace2()
    elif args.search_space == '3':
        search_space = SearchSpace3()
    else:
        raise ValueError('Unknown search space')

    if args.benchmark == 'ptb':
        raise ValueError('PTB not supported.')
    else:
        data_size = 25000
        time_steps = 1

    B = int(args.epochs * data_size / args.batch_size / time_steps)
    if args.benchmark == 'cnn':
        controller = Controller(search_space, args).cuda()
        model = ENASChild(controller, save_path=save_dir, seed=args.seed, batch_size=args.batch_size,
                          grad_clip=args.grad_clip, epochs=args.epochs,
                          num_intermediate_nodes=search_space.num_intermediate_nodes,
                          search_space=search_space, init_channels=args.init_channels, cutout=args.cutout)
    else:
        raise ValueError('Benchmarks other cnn on cifar are not available')

    searcher = ENAS(args=args, model=model, controller=controller, seed=args.seed, save_dir=save_dir,
                    search_space=search_space)

    if not args.eval_only:
        searcher.run()
        archs = searcher.get_eval_arch()
    else:
        np.random.seed(args.seed + 1)
        archs = searcher.get_eval_arch(2)
    logging.info(archs)
    arch = ' '.join([str(a) for a in archs[0][0]])
    with open('/tmp/arch', 'w') as f:
        f.write(arch)
    return arch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args for ENAS')
    parser.add_argument('--benchmark', dest='benchmark', type=str, default='cnn')
    parser.add_argument('--seed', dest='seed', type=int, default=100)
    parser.add_argument('--epochs', dest='epochs', type=int, default=50)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)
    parser.add_argument('--grad_clip', dest='grad_clip', type=float, default=0.25)
    parser.add_argument('--save_dir', dest='save_dir', type=str, default=None)
    parser.add_argument('--eval_only', dest='eval_only', type=int, default=0)
    # CIFAR-10 only argument.  Use either 16 or 24 for the settings for random_ws search
    # with weight-sharing used in our experiments.
    parser.add_argument('--init_channels', dest='init_channels', type=int, default=16)
    parser.add_argument('--shared_initial_steps', dest='init_channels', type=int, default=10)
    parser.add_argument('--search_space', choices=['1', '2', '3'], default='1')
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--cuda', dest='cuda', type=bool, default=True)

    parser.add_argument('--child_num_layers', type=int, default=6)
    parser.add_argument('--child_num_ops', type=int, default=3)
    parser.add_argument('--child_num_cells', type=int, default=5)

    parser.add_argument('--controller_lr', type=float, default=0.0035)
    parser.add_argument('--controller_tanh_constant', type=float, default=2.5)
    parser.add_argument('--controller_op_tanh_reduce', type=float, default=5.0)

    parser.add_argument('--lstm_size', type=int, default=64)
    parser.add_argument('--lstm_num_layers', type=int, default=1)
    parser.add_argument('--lstm_keep_prob', type=float, default=0)
    parser.add_argument('--temperature', type=float, default=5.0)

    parser.add_argument('--entropy_weight', type=float, default=0.1)
    parser.add_argument('--bl_dec', type=float, default=0.99)
    args = parser.parse_args()

    main(args)
