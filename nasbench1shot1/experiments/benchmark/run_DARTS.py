import argparse
import glob
import json
import logging
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset

from nasbench1shot1.core.search_spaces import SearchSpace1, SearchSpace2, SearchSpace3
from nasbench1shot1.optimizers.oneshot.base import utils
from nasbench1shot1.optimizers.oneshot.base.architect import Architect
from nasbench1shot1.optimizers.oneshot.base.model_search import Network

from nasbench1shot1.optimizers.oneshot.base.searcher import OneShotModelWrapper


def main(args):
    args.save = 'experiments/darts/search_space_{}/search-{}-{}-{}-{}'.format(args.search_space, args.save,
                                                                              time.strftime("%Y%m%d-%H%M%S"), args.seed,
                                                                              args.search_space)
    utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))


    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)






if __name__ == '__main__':
    parser = argparse.ArgumentParser("cifar")
    parser.add_argument('--data', type=str, default='../data', help='location of the darts corpus')
    parser.add_argument('--batch_size', type=int, default=96, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
    parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
    parser.add_argument('--layers', type=int, default=9, help='total number of layers')
    parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument('--cutout_prob', type=float, default=1.0, help='cutout probability')
    parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
    parser.add_argument('--save', type=str, default='EXP', help='experiment name')
    parser.add_argument('--seed', type=int, default=2, help='random_ws seed')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
    parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training darts')
    parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
    parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
    parser.add_argument('--output_weights', type=bool, default=True, help='Whether to use weights on the output nodes')
    parser.add_argument('--search_space', choices=['1', '2', '3'], default='1')
    parser.add_argument('--warm_start_epochs', type=int, default=0,
                        help='Warm start one-shot model before starting architecture updates.')
    args = parser.parse_args()

    main(args)

