import argparse

from nasbench1shot1.core.wrappers import NasbenchWrapper
from nasbench1shot1.core.evaluation import get_directory_list, eval_directory
from nasbench1shot1.core.utils import natural_keys


def main(args):
    nasbench = NasbenchWrapper(dataset_file=args.nasbench_path)

    directories = get_directory_list(args.model_path)
    directories.sort(key=natural_keys)
    for directory in directories:
        try:
            eval_directory(directory, nasbench, args.optimizer,
                           basename=args.basename)
        except Exception as e:
            print('error', e, directory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("oneshot evaluation")
    parser.add_argument('--nasbench_path',
                        type=str,
                        default='nasbench1shot1/data/nasbench_data/108_e/nasbench_full.tfrecord',
                        help='location of the nasbench-101 tfrecord data')
    parser.add_argument('--model_path',
                        type=str,
                        default='nasbench1shot1/experiments/darts/',
                        help='path where the one-shot models are saved')
    parser.add_argument('--optimizer',
                        type=str,
                        default='darts',
                        help='one-shot optimizer name')
    parser.add_argument('--basename',
                        type=str,
                        default='one_shot_architecture_*.obj',
                        help='basename of the .obj files containing the'
                        ' one-shot models.') # full_val_architecture_epoch_*.obj
    args = parser.parse_args()

    main()

