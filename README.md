## 'NAS-Bench-1Shot1: Benchmarking and Dissecting One-shot Neural Architecture Search'

To run e.g. darts on search space 1 execute from the root of the repository:

`PYTHONPATH=$PWD python optimizers/darts/train_search.py --seed=0 --save=baseline --search_space=1 --epochs=50`

To evaluate the one-shot architectures on NAS-Bench-101 download NAS-Bench-101 and insert the path to it in `nasbench_analysis/eval_darts_one_shot_model_in_nasbench.py`

Then run the following for evaluation:
`PYTHONPATH=$PWD python nasbench_analysis/eval_darts_one_shot_model_in_nasbench.py`

The NAS-Bench-101 test error and validation error for the searched architectures are written to the directory of the run and can then be analyzed using the experiment database as demonstrated in: `experiments/analysis/plot_results.py`
