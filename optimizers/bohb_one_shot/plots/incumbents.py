import os
import argparse
import pandas as pd
import numpy as np
from IPython import embed

import matplotlib
matplotlib.rcParams.update({'font.size': 14})
import matplotlib.pyplot as plt

import result as hpres
from util import Architecture, Model, get_trajectories, get_incumbent, plot_losses, merge_and_fill_trajectories


parser = argparse.ArgumentParser(description='fANOVA analysis')
parser.add_argument('--run_id', type=int, default=0)
parser.add_argument('--working_directory', type=str, help='directory where to'
                    ' store the live rundata', default='../bohb_output')
parser.add_argument('--space', type=int, default=3, help='NASBench space')
parser.add_argument('--seed', type=int, default=1, help='Seed')
parser.add_argument('--darts_id', type=int, default=1, help='Seed')
parser.add_argument('--gdas_id', type=int, default=1, help='Seed')
parser.add_argument('--pcdarts_id', type=int, default=1, help='Seed')

args = parser.parse_args()

s1_min = 0.05448716878890991
s2_min = 0.057592153549194336
s3_min = 0.05338543653488159


def trajectory_plot(opt_dict):
    fig, ax = plt.subplots(1, figsize=(8, 4.5))

    for m, all_runs in opt_dict.items():
        # darts- bohb results
        for x in all_runs:
            x.time_stamps['finished'] += x.info['runtime'][0]

        all_runs = sorted(all_runs, key=lambda x: x.time_stamps['finished'])

        #embed()

        darts_results = [[x.info['test_error'][0] - eval('s{}_min'.format(args.space)),
                          x.time_stamps['finished']-x.time_stamps['started']] for x
                         in all_runs]
        darts_results = np.array(darts_results)
        cumulated_runtimes = [np.sum(darts_results[:, 1][0:i+1]) for i in
                              range(len(darts_results[:, 1]))]

        time_stamps, test_regret = get_incumbent(darts_results[:, 0],
                                                 cumulated_runtimes)
        df = pd.DataFrame({'a': test_regret}, index=time_stamps)
        darts_traj = {m: {
            'time_stamps': np.array(df.index),
            'losses': np.array(df.T)
        }}
        plot_losses(fig, ax, None, darts_traj, regret=False,
                    plot_mean=True)

    # RE and RS
    re_results = get_trajectories(args, eval('s{}_min'.format(args.space)),
                                  path='../../experiments/discrete_optimizers',
                                  methods=['RE', 'RS', 'RL', 'SMAC', 'HB',
                                           'BOHB', 'TPE'])
    plot_losses(fig, ax, None, re_results, regret=False, plot_mean=True)


    ax.set_xscale('log')
    ax.set_xlim(left=4e2) #s1
    ax.set_yscale('log')
    #ax.set_ylim([0.002, 2e-1]) #s3

    ax.set_ylabel('test regret')
    ax.set_xlabel('simulated wallclock time [s]')
    plt.legend(fontsize=10)
    plt.title("Space %d"%(args.space))
    plt.grid(True, which="both",ls="-")
    plt.tight_layout()

    os.makedirs('./incumbents', exist_ok=True)
    fig_name = './incumbents'+'/s%d-run%d-seed%d.png'%(
        args.space, args.darts_id, args.seed
    )
    plt.savefig(fig_name)
    #plt.show()


if __name__=='__main__':
    darts_logs_dir = '{}/search_space_{}/darts/run{}-seed1'.format(
        args.working_directory, args.space, args.darts_id
    )
    #pcdarts_logs_dir = '{}/search_space_{}/pc_darts/run{}-seed1'.format(
    #    args.working_directory, args.space, args.pcdarts_id
    #)
    #gdas_logs_dir = '{}/search_space_{}/gdas/run{}-seed1'.format(
    #    args.working_directory, args.space, args.gdas_id
    #)

    res_1 = hpres.logged_results_to_HB_result(darts_logs_dir)
    #res_2 = hpres.logged_results_to_HB_result(pcdarts_logs_dir)
    #res_3 = hpres.logged_results_to_HB_result(gdas_logs_dir)

    runs_1 = list(filter(lambda r: not (r.info is None or r.loss is None),
                       res_1.get_all_runs()))
    #runs_2 = list(filter(lambda r: not (r.info is None or r.loss is None),
    #                   res_2.get_all_runs()))
    #runs_3 = list(filter(lambda r: not (r.info is None or r.loss is None),
    #                   res_3.get_all_runs()))

    opt_dict = {
        'BOHB-DARTS': runs_1,
    #    'BOHB-PC-DARTS': runs_2,
    #    'BOHB-GDAS': runs_3
    }

    trajectory_plot(opt_dict)
