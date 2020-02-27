import os
import argparse
import pandas as pd
import numpy as np
from IPython import embed

import matplotlib
matplotlib.rcParams.update({'font.size': 14})
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, inset_axes, InsetPosition
from matplotlib.ticker import NullFormatter

import result as hpres
from util import Architecture, Model, get_trajectories, get_incumbent, plot_losses, merge_and_fill_trajectories


parser = argparse.ArgumentParser(description='fANOVA analysis')
parser.add_argument('--run_id', type=int, default=0)
parser.add_argument('--working_directory', type=str, help='directory where to'
                    ' store the live rundata', default='../bohb_output')
parser.add_argument('--space', type=int, default=3, help='NASBench space')
parser.add_argument('--seed', type=int, default=1, help='Seed')
parser.add_argument('--darts_id', type=str, default='4659076 4723555 4723571', help='Seed')
parser.add_argument('--gdas_id', type=str, default='4659117 4729794 4729810', help='Seed')
parser.add_argument('--pcdarts_id', type=str, default='4659133 4723587 4723603', help='Seed')

args = parser.parse_args()

s1_min = 0.05448716878890991
s2_min = 0.057592153549194336
s3_min = 0.05338543653488159


def trajectory_plot(opt_dict):
    fig, ax = plt.subplots(1, figsize=(8, 4.5))
    #axins = zoomed_inset_axes(ax, 1.5, loc=9)
    ax2 = plt.axes([0, 0, 1, 1])
    ip = InsetPosition(ax, [0.4, 0.6, 0.35, 0.4])
    ax2.set_axes_locator(ip)

    mark_inset(ax, ax2, loc1=4, loc2=3, fc='none', ec='0.5')

    all_trajectories = {}
    for m, all_runs in opt_dict.items():
        dfs = []
        for seed, run in enumerate(all_runs):
            # darts- bohb results
            for x in run:
                x.time_stamps['finished'] += x.info['runtime'][0]

            run = sorted(run, key=lambda x: x.time_stamps['finished'])

            #embed()

            darts_results = [[x.info['test_error'][0] - eval('s{}_min'.format(args.space)),
                              x.time_stamps['finished']-x.time_stamps['started']] for x
                             in run]
            darts_results = np.array(darts_results)
            cumulated_runtimes = [np.sum(darts_results[:, 1][0:i+1]) for i in
                                  range(len(darts_results[:, 1]))]

            time_stamps, test_regret = get_incumbent(darts_results[:, 0],
                                                     cumulated_runtimes)
            print(seed, ' MIN: ', min(test_regret))

            df = pd.DataFrame({str(seed): test_regret}, index=time_stamps)
            dfs.append(df)

        #embed()
        df = merge_and_fill_trajectories(dfs, default_value=None)
        if df.empty:
            continue
        print(m, df.shape)

        all_trajectories[m] = {
            'time_stamps': np.array(df.index),
            'losses': np.array(df.T)
        }

    # discrete optimizers
    discrete_opt_results = get_trajectories(args, eval('s{}_min'.format(args.space)),
                                            path='../../experiments/discrete_optimizers',
                                            methods=['RE', 'RS', 'RL', 'SMAC', 'HB',
                                                     'BOHB', 'TPE'])
    all_trajectories.update(discrete_opt_results)

    plot_losses(fig, ax, ax2, all_trajectories, regret=False, plot_mean=True,
                marker_size=8)

    x1, x2, y1, y2 = 3e5, 6e5, 5e-3, 8e-3 # specify the limits
    ax2.set_xlim(x1, x2) # apply the x-limits
    ax2.set_ylim(y1, y2)
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    #ax2.xaxis.set_major_formatter(NullFormatter())
    #ax2.yaxis.set_minor_formatter(NullFormatter())
    #ax2.tick_params(axis=u'both', which=u'both',length=0)
    #ax2.get_xaxis().set_ticks([])
    #ax2.get_yaxis().set_ticks([])

    ax.set_xscale('log')
    ax.set_xlim([4e2, 6e6]) #s1
    ax.set_yscale('log')
    ax.set_ylim([3e-3, 5e-1]) #s3

    ax.set_ylabel('test regret')
    ax.set_xlabel('simulated wallclock time [s]')
    ax.legend(fontsize=8, loc=1)
    ax.set_title("Space %d"%(args.space))
    ax.grid(True, which="both",ls="-")
    ax2.grid(True, which="both",ls="-")
    plt.tight_layout()

    os.makedirs('./incumbents', exist_ok=True)
    fig_name = './incumbents'+'/s%d-run%d-seed%d.png'%(
        args.space, int(args.darts_id), args.seed
    )
    plt.savefig(fig_name)
    plt.show()


if __name__=='__main__':
    dirs = ['{}/search_space_{}/darts/'.format(args.working_directory,
                                              args.space),
            '{}/search_space_{}/pc_darts/'.format(args.working_directory,
                                                          args.space),
            '{}/search_space_{}/gdas/'.format(args.working_directory,
                                                          args.space)]

    opt_dict = {}
    for m, path in zip(['BOHB-DARTS', 'BOHB-PC-DARTS', 'BOHB-GDAS'], dirs):
        if m == 'BOHB-DARTS':
            ids = [int(x) for x in args.darts_id.split()]
        if m == 'BOHB-PC-DARTS':
            ids = [int(x) for x in args.pcdarts_id.split()]
        if m == 'BOHB-GDAS':
            ids = [int(x) for x in args.gdas_id.split()]

        runs = []
        for seed in range(1, len(ids)+1):
            logs_dir = path + 'run{}-seed{}'.format(ids[seed-1], seed)
            res = hpres.logged_results_to_HB_result(logs_dir)
            run = list(filter(lambda r: not (r.info is None or r.loss is None),
                              res.get_all_runs()))
            runs.append(run)

        opt_dict[m] = runs

    trajectory_plot(opt_dict)
