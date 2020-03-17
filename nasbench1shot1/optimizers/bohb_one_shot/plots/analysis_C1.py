import os
import itertools
import argparse
#from IPython import embed

import result as hpres
import hpbandster.visualization as hpvis

import numpy as np
import scipy.stats as sps

import matplotlib
matplotlib.rcParams.update({'font.size': 14})
import matplotlib.pyplot as plt

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

import fanova
import fanova.visualizer


parser = argparse.ArgumentParser(description='fANOVA analysis')
parser.add_argument('--run_id', type=int, default=0)
parser.add_argument('--working_directory', type=str, help='directory where to'
                    ' store the live rundata', default='../bohb_output')
parser.add_argument('--space', type=int, default=1, help='NASBench space')
parser.add_argument('--seed', type=int, default=1, help='Seed')
args = parser.parse_args()


def extract_HB_learning_curves(runs):
    """
    function to get the hyperband learning curves.
    This is an example function showing the interface to use the
    HB_result.get_learning_curves method.
    Parameters:
        -----------
        runs: list of HB_result.run objects
        the performed runs for an unspecified config
        Returns:
            --------
            list of learning curves: list of lists of tuples
            An individual learning curve is a list of (t, x_t) tuples.
            This function must return a list of these. One could think
            of cases where one could extract multiple learning curves
            from these runs, e.g. if each run is an independent training
            run of a neural network on the data.
    """
    sr = filter(lambda r: not r.loss is None, sorted(runs, key=lambda r: r.budget))
    value = [[(r.budget, r.info['test_error']) for r in sr],]
    return(value)


bohb_logs_dir = '{}/search_space_{}/darts/run{}-seed{}'.format(
    args.working_directory, args.space, args.run_id, args.seed
)
res = hpres.logged_results_to_HB_result(bohb_logs_dir)

lcs_temp = res.get_learning_curves(lc_extractor = extract_HB_learning_curves)
lcs = dict(lcs_temp)
for key, value in lcs_temp.items():
    if value == [[]]:
        del lcs[key]

tool_tips = hpvis.default_tool_tips(res, lcs)
#embed()

inc_id = res.get_incumbent_id()

id2conf = res.get_id2config_mapping()

inc_trajectory = res.get_incumbent_trajectory()
print(inc_trajectory)
print(res.get_runs_by_id(inc_id))

all_runs = list(filter(lambda r: not (r.info is None or r.loss is None),
                       res.get_all_runs()))


budgets = res.HB_config['budgets']

run_times = np.array([(r.budget,
                       r.time_stamps['finished']-r.time_stamps['started']) for
                      r in all_runs])


def compare_val_and_test_error():
    errors = np.array([(r.info['test_error'], r.info['val_error']) for r in
                       all_runs])
    plt.scatter(errors[:,0], errors[:,1])
    plt.plot([0,100], [0,100])
    plt.show()

#plt.plot(inc_trajectory['times_finished'], inc_trajectory['losses'], label=run)
#plt.legend()
#plt.show()

#hpvis.interactive_HB_plot(lcs, tool_tip_strings=tool_tips)

runs_by_budget = {}

for b in budgets:
    runs_by_budget[b] = list(filter(lambda r: r.budget == b, all_runs))


def fanova_analysis():
    config_space = CS.ConfigurationSpace()
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

    for b in reversed(budgets):
        X, y, new_cs = res.get_fANOVA_data(config_space, budgets=[b])
        f = fanova.fANOVA(X, y, new_cs)

        dir = './fANOVA/search_space_%d/run%d-seed%d/plots_%i'%(
            args.space, args.run_id, args.seed, b
        )
        os.makedirs(dir, exist_ok=True)

        vis = fanova.visualizer.Visualizer(f, new_cs, dir, y_label='Validation Error')

        print(b)

        best_run_idx = np.argsort([r.loss for r in runs_by_budget[b]])[0]
        best_run = runs_by_budget[b][best_run_idx]

        inc_conf = id2conf[best_run.config_id]['config']
        inc_conf['budget'] = best_run.budget
        inc_line_style = {'linewidth': 3, 'color': 'lightgray', 'linestyle': 'dashed'}

        for i, hp in enumerate(config_space.get_hyperparameters()):
            print(f.quantify_importance([hp.name]))
            fig = vis.plot_marginal(i, show=False) # hp.name instead of i
            fig.axvline(x=inc_conf[hp.name], **inc_line_style)
            #fig.yscale('log')
            fig.xscale('log')
            fig.title('importance %3.1f%%'%(
                f.quantify_importance([hp.name])[(hp.name,)]['individual importance']*100)
            )
            fig.tight_layout()
            fig.savefig(dir+'/%s.png'%hp.name)
            fig.close()

        for hp1, hp2 in itertools.combinations(config_space.get_hyperparameters(), 2):
            n1, n2 = hp1.name,hp2.name
            fig = vis.plot_pairwise_marginal([n1,n2], show=False, three_d=False)
            #fig.axvline(x=inc_conf[n1], **inc_line_style)
            #fig.axhline(y=inc_conf[n2], **inc_line_style
            xlims = fig.xlim()
            ylims = fig.ylim()

            fig.scatter([inc_conf[n1]], [inc_conf[n2]], color='lightgray',
                        s=800, marker='x', linewidth=5)
            fig.xlim(xlims)
            fig.ylim(ylims)

            importance = f.quantify_importance([n1,n2])[(n1,n2)]['total importance']
            fig.title('Space {}, Budget: {} epochs'.format(args.space, b))
            #fig.title("importance %3.1f%%"%(importance*100))
            fig.tight_layout()
            fig.savefig(dir+'/%s_%s.png'%(n1,n2))
            fig.close()

        print(f.get_most_important_pairwise_marginals())
        #vis.create_all_plots(three_d=False)


def trajectory_plot():
    print(inc_trajectory.keys())
    plt.figure(figsize=(8,4.5))

    #plt.step(inc_trajectory['times_finished'], inc_trajectory['losses'],
    #         where='post', color='black', linewidth=3)

    for b,c in zip(budgets, ['gray','blue', 'orange']):
        run_points = np.array([
            (r.time_stamps['finished']+(r.info['runtime'][0]*60),
                                 r.info['test_error'][0] ) for r in
                               runs_by_budget[b]])
        plt.scatter(
            run_points[:,0], run_points[:,1], color=c,
            label='budget = %i epochs'%b, s=75
        )

    #plt.scatter(inc_trajectory['times_finished'][:-1], inc_trajectory['losses'][:-1], c='red')
    plt.xscale('log')
    #plt.xlim([4e4, 3e5]) #s3
    #plt.xlim([5e4, 4e5]) #s1
    #plt.xlim([7e4, 3e5]) #s2
    plt.yscale('log')
    plt.ylim([0.05, 0.3]) #s3
    #plt.ylim([5e-2, 2e-1]) #s1
    #plt.ylim([5e-2, 9e-2]) #s2

    plt.ylabel('test error')
    plt.xlabel('wallclock time [s]')
    plt.legend(fontsize=8)
    plt.grid(True, which="both",ls="-")
    plt.tight_layout()

    os.makedirs('./incumbents', exist_ok=True)
    fig_name = './incumbents'+'/s%d-run%d-seed%d.png'%(
        args.space, args.run_id, args.seed
    )
    plt.savefig(fig_name)
    plt.show()
    #embed()

fanova_analysis()
#trajectory_plot()
#embed()
