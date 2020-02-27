import os
import pickle
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython import embed


colors={
        'BOHB-PC-DARTS': 'darkorange',
        'BOHB-DARTS': 'dodgerblue',
        'BOHB-GDAS'   : 'forestgreen',
        'RE': 'crimson',
		'RS': 'darkorchid',
		'RL': 'sienna',
		'TPE': 'deepskyblue',
        'SMAC': 'violet',
        'HB': 'darkgray',
        'BOHB': 'gold'
}

markers={
        'BOHB-DARTS': '^',
        'BOHB-PC-DARTS': 'v',
        'BOHB-GDAS'   : 'x',
        'RS': 'D',
		'RE': 'o',
		'RL': 's',
		'SMAC': 'h',
        'HB': '>',
        'BOHB': '*',
        'TPE': '<'
}


def get_incumbent(losses, time_stamps):
    return_dict = {'time_stamps': [],
                   'losses': [],
    }

    current_incumbent = float('inf')
    incumbent_budget = -float('inf')

    for l, t in zip(losses, time_stamps):
        if l < current_incumbent:
            current_incumbent = l
            return_dict['losses'].append(l)
            return_dict['time_stamps'].append(t)
        else:
            return_dict['losses'].append(return_dict['losses'][-1])
            return_dict['time_stamps'].append(t)
    return return_dict.values()


def get_trajectories(args, global_min, path='regularized_evolution',
                     methods=['RE', 'RS']):
    all_trajectories = {}
    for m in methods:
        dfs = []
        for seed in range(500):
            filename = os.path.join(path, m,
                                    'algo_{}_0_ssp_{}_seed_{}.obj'.format(m, args.space,
                                                                          seed))
            try:
                with open(filename, 'rb') as f:
                    data = pickle.load(f)
                    losses = [1 - x.test_accuracy - global_min for x in data]
                    times = np.array([x.training_time for x in data])
                    times = [np.sum(times[:i+1]) for i in range(len(times))]
                    if m in ['HB', 'BOHB']:
                        costs = np.array([x.budget for x in data])
                        costs = np.array(
                            [np.sum(costs[:i+1]) for i in range(len(costs))]
                        )
                        n = len(np.where(costs <= 280*108)[0])
                        times, losses = get_incumbent(losses[:n], times[:n])
                    else:
                        times, losses = get_incumbent(losses, times)
                    print(seed, ' MIN: ', min(losses))
                    df = pd.DataFrame({str(seed): losses}, index=times)
                #embed()
                dfs.append(df)
            except FileNotFoundError:
                break
        df = merge_and_fill_trajectories(dfs, default_value=None)
        if df.empty:
            continue
        print(m, df.shape)

        all_trajectories[m] = {
            'time_stamps': np.array(df.index),
            'losses': np.array(df.T)
        }

    return all_trajectories


def merge_and_fill_trajectories(pandas_data_frames, default_value=None):
	# merge all tracjectories keeping all time steps
	df = pd.DataFrame().join(pandas_data_frames, how='outer')

	# forward fill to make it a propper step function
	df=df.fillna(method='ffill')

	if default_value is None:
	# backward fill to replace the NaNs for the early times by
	# the performance of a random configuration
		df=df.fillna(method='bfill')
	else:
		df=df.fillna(default_value)

	return(df)


def plot_losses(fig, ax, axins, incumbent_trajectories, regret=True,
                incumbent=None, show=True, linewidth=3, marker_size=10,
                xscale='log', xlabel='wall clock time [s]', yscale='log',
                ylabel=None, legend_loc = 'best', xlim=None, ylim=None,
                plot_mean=True, labels={}, markers=markers, colors=colors,
                figsize=(16,9)):

    if regret:
        if ylabel is None: ylabel = 'regret'
		# find lowest performance in the data to update incumbent

        if incumbent is None:
            incumbent = np.inf
            for tr in incumbent_trajectories.values():
                incumbent = min(tr['losses'][:,-1].min(), incumbent)
            print('incumbent value: ', incumbent)

    for m,tr in incumbent_trajectories.items():
        trajectory = np.copy(tr['losses'])
        if (trajectory.shape[0] == 0): continue
        if regret: trajectory -= incumbent

        sem  =  np.sqrt(trajectory.var(axis=0, ddof=1)/tr['losses'].shape[0])
        if plot_mean:
            mean =  trajectory.mean(axis=0)
        else:
            mean = np.median(trajectory,axis=0)
            sem *= 1.253

        if 'DARTS' in m or 'GDAS' in m:
            ax.fill_between(tr['time_stamps'], mean-2*sem, mean+2*sem,
                            color=colors[m], alpha=0.2)

        ax.plot(tr['time_stamps'],mean,
                label=labels.get(m, m), color=colors.get(m, None),linewidth=linewidth,
                marker=markers.get(m,None), markersize=marker_size, markevery=(0.1,0.1))

        if axins is not None:
            axins.plot(tr['time_stamps'],mean,
                       label=labels.get(m, m), color=colors.get(m, None),linewidth=linewidth,
                       marker=markers.get(m,None), markersize=marker_size, markevery=(0.1,0.1))

    return (fig, ax)
