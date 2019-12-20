import os
import time
import math
import copy
import logging
import numpy as np
import ConfigSpace as CS

from hpbandster.core.master import Master
from hpbandster.optimizers.iterations import SuccessiveHalving
#from hpbandster.optimizers.config_generators.bohb import BOHB as CG_BOHB
from custom_bohb.bohb_gen import CG_BOHB_CUSTOM as CG_BOHB

class BOHB(Master):
	def __init__(self, configspace = None,
					eta=3, min_budget=0.01, max_budget=1,
					min_points_in_model = None,	top_n_percent=15,
					num_samples = 64, random_fraction=1/3, bandwidth_factor=3,
					min_bandwidth=1e-3, start_from_default=False,
					**kwargs ):


		if configspace is None:
			raise ValueError("You have to provide a valid CofigSpace object")

		cg = CG_BOHB( configspace = configspace,
					min_points_in_model = min_points_in_model,
					top_n_percent=top_n_percent,
					num_samples = num_samples,
					random_fraction=random_fraction,
					bandwidth_factor=bandwidth_factor,
					min_bandwidth = min_bandwidth,
					start_from_default=start_from_default
					)

		super().__init__(config_generator=cg, **kwargs)

		# Hyperband related stuff
		self.eta = eta
		self.min_budget = min_budget
		self.max_budget = max_budget

		# precompute some HB stuff
		self.max_SH_iter = -int(np.log(min_budget/max_budget)/np.log(eta)) + 1
		self.budgets = max_budget * np.power(eta, -np.linspace(self.max_SH_iter-1, 0, self.max_SH_iter))

		self.config.update({
						'eta'        : eta,
						'min_budget' : min_budget,
						'max_budget' : max_budget,
						'budgets'    : self.budgets,
						'max_SH_iter': self.max_SH_iter,
						'min_points_in_model' : min_points_in_model,
						'top_n_percent' : top_n_percent,
						'num_samples' : num_samples,
						'random_fraction' : random_fraction,
						'bandwidth_factor' : bandwidth_factor,
						'min_bandwidth': min_bandwidth
					})

	def get_next_iteration(self, iteration, iteration_kwargs={}):
		# number of 'SH rungs'
		s = self.max_SH_iter - 1 - (iteration%self.max_SH_iter)
		# number of configurations in that bracket
		n0 = int(np.floor((self.max_SH_iter)/(s+1)) * self.eta**s)
		ns = [max(int(n0*(self.eta**(-i))), 1) for i in range(s+1)]

		return(SuccessiveHalving(HPB_iter=iteration, num_configs=ns, budgets=self.budgets[(-s-1):], config_sampler=self.config_generator.get_config, **iteration_kwargs))
