import os 
import itertools
import numpy as np 
from scipy.stats import truncnorm
from random_td_generic_mdp import TimeDelayedMDP 

class CruiseControl: 
	def __init__(self, no_td_mdp_path):

		self.zero_td_mdp = np.load(no_td_mdp_path, allow_pickle=True).item()

		"""
		### actions abstraction
		"""
		self.ego_acc_list = [-1, -0.5, 0, 0.5, 1]
		num_actions = len(self.ego_acc_list)
		self.abstract_actions = [i for i in range(1, num_actions+1)] + [num_actions,]
		
		"""
		### states abstraction
		"""
		
		self.state_action_pairs = list(self.zero_td_mdp.keys())
		self.abstract_states = [self.state_action_pairs[i][0] for i in range(0, len(self.state_action_pairs), num_actions)]
		
		print(self.abstract_states)
		print(self.abstract_actions)


	

max_time_delay = 4
no_td_mdp_path = "constant_generated/mdp_0_td.npy"
td_dist = {0:0.6, 1:0.2, 2:0.1, 3:0.08, 4:0.02}
env = CruiseControl(no_td_mdp_path)
#mdp = TimeDelayedMDP(env, max_time_delay, td_dist)