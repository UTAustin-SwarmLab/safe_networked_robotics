import sys
sys.path.remove('/usr/lib/python3/dist-packages')
import os 
import itertools
import numpy as np 
from scipy.stats import truncnorm
from random_td_generic_mdp import TimeDelayedMDP 

class GridWorld: 
	def __init__(self, no_td_mdp_path): 

		self.zero_td_mdp = np.load(no_td_mdp_path, allow_pickle=True).item()

		"""
		### actions abstraction
		"""
		self.actions_list = [0, 1, 2, 3, 4] 
		num_actions = len(self.actions_list)
		self.abstract_actions =[-1,] + self.actions_list
		
		"""
		### states abstraction
		"""
		
		self.state_action_pairs = list(self.zero_td_mdp.keys())
		self.abstract_states = [self.state_action_pairs[i][0] for i in range(0, len(self.state_action_pairs), num_actions)]
		print(self.abstract_states)

os.makedirs('random_generated', exist_ok=True)
max_time_delay = 3
no_td_mdp_path = "constant_generated/mdp_no_td.npy"
td_dist = np.array([[0.9, 0.1, 0.0, 0.0], [0.8, 0.1, 0.1, 0.0], [0.7, 0.1, 0.1, 0.1], [0.7, 0.1, 0.1, 0.1]]) #aggresive
print(td_dist)
env = GridWorld(no_td_mdp_path) 
mdp = TimeDelayedMDP(env, max_time_delay, td_dist)