import sys
sys.path.remove('/usr/lib/python3/dist-packages')
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
		self.abstract_actions =[-1,] + list(range(num_actions))
		
		"""
		### states abstraction
		""" 
		
		self.state_action_pairs = list(self.zero_td_mdp.keys())
		self.abstract_states = [self.state_action_pairs[i][0] for i in range(0, len(self.state_action_pairs), num_actions)]
	
os.makedirs('random_generated', exist_ok=True)
max_time_delay = 3
no_td_mdp_path = "constant_generated/mdp_0_td.npy"

td_dist = np.array([[0.9, 0.1, 0.0, 0.0], [0.8, 0.1, 0.1, 0.0], [0.7, 0.1, 0.1, 0.1], [0.7, 0.1, 0.1, 0.1]]) #aggresive
#conservative_td_dist = np.array([[0.5, 0.5, 0.0, 0.0], [0.0, 0.5, 0.5, 0.0], [0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 0.5, 0.5]]) #conservative
env = CruiseControl(no_td_mdp_path)
mdp = TimeDelayedMDP(env, max_time_delay, td_dist)