import sys
import os 
import itertools
import numpy as np  
from scipy.stats import truncnorm
from random_td_generic_mdp import TimeDelayedMDP 

class CruiseControl: 
	def __init__(self, no_td_mdp_path, no_td_unsafe_states_path, no_td_initial_states_path): 

		self.zero_td_mdp = np.load(no_td_mdp_path, allow_pickle=True).item()
		self.zero_td_unsafe_states = np.load(no_td_unsafe_states_path, allow_pickle=True)
		self.zero_td_initial_states = np.load(no_td_initial_states_path, allow_pickle=True)

		"""
		### actions abstraction
		"""
		self.ego_acc_list = [-0.5, -0.25, 0.0, 0.25]
		num_actions = len(self.ego_acc_list)
		self.abstract_actions = list(range(num_actions+1)) 
		print(self.abstract_actions)
				 
		"""
		### states abstraction
		""" 
		
		self.state_action_pairs = list(self.zero_td_mdp.keys())
		self.abstract_states = [self.state_action_pairs[i][0] for i in range(0, len(self.state_action_pairs), num_actions)]
	
os.makedirs('random_generated', exist_ok=True)
max_time_delay = 3
no_td_mdp_path = "constant_generated/mdp_0_td.npy"
no_td_unsafe_states_path = "constant_generated/unsafe_0_td.npy"
no_td_initial_states_path = "constant_generated/initial_0_td.npy"

td_dist = np.array([[0.9, 0.1, 0.0, 0.0], [0.8, 0.1, 0.1, 0.0], [0.7, 0.1, 0.1, 0.1], [0.7, 0.1, 0.1, 0.1]]) #aggresive
env = CruiseControl(no_td_mdp_path, no_td_unsafe_states_path, no_td_initial_states_path)
mdp = TimeDelayedMDP(env, max_time_delay, td_dist) 