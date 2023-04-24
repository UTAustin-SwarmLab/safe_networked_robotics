import sys
sys.path.remove('/usr/lib/python3/dist-packages')
import os 
import itertools
import numpy as np  
from random_td_generic_mdp import TimeDelayedMDP 

class CruiseControl: 
	def __init__(self, no_td_mdp_path): 

		self.zero_td_mdp = np.load(no_td_mdp_path, allow_pickle=True).item()

		"""
		### actions abstraction
		"""
		self.env_min_ego_vel = -0.5
		self.env_max_ego_vel = 1.0
		self.del_ego_vel = 0.25 

		self.ego_vel_tuples = []
		for i in range(int((self.env_max_ego_vel - self.env_min_ego_vel) / self.del_ego_vel)):
			self.ego_vel_tuples.append((self.env_min_ego_vel + i * self.del_ego_vel, self.env_min_ego_vel + (i + 1) * self.del_ego_vel))
		#print(self.ego_vel_tuples)
		num_actions = len(self.ego_vel_tuples)
		self.abstract_actions =[-1,] + list(range(num_actions))
		print(self.abstract_actions)		
		"""
		### states abstraction
		""" 
		
		self.state_action_pairs = list(self.zero_td_mdp.keys())
		#print(self.state_action_pairs)
		self.abstract_states = [self.state_action_pairs[i][0] for i in range(0, len(self.state_action_pairs), num_actions)]
		print(self.abstract_states)

os.makedirs('random_generated', exist_ok=True)
no_td_mdp_path = "constant_generated/mdp_0_td.npy"

td_dist_path = "random_generated/td_dist.npy"
td_dist = np.load(td_dist_path)
max_delay = td_dist.shape[0]-1

env = CruiseControl(no_td_mdp_path)
mdp = TimeDelayedMDP(env, max_delay, td_dist)