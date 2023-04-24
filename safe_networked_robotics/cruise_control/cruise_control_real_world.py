import os 
import itertools
import numpy as np 
from random_td_generic_mdp import TimeDelayedMDP 

class CruiseControlDemo:
	def __init__(self):
		self.del_t = 0.05 

		"""
		### relative distance abstraction 
		""" 
		min_rel_dist = 0
		max_rel_dist = 3
		del_rel_dist = 0.2 
		ultimate_rel_dist = 10

		self.rel_dist_tuples = []
		self.min_rel_dist_list = []
		self.max_rel_dist_list = []
		for i in range(int((max_rel_dist - min_rel_dist) / del_rel_dist)):
			self.rel_dist_tuples.append((min_rel_dist + i * del_rel_dist, min_rel_dist + (i + 1) * del_rel_dist))
			self.min_rel_dist_list.append(min_rel_dist + i * del_rel_dist)
			self.max_rel_dist_list.append(min_rel_dist + (i + 1) * del_rel_dist)

		self.rel_dist_tuples.append((max_rel_dist, ultimate_rel_dist))
		self.min_rel_dist_list.append(max_rel_dist)
		self.max_rel_dist_list.append(ultimate_rel_dist)

		"""
		### front vehicle velocity limits 
		"""
		self.env_min_fv_vel = -0.5
		self.env_max_fv_vel = 0.5

		"""
		### actions abstraction
		"""
		env_min_ego_vel = -1 
		env_max_ego_vel = 1
		del_ego_vel = 0.25 

		self.ego_vel_tuples = []
		for i in range(int((env_max_ego_vel - env_min_ego_vel) / del_ego_vel)):
			self.ego_vel_tuples.append((env_min_ego_vel + i * del_ego_vel, env_min_ego_vel + (i + 1) * del_ego_vel))

		"""
		### creating abstract states and abstract actions
		"""
		self.num_state_features = 1
		self.abstract_states = [tuple([i]) for i in range(len(self.rel_dist_tuples))]
		self.abstract_actions = [i for i in range(len(self.ego_vel_tuples))]

		print(self.rel_dist_tuples)
		print(self.abstract_states)

	def forward(self, abstract_state, abstract_action):
		#print(abstract_state, abstract_action)
		rel_dist_idx = abstract_state[0]
		state = self.rel_dist_tuples[rel_dist_idx]
		action = self.ego_vel_tuples[abstract_action]

		state_min_rel_dist = state[0]
		state_max_rel_dist = state[1]

		min_ego_vel = action[0]
		max_ego_vel = action[1]		

		min_rel_vel = self.env_min_fv_vel - max_ego_vel 
		min_rel_dist_traveled = min_rel_vel * self.del_t
	
		max_rel_vel = self.env_max_fv_vel - min_ego_vel
		max_rel_dist_traveled = max_rel_vel * self.del_t 

		next_state_min_rel_dist = state_min_rel_dist + min_rel_dist_traveled
		next_state_max_rel_dist = state_max_rel_dist + max_rel_dist_traveled

		#print(next_state_min_rel_dist, next_state_max_rel_dist)

		###############################################################################################
		###################### Function to get indices ################################################
		###############################################################################################
		def get_indices(min_val, max_val, min_list, max_list):
			# calculate the difference list
			min_diff_list = [min_val - min_list[idx] for idx in range(len(min_list))]
			min_idx = -1 
			# if the min val is lower than the minimum most value possible, min_idx = 0
			if min_val < min_list[0]:
				min_idx = 0
			# if the min val is higher than the maximum most value possible, min_idx = len(min_list)
			elif min_val > min_list[-1]:
				min_idx = len(min_list) - 1
			else:
				for idx in range(len(min_list)):
					if min_diff_list[idx] < 0:
						min_idx = idx - 1
						break 
					elif min_diff_list[idx] == 0:
						min_idx = idx
						break 
		
			max_diff_list = [max_val - max_list[idx] for idx in range(len(max_list))]
			max_idx = -1 
			if max_val < max_list[0]:
				max_idx = 0
			elif max_val > max_list[-1]:
				max_idx = len(max_list) - 1
			else:
				for idx in range(len(max_list)):
					if max_diff_list[idx] <= 0:
						max_idx = idx 
						break 
			indices = list(range(int(min_idx), int(max_idx+1)))
		
			return indices

		next_states_rel_dist = get_indices(next_state_min_rel_dist, next_state_max_rel_dist, self.min_rel_dist_list, self.max_rel_dist_list) 

		next_states = list(itertools.product(next_states_rel_dist))	

		return next_states
		
		#return [1]

max_time_delay = 4
env = CruiseControlDemo()
#mdp = TimeDelayedMDP(env, max_time_delay)
