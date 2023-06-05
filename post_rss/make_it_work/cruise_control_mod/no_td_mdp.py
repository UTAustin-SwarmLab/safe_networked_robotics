import sys
import os
import math
import torch
import itertools
import numpy as np 

del_t = 1.0

"""
### relative distance abstraction
"""
min_rel_dist = 5 
max_rel_dist = 25 
del_rel_dist = 1.0  

rel_dist_list = []
for i in range(int((max_rel_dist - min_rel_dist) / del_rel_dist + 1)):
	rel_dist_list.append(min_rel_dist + i * del_rel_dist)

"""
### relative velocity abstraction
"""
min_rel_vel = -5
max_rel_vel = 5
del_vel = 0.5 

rel_vel_list = [] 
for i in range(int((max_rel_vel - min_rel_vel) / del_vel)+1):
	rel_vel_list.append(min_rel_vel + i * del_vel)

num_states = len(rel_dist_list)*len(rel_vel_list)

"""
### actions abstraction
""" 
env_min_fv_acc = -0.2
env_max_fv_acc = 0.2
fv_acc_list = [env_min_fv_acc, env_max_fv_acc]

ego_acc_values = [-0.5, -0.25, 0.0, 0.25]
num_actions = len(ego_acc_values)

"""
### possible time delay states
"""
td = 0

"""
### for storage
"""
mdp = {}

""" 
### iteration over possible states
"""
mdp_unsafe_states = []
mdp_initial_states = []

for state_rel_dist in rel_dist_list:
	rel_dist_idx = rel_dist_list.index(state_rel_dist)
	for state_rel_vel in rel_vel_list:
		rel_vel_idx = rel_vel_list.index(state_rel_vel)
		state_id = rel_dist_idx*len(rel_vel_list)+rel_vel_idx
		print("state id : ", state_id)
		# print(state_rel_dist, state_rel_vel)

		if state_rel_dist <= 5.0:
			# print("unsafe state")
			mdp_unsafe_states.append(1.0)
		else:
			# print("not unsafe state yet")
			mdp_unsafe_states.append(0.0) 

		if state_rel_dist >= 25.0 and state_rel_vel >= 0.0:
			# print("initial state")
			mdp_initial_states.append(1.0)
		else:
			# print("not an initial state")
			mdp_initial_states.append(0.0)

		for ego_acc in ego_acc_values:
			action = ego_acc_values.index(ego_acc) 
			state_action_pair = (state_id, action)
			# print(state_action_pair, ego_acc)

			next_rel_dist_list = []
			next_rel_vel_list = []
			for fv_acc in fv_acc_list:
				# print("+++++++++++++++++++++++++++")
				# print(fv_acc, ego_acc)
				rel_acc = fv_acc - ego_acc
				# print(rel_acc) 
				rel_dist_change = state_rel_vel * del_t + 0.5 * rel_acc * del_t ** 2
				rel_vel_change = rel_acc * del_t 
				next_rel_dist_list.append(state_rel_dist+rel_dist_change)
				next_rel_vel_list.append(state_rel_vel+rel_vel_change)

			# print(next_rel_dist_list, next_rel_vel_list)

			next_rel_dist_idxs = np.digitize(next_rel_dist_list, rel_dist_list)
			for i in range(next_rel_dist_idxs.shape[0]):
				if next_rel_dist_idxs[i] == 0:
					continue
				else:
					next_rel_dist_idxs[i] -= 1
			next_rel_vel_idxs = np.digitize(next_rel_vel_list, rel_vel_list)
			for i in range(next_rel_vel_idxs.shape[0]):
				if next_rel_vel_idxs[i] == 0:
					continue
				else:
					next_rel_vel_idxs[i] -= 1
			# print(next_rel_dist_idxs, next_rel_vel_idxs)
			transitions = []
			for next_rel_dist_idx, next_rel_vel_idx in zip(next_rel_dist_idxs, next_rel_vel_idxs):
				# print(next_rel_dist_idx, next_rel_vel_idx)
				next_state_id = next_rel_dist_idx * len(rel_vel_list) + next_rel_vel_idx
				transitions.append(next_state_id)
			transitions = list(transitions)

			mdp[state_action_pair] = transitions 

print(mdp)
os.makedirs('constant_generated', exist_ok=True)
np.save('constant_generated/mdp_%d_td' % td, mdp)
np.save('constant_generated/unsafe_%d_td' % td, mdp_unsafe_states)
np.save('constant_generated/initial_%d_td' % td, mdp_initial_states)

