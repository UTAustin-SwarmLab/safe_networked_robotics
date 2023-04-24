import os
import sys
import math
import itertools
import numpy as np 

del_t = 0.05

"""
### relative distance abstraction
"""
min_rel_dist = 0
max_rel_dist = 5
del_rel_dist = 0.5
ultimate_rel_dist = 10

rel_dist_tuples = []
min_rel_dist_list = []
max_rel_dist_list = []
for i in range(int((max_rel_dist - min_rel_dist) / del_rel_dist)):
	rel_dist_tuples.append((min_rel_dist + i * del_rel_dist, min_rel_dist + (i + 1) * del_rel_dist))
	min_rel_dist_list.append(min_rel_dist + i * del_rel_dist)
	max_rel_dist_list.append(min_rel_dist + (i + 1) * del_rel_dist)

rel_dist_tuples.append((max_rel_dist, ultimate_rel_dist))
min_rel_dist_list.append(max_rel_dist)
max_rel_dist_list.append(ultimate_rel_dist)

"""
### front vehicle velocity limits 
"""
env_min_fv_vel = -0.5
env_max_fv_vel = 0.5

"""
### actions abstraction
"""
env_min_ego_vel = -1
env_max_ego_vel = 1
del_ego_vel = 0.2 

ego_vel_tuples = []
for i in range(int((env_max_ego_vel - env_min_ego_vel) / del_ego_vel)):
	ego_vel_tuples.append((env_min_ego_vel + i * del_ego_vel, env_min_ego_vel + (i + 1) * del_ego_vel))

num_actions = len(ego_vel_tuples)
print(rel_dist_tuples)
print(ego_vel_tuples)

"""
### possible time delay states
"""
td = int(sys.argv[1])

uactions = list(itertools.product(ego_vel_tuples, repeat=td))
ustates = []
for uact in uactions:
	ustate = []
	for t in range(td):
		ustate.append(ego_vel_tuples.index(uact[t]))
	ustate = tuple(ustate)
	ustates.append(ustate)
#print(ustates)

"""
### for storage
"""
mdp = {}
states = []

"""
### iteration over possible states
"""

for state_rel_dist in rel_dist_tuples:

	state_min_rel_dist = state_rel_dist[0] 
	state_max_rel_dist = state_rel_dist[1]

	rel_dist_idx = rel_dist_tuples.index(state_rel_dist)
	rel_dist_str = "(s=%d)" % rel_dist_idx 

	for ustate in ustates:
		state = (rel_dist_idx,) + ustate 
		states.append(state)

		for action in ego_vel_tuples:
			act_idx = ego_vel_tuples.index(action)
			act_str = "[a" +  str(act_idx) + "]"
				
			state_action_pair = (state, act_idx)
			#print('------------------')
			#print(state_action_pair)
			#print(state_rel_dist, state_ego_vel, state_fv_vel)
			
			state_min_ego_vel = ego_vel_tuples[state[-td]][0]
			state_max_ego_vel = ego_vel_tuples[state[-td]][1]

			# modifying the min_rel_acc so that the limits of min_rel_vel is maintained
			min_rel_vel = env_min_fv_vel - state_max_ego_vel 
			min_rel_dist_traveled = min_rel_vel * del_t

			# modifying the max_rel_acc so that the limits of max_rel_vel is maintained
			max_rel_vel = env_max_fv_vel - state_min_ego_vel
			max_rel_dist_traveled = max_rel_vel * del_t 

			# calculating the minimum and maximum values for the next state relative distance
			next_state_min_rel_dist = state_min_rel_dist + min_rel_dist_traveled
			next_state_max_rel_dist = state_max_rel_dist + max_rel_dist_traveled

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
				indices = np.arange(int(min_idx), int(max_idx+1), dtype=np.int32)
				return indices 
			next_states_rel_dist = get_indices(next_state_min_rel_dist, next_state_max_rel_dist, min_rel_dist_list, max_rel_dist_list) 
			next_ustate = ustate[1:] + (act_idx,)
			transitions = []
			for next_state_rel_dist in next_states_rel_dist:
				next_state = (next_state_rel_dist,) + next_ustate
				transitions.append(next_state)
			mdp[state_action_pair] = transitions 

			#print(next_state_min_rel_dist, next_state_max_rel_dist)
			print(next_state_min_rel_dist, next_state_max_rel_dist, next_states_rel_dist, transitions)

os.makedirs('generated', exist_ok=True)
np.save('generated/mdp_%d_td' % td, mdp)
np.save('generated/states_%d_td' % td, states)
