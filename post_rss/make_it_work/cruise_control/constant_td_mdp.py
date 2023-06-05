import os
import sys
sys.path.remove('/usr/lib/python3/dist-packages')
import math
import itertools
import numpy as np 

del_t = 1.0

""" 
### relative distance abstraction
"""
min_rel_dist = 0
max_rel_dist = 25
del_rel_dist = 1.0 

rel_dist_tuples = []
min_rel_dist_list = []
max_rel_dist_list = []
for i in range(int((max_rel_dist - min_rel_dist) / del_rel_dist)):
	rel_dist_tuples.append((min_rel_dist + i * del_rel_dist, min_rel_dist + (i + 1) * del_rel_dist))
	min_rel_dist_list.append(min_rel_dist + i * del_rel_dist)
	max_rel_dist_list.append(min_rel_dist + (i + 1) * del_rel_dist)

rel_dist_tuples.append((25.0, 100.0))
min_rel_dist_list.append(25.0)
max_rel_dist_list.append(100.0)
rel_dist_pts = np.arange(min_rel_dist, max_rel_dist, del_rel_dist)
rel_dist_pts = np.append(rel_dist_pts, 25)

"""
### relative velocity abstraction
"""
min_rel_vel = -5 
max_rel_vel = 5
del_vel = 1.0


rel_vel_tuples = []
min_rel_vel_list = []
max_rel_vel_list = []
for i in range(int((max_rel_vel - min_rel_vel) / del_vel)):
	rel_vel_tuples.append((min_rel_vel + i * del_vel, min_rel_vel + (i + 1) * del_vel))
	min_rel_vel_list.append(min_rel_vel + i * del_vel)
	max_rel_vel_list.append(min_rel_vel + (i + 1) * del_vel)
rel_vel_pts = np.arange(min_rel_vel, max_rel_vel, del_vel)

"""
### actions abstraction
"""
env_min_fv_acc = -0.5 
env_max_fv_acc = 0.5 

ego_acc_list = [-1, -0.5, 0, 0.5, 1]
num_actions = len(ego_acc_list)
abstract_actions = [i for i in range(1, num_actions+1)]

"""
### possible time delay states
"""
td = int(sys.argv[1])
"""
uactions = list(itertools.product(ego_acc_list, repeat=td))
ustates = []
for uact in uactions:
	ustate = []
	for t in range(td):
		ustate.append(ego_acc_list.index(uact[t]))
	ustate = tuple(ustate)
	ustates.append(ustate)
"""
ustates = list(itertools.product(abstract_actions, repeat=td))

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

	for state_rel_vel in rel_vel_tuples:
	
		state_min_rel_vel = state_rel_vel[0]
		state_max_rel_vel = state_rel_vel[1]

		rel_vel_idx = rel_vel_tuples.index(state_rel_vel)
		rel_vel_str = "(v=%d)" % rel_vel_idx

		for ustate in ustates:
			state = (rel_dist_idx * len(rel_vel_tuples) + rel_vel_idx,) + ustate 
			states.append(state)

			for action in abstract_actions:
				act_str = "[a" +  str(action) + "]"
				
				state_action_pair = (state, action)
				#print('------------------')
				#print(state_action_pair)
				#print(state_rel_dist, state_ego_vel, state_fv_vel)

				#######################################################################################
				########### function to modify acceleration ###########################################
				#######################################################################################

				def modify_acc(vel, acc, max_vel, min_vel, del_t):
					if vel >= max_vel:
						if vel + acc * del_t >= max_vel:
							acc = 0.0 
					else:
						if vel + acc * del_t >= max_vel:
							acc = (max_vel - vel) / del_t 
					if vel <= min_vel:
						if vel + acc * del_t <= min_vel:
							acc = 0.0 
					else:
						if vel + acc * del_t <= min_vel:
							acc = (min_vel - vel) / del_t 
					return acc

				ego_acc = ego_acc_list[state[-td]-1]
				# modifying the min_rel_acc so that the limits of min_rel_vel is maintained
				min_rel_acc = env_min_fv_acc - ego_acc
				#min_rel_acc = modify_acc(state_min_rel_vel, min_rel_acc, max_rel_vel, min_rel_vel, del_t)
				min_rel_dist_traveled = state_min_rel_vel * del_t + 0.5 * min_rel_acc * del_t ** 2

				# modifying the max_rel_acc so that the limits of max_rel_vel is maintained
				max_rel_acc = env_max_fv_acc - ego_acc
				#max_rel_acc = modify_acc(state_max_rel_vel, max_rel_acc, max_rel_vel, min_rel_vel, del_t)
				max_rel_dist_traveled = state_max_rel_vel * del_t + 0.5 * max_rel_acc * del_t ** 2

				# calculating the minimum and maximum values for the next state relative distance
				next_state_min_rel_dist = state_min_rel_dist + min_rel_dist_traveled
				next_state_max_rel_dist = state_max_rel_dist + max_rel_dist_traveled
				#print(next_state_min_rel_dist, next_state_max_rel_dist)

				# calculating the minimum and maximum values for the next state ego velocity
				next_state_min_rel_vel = state_min_rel_vel + min_rel_acc * del_t 
				next_state_max_rel_vel = state_max_rel_vel + max_rel_acc * del_t
				#print(next_state_min_ego_vel, next_state_max_ego_vel)

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
				next_states_rel_vel = get_indices(next_state_min_rel_vel, next_state_max_rel_vel, min_rel_vel_list, max_rel_vel_list)
				next_ustate = ustate[1:] + (action,)

				transitions = []
				for next_state_rel_dist in next_states_rel_dist:
					for next_state_rel_vel in next_states_rel_vel:
						next_state = (next_state_rel_dist * len(rel_vel_tuples) + next_state_rel_vel,) + next_ustate
						transitions.append(next_state)

				mdp[state_action_pair] = transitions 

				#print(next_state_min_rel_dist, next_state_max_rel_dist)
				#print(next_states_rel_dist)

os.makedirs('constant_generated', exist_ok=True)
np.save('constant_generated/mdp_%d_td' % td, mdp)
np.save('constant_generated/states_%d_td' % td, states)