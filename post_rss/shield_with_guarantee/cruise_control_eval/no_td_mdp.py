import sys
import os
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
for i in range(int((max_rel_dist - min_rel_dist) / del_rel_dist)):
	rel_dist_tuples.append((min_rel_dist + i * del_rel_dist, min_rel_dist + (i + 1) * del_rel_dist))

neg_large_val = -1000
pos_large_val = 1000

rel_dist_tuples = [(neg_large_val, min_rel_dist)] + rel_dist_tuples + [(max_rel_dist, pos_large_val)]

min_rel_dist_list = [tup[0] for tup in rel_dist_tuples]
max_rel_dist_list = [tup[1] for tup in rel_dist_tuples]

#print(rel_dist_tuples)
#print(min_rel_dist_list) 
#print(max_rel_dist_list)

"""
### relative velocity abstraction
"""
min_rel_vel = -10
max_rel_vel = 10
del_vel = 1.0


rel_vel_tuples = []
for i in range(int((max_rel_vel - min_rel_vel) / del_vel)):
	rel_vel_tuples.append((min_rel_vel + i * del_vel, min_rel_vel + (i + 1) * del_vel))

neg_large_val = -100
pos_large_val = 100

rel_vel_tuples = [(neg_large_val, min_rel_vel)] + rel_vel_tuples + [(max_rel_vel, pos_large_val)]

min_rel_vel_list = [tup[0] for tup in rel_vel_tuples]
max_rel_vel_list = [tup[1] for tup in rel_vel_tuples]

#print(rel_vel_tuples)
#print(min_rel_vel_list)
#print(max_rel_vel_list)

"""
### actions abstraction
""" 
env_min_fv_acc = -0.25
env_max_fv_acc = 0.25

min_ego_acc = -1
max_ego_acc = 1 
del_acc = 0.5

ego_acc_values = [-1.0, -0.5, 0.0, 0.5, 1.0]

"""
### possible time delay states
"""
td = 0

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

	for state_rel_vel in rel_vel_tuples:
		state_min_rel_vel = state_rel_vel[0]
		state_max_rel_vel = state_rel_vel[1]

		rel_vel_idx = rel_vel_tuples.index(state_rel_vel)
		state = rel_dist_idx*len(rel_vel_tuples) + rel_vel_idx

		for ego_acc in ego_acc_values:
			action = ego_acc_values.index(ego_acc)
			state_action_pair = (state, action)

			max_ego_acc = ego_acc
			min_rel_acc = env_min_fv_acc - max_ego_acc 
			min_rel_dist_traveled = state_min_rel_vel * del_t + 0.5 * min_rel_acc * del_t ** 2

			min_ego_acc = ego_acc
			max_rel_acc = env_max_fv_acc - min_ego_acc
			max_rel_dist_traveled = state_max_rel_vel * del_t + 0.5 * max_rel_acc * del_t ** 2
 
			# calculating the minimum and maximum values for the next state relative distance
			next_state_min_rel_dist = state_min_rel_dist + min_rel_dist_traveled
			next_state_max_rel_dist = state_max_rel_dist + max_rel_dist_traveled
			#print(next_state_min_rel_dist, next_state_max_rel_dist)

			# calculating the minimum and maximum values for the next state ego velocity
			next_state_min_rel_vel = state_min_rel_vel + min_rel_acc * del_t 
			next_state_max_rel_vel = state_max_rel_vel + max_rel_acc * del_t
			#print(next_state_min_rel_vel, next_state_max_rel_vel)

			###############################################################################################
			###################### Function to get indices ################################################
			###############################################################################################

			def get_indices(min_val, max_val, tuples):
				if min_val < tuples[0][0]:
					min_idx = 0
				elif min_val > tuples[-1][-1]:
					min_idx = len(tuples)-1
				else:
					for idx in range(len(tuples)):
						if tuples[idx][0] <= min_val <= tuples[idx][1]:
							min_idx = idx 
							break 
				
				if max_val < tuples[0][0]:
					max_idx = 0
				elif max_val > tuples[-1][-1]:
					max_idx = len(tuples)-1
				else:
					for idx in range(len(tuples)):
						if tuples[idx][0] <= max_val <= tuples[idx][1]:
							max_idx = idx 
							break

				indices = np.arange(int(min_idx), int(max_idx+1), dtype=np.int32)
				return indices

			next_states_rel_dist = get_indices(next_state_min_rel_dist, next_state_max_rel_dist, rel_dist_tuples) 
			next_states_rel_vel = get_indices(next_state_min_rel_vel, next_state_max_rel_vel, rel_vel_tuples)
			#print(next_states_rel_dist, next_states_rel_vel)

			transitions = []
			for next_state_rel_dist in next_states_rel_dist:
				for next_state_rel_vel in next_states_rel_vel:
					next_state = next_state_rel_dist * len(rel_vel_tuples) + next_state_rel_vel
					transitions.append(next_state)

			mdp[state_action_pair] = transitions 

print(mdp)

os.makedirs('constant_generated', exist_ok=True)
np.save('constant_generated/mdp_%d_td' % td, mdp)