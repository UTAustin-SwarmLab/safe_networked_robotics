import sys
sys.path.remove('/usr/lib/python3/dist-packages')
import os
import sys
import math
import itertools
import numpy as np 

del_t = 0.1
 
"""
### relative distance abstraction
"""
min_rel_dist = 0
max_rel_dist = 5
del_rel_dist = 0.5  

rel_dist_tuples = []
for i in range(int((max_rel_dist - min_rel_dist) / del_rel_dist)):
	rel_dist_tuples.append((min_rel_dist + i * del_rel_dist, min_rel_dist + (i + 1) * del_rel_dist))

pos_large_val = 10
neg_large_val = -1
rel_dist_tuples = [(neg_large_val, min_rel_dist)] + rel_dist_tuples + [(max_rel_dist, pos_large_val)]

"""
### front vehicle velocity limits 
"""
env_min_fv_vel = 0.0
env_max_fv_vel = 0.5

"""
### actions abstraction
"""
env_min_ego_vel = -0.5
env_max_ego_vel = 1.0
del_ego_vel = 0.25 

ego_vel_tuples = []
for i in range(int((env_max_ego_vel - env_min_ego_vel) / del_ego_vel)):
	ego_vel_tuples.append((env_min_ego_vel + i * del_ego_vel, env_min_ego_vel + (i + 1) * del_ego_vel))

num_actions = len(ego_vel_tuples)
print(rel_dist_tuples)
print(ego_vel_tuples)

"""
### for storage
"""
mdp = {}

"""
### iteration over possible states
"""
for state_rel_dist in rel_dist_tuples:
	state_min_rel_dist = state_rel_dist[0] 
	state_max_rel_dist = state_rel_dist[1]
	rel_dist_idx = rel_dist_tuples.index(state_rel_dist)
	state = (rel_dist_idx,)
	print("physical state : ", state)

	for act in ego_vel_tuples:
		action = ego_vel_tuples.index(act)
		state_action_pair = (state, action)
		print(state_action_pair)
			
		min_ego_vel = act[0]
		max_ego_vel = act[1]

		min_rel_vel = env_min_fv_vel - max_ego_vel 
		min_rel_dist_traveled = min_rel_vel * del_t

		max_rel_vel = env_max_fv_vel - min_ego_vel
		max_rel_dist_traveled = max_rel_vel * del_t 

		next_state_min_rel_dist = state_min_rel_dist + min_rel_dist_traveled
		next_state_max_rel_dist = state_max_rel_dist + max_rel_dist_traveled

		#print(next_state_min_rel_dist, next_state_max_rel_dist)

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
		transitions = [(next_state_rel_dist,) for next_state_rel_dist in next_states_rel_dist]

		mdp[state_action_pair] = transitions 
		#print(transitions)
		#print(next_states_rel_dist)
				
os.makedirs('constant_generated', exist_ok=True)
np.save('constant_generated/mdp_0_td', mdp)
#np.save('constant_generated/states_%d_td' % td, states)
