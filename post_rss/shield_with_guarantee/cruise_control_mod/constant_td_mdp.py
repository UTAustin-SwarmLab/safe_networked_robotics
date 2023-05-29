import sys
import os
import math
import itertools
import numpy as np 

del_t = 1.0 

"""
### relative distance abstraction
"""
min_rel_dist = 5 
max_rel_dist = 25 
del_rel_dist = 0.5  

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

"""
### actions abstraction
""" 
env_min_fv_acc = 0
env_max_fv_acc = 0.5
fv_acc_list = [0, env_max_fv_acc]

# ego_acc_values = [-0.5, -0.25, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
ego_acc_values = [-0.5, -0.25, 0.0, 0.25, 0.5]
num_actions = len(ego_acc_values)



"""
### possible time delay states
"""
td = int(sys.argv[1])
ustates = list(itertools.product(list(range(len(ego_acc_values))), repeat=td))

"""
### for storage
"""
mdp = {}

num_physical_states = len(rel_dist_list) * len(rel_vel_list)
num_ustates = len(ustates)
num_states = num_physical_states*num_ustates
num_actions = len(ego_acc_values)

print(num_physical_states, num_ustates, num_states, num_actions)

def convert_state_to_int(state):
	increments = [num_actions**k for k in range(td)]
	increments.reverse()
	increments = [num_ustates,] + increments
	return np.sum(np.multiply(list(state), increments))


"""
### iteration over possible states
"""

mdp_unsafe_states = []
mdp_initial_states = []

for state_rel_dist in rel_dist_list:
	rel_dist_idx = rel_dist_list.index(state_rel_dist)
	for state_rel_vel in rel_vel_list:
		rel_vel_idx = rel_vel_list.index(state_rel_vel)

		physical_state = (rel_dist_idx*len(rel_vel_list) + rel_vel_idx,)

		for ustate in ustates:
			state = physical_state + ustate 
			state_id = convert_state_to_int(state)
			# print(state, state_id)
			
			if state_rel_dist <= 5.0:
				mdp_unsafe_states.append(1.0)
			else:
				mdp_unsafe_states.append(0.0) 

			if state_rel_dist >= 10.0 and state_rel_vel >= 0:
				mdp_initial_states.append(1.0)
			else:
				mdp_initial_states.append(0.0)

			for acc_val in ego_acc_values:
				action = ego_acc_values.index(acc_val)
				state_action_pair = (state_id, action)
				print('------------------')
				print(state_action_pair)
				print(state_rel_dist, state_rel_vel, ustate, action)
				ego_acc = ego_acc_values[state[-td]]
				print(ego_acc)

				next_rel_dist_list = []
				next_rel_vel_list = []
				for fv_acc in fv_acc_list:
					rel_acc = fv_acc - ego_acc
					rel_dist_change = state_rel_vel * del_t + 0.5 * rel_acc * del_t ** 2
					rel_vel_change = state_rel_vel + rel_acc * del_t 
					next_rel_dist_list.append(state_rel_dist+rel_dist_change)
					next_rel_vel_list.append(state_rel_vel+rel_vel_change)
				
				print(next_rel_dist_list, next_rel_vel_list)
				next_rel_dist_list = np.digitize(next_rel_dist_list, rel_dist_list)
				for i in range(next_rel_dist_list.shape[0]):
					if next_rel_dist_list[i] == 0:
						continue
					else:
						next_rel_dist_list[i] -= 1
				next_rel_vel_list = np.digitize(next_rel_vel_list, rel_vel_list)
				for i in range(next_rel_vel_list.shape[0]):
					if next_rel_vel_list[i] == 0:
						continue
					else:
						next_rel_vel_list[i] -= 1
				print(next_rel_dist_list, next_rel_vel_list)

				next_ustate = ustate[1:] + (action,)

				transitions = []
				for next_rel_dist, next_rel_vel in zip(next_rel_dist_list, next_rel_vel_list):
					next_physical_state = (next_rel_dist * len(rel_vel_list) + next_rel_vel,)
					print(next_physical_state)
					next_state = next_physical_state + next_ustate
					next_state_id = convert_state_to_int(next_state)
					transitions.append(next_state_id)
				transitions = list(np.unique(transitions))
				print(transitions)
				mdp[state_action_pair] = transitions 

# print(mdp)
os.makedirs('constant_generated', exist_ok=True)
np.save('constant_generated/mdp_%d_td' % td, mdp)
np.save('constant_generated/unsafe_%d_td' % td, mdp_unsafe_states)
np.save('constant_generated/initial_%d_td' % td, mdp_initial_states)
