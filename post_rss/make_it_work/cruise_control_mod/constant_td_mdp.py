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
del_rel_dist = 1.0  

rel_dist_list = []
for i in range(int((max_rel_dist - min_rel_dist) / del_rel_dist + 1)):
	rel_dist_list.append(min_rel_dist + i * del_rel_dist)
# print(rel_dist_list)

"""
### relative velocity abstraction
"""
min_rel_vel = -5
max_rel_vel = 5
del_vel = 0.5

rel_vel_list = []
for i in range(int((max_rel_vel - min_rel_vel) / del_vel)+1):
	rel_vel_list.append(min_rel_vel + i * del_vel)
# print(len(rel_vel_list))

"""
### actions abstraction
"""  
env_min_fv_acc = -0.2
env_max_fv_acc = 0.2
fv_acc_list = [env_min_fv_acc, env_max_fv_acc]
# print(fv_acc_list)

ego_acc_values = [-0.5, -0.25, 0.0, 0.25]
num_actions = len(ego_acc_values)
init_act = 0.0
# print(ego_acc_values)



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

# print(num_physical_states, num_ustates, num_states, num_actions)

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
		# print(rel_dist_idx, rel_vel_idx, physical_state, '-----')

		for ustate in ustates:
			# print(ustate) 
			state = physical_state + ustate  
			# print(state)
			state_id = convert_state_to_int(state)
			print("state id : ", state_id)
			# print(state_rel_dist, state_rel_vel, rel_dist_idx, rel_vel_idx, physical_state, state, state_id)
			
			if state_rel_dist <= 5.0:
				# print("unsafe state")
				mdp_unsafe_states.append(1.0)
			else:
				mdp_unsafe_states.append(0.0) 

			if state_rel_dist >= 25.0 and state_rel_vel >= 0 and list(state[1:]) == list((ego_acc_values.index(init_act) for uidx in range(td))):
				# print(ustate, state_rel_dist, state_rel_vel)
				# print("initial states")
				mdp_initial_states.append(1.0)
			else:
				mdp_initial_states.append(0.0)

			for acc_val in ego_acc_values:
				action_id = ego_acc_values.index(acc_val)
				state_action_pair = (state_id, action_id)
				# print(state_action_pair, state, state_rel_dist, state_rel_vel)
				ego_acc = ego_acc_values[state[-td]]
				# print(ego_acc)
				# print('-------------')
				# print(state, state_id, state_rel_dist, state_rel_vel)

				next_rel_dist_list = []
				next_rel_vel_list = []
				for fv_acc in fv_acc_list:
					rel_acc = fv_acc - ego_acc
					rel_dist_change = state_rel_vel * del_t + 0.5 * rel_acc * del_t ** 2
					rel_vel_change = rel_acc * del_t 
					# print(fv_acc, ego_acc, rel_acc, rel_dist_change, rel_vel_change)
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

				next_ustate = ustate[1:] + (action_id,)
				# print(next_ustate)

				transitions = []
				for next_rel_dist_idx, next_rel_vel_idx in zip(next_rel_dist_idxs, next_rel_vel_idxs):
					# print(next_rel_dist_idx, next_rel_vel_idx)
					next_physical_state = (next_rel_dist_idx * len(rel_vel_list) + next_rel_vel_idx,)
					# print(next_physical_state)
					next_state = next_physical_state + next_ustate
					# print(next_state)
					next_state_id = convert_state_to_int(next_state)
					transitions.append(next_state_id)
				# print(action, ustate, next_ustate)
				transitions = list(transitions)
				# print(transitions)
				mdp[state_action_pair] = transitions 

# print(mdp)
os.makedirs('constant_generated', exist_ok=True)
np.save('constant_generated/mdp_%d_td' % td, mdp)
np.save('constant_generated/unsafe_%d_td' % td, mdp_unsafe_states)
np.save('constant_generated/initial_%d_td' % td, mdp_initial_states)
