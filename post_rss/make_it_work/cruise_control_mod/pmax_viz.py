import os
import sys
import numpy as np 
import matplotlib.pyplot as plt

td = int(sys.argv[1])

num_rel_dist_states = 21
num_rel_vel_states = 21
num_actions = 4
num_ustates = num_actions ** td

def convert_state_to_int(state):
	increments = [num_actions**k for k in range(td)]
	increments.reverse()
	increments = [num_ustates,] + increments
	return np.sum(np.multiply(list(state), increments))



state_values_path = os.path.join('constant_generated/%d_td/' % td, 'max_safety_state_values.npy')

state_values = np.load(state_values_path, allow_pickle=True)

vis_array = np.zeros((num_rel_dist_states, num_rel_vel_states))

for rel_dist_idx in range(num_rel_dist_states):
    for rel_vel_idx in range(num_rel_vel_states):
        physical_state = rel_dist_idx * num_rel_vel_states + rel_vel_idx 

        # stay control vector
        stay_control_vector = tuple([2 for t in range(td)])
        state = (physical_state,) + stay_control_vector
        # print(state, convert_state_to_int(state))
        state_id = convert_state_to_int(state)
    
        max_safety_prob = state_values[state_id]
        vis_array[rel_dist_idx, rel_vel_idx] = max_safety_prob

# stay_control_vector
plt.imshow(vis_array, cmap=plt.cm.Blues, extent=[-5,5,25,5])
plt.title('Pmax values for the stay control vector', size=12)
plt.ylabel('Relative Distance (m)', size=12)
plt.xlabel('Relative velocity (m/s)', size=12)
plt.savefig('constant_generated/stay_control_vector_%d_td.png' % td)
plt.clf()
plt.cla()
plt.close()

print(vis_array)