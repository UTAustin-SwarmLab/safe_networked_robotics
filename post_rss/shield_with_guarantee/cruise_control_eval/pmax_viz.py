import sys
sys.path.remove('/usr/lib/python3/dist-packages')
import itertools
import numpy as np 
import matplotlib.pyplot as plt

num_rel_dist_states = 27
num_rel_vel_states = 22

td = int(sys.argv[1])
per_td_vis_arrays = {}

state_values = np.load('constant_generated/state_values_%d_td.npy' % td, allow_pickle=True).item()
states = list(state_values.keys())

vis_array = np.zeros((num_rel_dist_states, num_rel_vel_states))

for rel_dist_val in range(num_rel_dist_states):
    for rel_vel_val in range(num_rel_vel_states):
        physical_state = rel_dist_val * num_rel_vel_states + rel_vel_val 

        # stay control vector
        stay_control_vector = tuple([2 for t in range(td)])
        state = (physical_state,) + stay_control_vector
        max_safety_prob = 1-state_values[state]
        vis_array[rel_dist_val, rel_vel_val] = max_safety_prob


# stay_control_vector
plt.imshow(vis_array, cmap=plt.cm.Blues, extent=[-10,10,25,0])
plt.title('Pmax values for the stay control vector', size=12)
plt.ylabel('Relative Distance (m)', size=12)
plt.xlabel('Relative velocity (m/s)', size=12)
plt.legend()
plt.savefig('constant_generated/stay_control_vector_%d_td.png' % td)
plt.clf()
plt.cla()
plt.close()

print(vis_array)