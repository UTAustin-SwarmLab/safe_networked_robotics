import sys
import itertools
import numpy as np 
import matplotlib.pyplot as plt
 

num_rel_dist_states = 26
num_rel_vel_states = 10

max_td = 2
per_td_vis_arrays = {}

state_values = np.load('generated/random_td_state_values_%d_td.npy' % max_td, allow_pickle=True).item()
states = list(state_values.keys())

wc_vis_array = np.zeros((num_rel_dist_states, num_rel_vel_states))
bc_vis_array = np.zeros((num_rel_dist_states, num_rel_vel_states))
sc_vis_array = np.zeros((num_rel_dist_states, num_rel_vel_states))

for rel_dist_val in range(num_rel_dist_states):
    for rel_val in range(num_rel_vel_states):
        physical_state = (rel_dist_val, rel_val)
        abstract_state = rel_dist_val * num_rel_vel_states + rel_val 
        

        relevant_states = [s for s in states if s[0] == abstract_state]
        relevant_states = [s for s in relevant_states if s[-1] == 0]
        #relevant_states = [s for s in relevant_states if s[-2] == max_td]
        #relevant_states = [s for s in relevant_states if s[1:-2] == [0,0]]

        relevant_states_values = [state_values[s] for s in relevant_states]

        # worst_control_vector
        wc_pmax_val = min(relevant_states_values)
        wc_vis_array[rel_dist_val, rel_val] = wc_pmax_val 

        # best_control_vector
        bc_pmax_val = max(relevant_states_values) 
        bc_vis_array[rel_dist_val, rel_val] = bc_pmax_val 

        # stay control vector
        stay_control_vector = tuple([3 for t in range(max_td)])
        state = (abstract_state,) + stay_control_vector + (0,0)
        sc_pmax_val = state_values[state]
        sc_vis_array[rel_dist_val, rel_val] = sc_pmax_val
        
min_val, max_val = 0, 1

plt.imshow(wc_vis_array, cmap=plt.cm.Blues)#, extent=[-10,10,50,0])
plt.title('Pmax values for the worst case control vector', size=12)
plt.ylabel('Relative Distance (m)', size=12)
plt.xlabel('Relative velocity (m/s)', size=12)
plt.legend()
plt.savefig('generated/random_td_worst_case_control_vector_%d_td.png' % max_td)
plt.clf()
plt.cla()
plt.close()

# best_control_vector
plt.imshow(bc_vis_array, cmap=plt.cm.Blues)#, extent=[-10,10,50,0])
plt.title('Pmax values for the best case control vector', size=12)
plt.ylabel('Relative Distance (m)', size=12)
plt.xlabel('Relative velocity (m/s)', size=12)
plt.legend()
plt.savefig('generated/random_td_best_case_control_vector_%d_td.png' % max_td)
plt.clf()
plt.cla()
plt.close()

# stay_control_vector
plt.imshow(sc_vis_array, cmap=plt.cm.Blues)#, extent=[-10,10,50,0])
plt.title('Pmax values for the stay control vector', size=12)
plt.ylabel('Relative Distance (m)', size=12)
plt.xlabel('Relative velocity (m/s)', size=12)
plt.legend()
plt.savefig('generated/random_td_stay_control_vector_%d_td.png' % max_td)
plt.clf()
plt.cla()
plt.close()

print(sc_vis_array)
