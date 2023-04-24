import sys
import itertools
import numpy as np 
import matplotlib.pyplot as plt


num_rel_dist_states = 51 
num_rel_vel_states = 20
num_physical_state_features = 2

td = int(sys.argv[1])
per_td_vis_arrays = {}

state_values = np.load('generated/own_state_values_%d_td.npy' % td, allow_pickle=True).item()
states = list(state_values.keys())

wc_vis_array = np.zeros((num_rel_dist_states, num_rel_vel_states))
bc_vis_array = np.zeros((num_rel_dist_states, num_rel_vel_states))
sc_vis_array = np.zeros((num_rel_dist_states, num_rel_vel_states))

for rel_dist_val in range(num_rel_dist_states):
    for rel_val in range(num_rel_vel_states):
        physical_state = (rel_dist_val, rel_val)

        relevant_states = [s for s in states if s[:num_physical_state_features] == physical_state]
        relevant_states_values = [state_values[s] for s in relevant_states]

        # worst_control_vector
        wc_pmax_val = min(relevant_states_values)
        wc_vis_array[rel_dist_val, rel_val] = wc_pmax_val 

        # best_control_vector
        bc_pmax_val = max(relevant_states_values) 
        bc_vis_array[rel_dist_val, rel_val] = bc_pmax_val 

        # stay control vector
        stay_control_vector = tuple([2 for t in range(td)])
        state = physical_state + stay_control_vector
        sc_pmax_val = state_values[state]
        sc_vis_array[rel_dist_val, rel_val] = sc_pmax_val


min_val, max_val = 0, 1

# worst_control_vector
fig_wc, ax_wc = plt.subplots(1,1)
ax_wc.imshow(wc_vis_array, cmap=plt.cm.Blues, extent=[-10,10,50,0])

fig_wc.suptitle('Pmax values for the worst case control vector')
fig_wc.supylabel('Relative Distance (m)')
fig_wc.supxlabel('Relative velocity (m/s)')

plt.tight_layout()
plt.savefig('generated/worst_case_control_vector_%d_td.png' % td)


# best_control_vector
fig_bc, ax_bc = plt.subplots(1,1)
ax_bc.matshow(bc_vis_array, cmap=plt.cm.Blues, extent=[-10,10,50,0])

fig_bc.suptitle('Pmax values for the best case control vector')
fig_bc.supylabel('Relative Distance (m)')
fig_bc.supxlabel('Relative velocity (m/s)')

plt.tight_layout()
plt.savefig('generated/best_case_control_vector_%d_td.png' % td)

# stay_control_vector
fig_sc, ax_sc = plt.subplots(1,1)
ax_sc.matshow(sc_vis_array, cmap=plt.cm.Blues, extent=[-10,10,50,0])

fig_sc.suptitle('Pmax values for the stay control vector')
fig_sc.supylabel('Relative Distance (m)')
fig_sc.supxlabel('Ego-Vehicle velocity (m/s)')

plt.tight_layout()
plt.savefig('generated/stay_control_vector_%d_td.png' % td)

#print(wc_vis_array)
#print(bc_vis_array)
print(wc_vis_array-bc_vis_array)