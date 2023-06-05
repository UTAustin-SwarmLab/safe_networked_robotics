import os
import sys
import numpy as np 

def ValueIteration(mdp, labels, eps=1e-10):
    
    num_states = labels.shape[0]
    num_state_action_pairs = len(list(mdp.keys()))
    num_actions = int(num_state_action_pairs/num_states)
    V = np.zeros((num_states,))
    Q = np.zeros((num_states, num_actions))

    # Start value iteration
    guarantee = False
    while not guarantee:
        # print("iteration : %d" % i)
        max_diff = 0
    
        for state in range(num_states):
            old_state_value = V[state]

            state_action_values_list = []    
            for a in range(num_actions): 
                state_action_pair = (state, a)
                next_states = mdp[state_action_pair]
                next_state_values = [V[next_state] for next_state in next_states]
                state_action_value = sum(next_state_values) / len(next_state_values)
                state_action_values_list.append(state_action_value)
                Q[state][a] = state_action_value

            state_value = min(state_action_values_list)
            V[state] = state_value

            if labels[state] == 1:
                V[state] = 1.0 
                Q[state] = [1.0,]*num_actions
            
            diff = abs(old_state_value - V[state])
            max_diff = max(max_diff, diff)

        if max_diff < eps:
            guarantee = True 

        print("max error : ", max_diff)

    return V, Q

td = int(sys.argv[1]) 

"""
loading the mdp, unsafe and initial state labels
"""

mdp = np.load('constant_generated/mdp_%d_td.npy' % td, allow_pickle=True).item()
state_action_pairs = list(mdp.keys())

bad_labels = np.load('constant_generated/unsafe_%d_td.npy' % td, allow_pickle=True)
initial_labels = np.load('constant_generated/initial_%d_td.npy' % td, allow_pickle=True)

# obtaining some basic quantities
num_states = bad_labels.shape[0]
num_state_action_pairs = len(list(mdp.keys()))
num_actions = int(num_state_action_pairs/num_states)
num_initial_states = np.where(initial_labels)[0].shape[0]
print(num_states, num_actions)

# obtaining the maximum safety probability alias minimum reachability
Vmin, Qmin = ValueIteration(mdp, bad_labels)

save_loc = 'constant_generated/Qmin_values_%d_td' % td 
np.save(save_loc, Qmin)