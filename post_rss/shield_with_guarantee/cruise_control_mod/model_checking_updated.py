import os
import sys
import numpy as np 
from decimal import Decimal

def ValueIteration(V, Q, mdp, labels, max_iter=10000, delta=1e-6):
    num_states = Q.shape[0]
    num_actions = Q.shape[1]

    # Start value iteration
    for i in range(max_iter):
        # print("iteration : %d" % i)
        max_diff = 0
    
        for state in range(num_states):
            old_state_value = V[state]

            if labels[state] == 1:
                V[state] = 1.0 
                Q[state] = [1.0,]*num_actions
                continue    

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
            
            diff = abs(old_state_value - state_value)
            max_diff = max(max_diff, diff)

        if max_diff < delta:
            break 
        # print("max error : ", max_diff)

    return V, Q

def ValueIterationForActualSafety(Vmin, Qmin, threshold, mdp, labels, initial_labels, max_iter=1000000000000, eps=1e-14):
    num_states = Qmin.shape[0]
    num_actions = Qmin.shape[1]
    V = np.zeros((num_states,))
    
    # Start value iteration
    for i in range(max_iter):
        # print("iteration : %d" % i)
        max_diff = 0
    
        for state in range(num_states):
            old_state_value = V[state]

            if labels[state] == 1:
                V[state] = 1.0 
                continue    

            allowed_actions = []    
            for a in range(num_actions):
                if (Qmin[(state,a)] <= threshold or Qmin[(state,a)] == Vmin[state]):
                    allowed_actions.append(a)
            # print(allowed_actions)
            action = max(allowed_actions)
            # print(action)
            state_action_pair = (state, action)
            next_states = mdp[state_action_pair]
            next_state_values = [V[next_state] for next_state in next_states]
            state_action_value = sum(next_state_values) / len(next_state_values)
            V[state] = state_action_value
            
            diff = abs(old_state_value - V[state])
            max_diff = max(max_diff, diff)
        if max_diff < eps:
            break 
        print(max_diff)
    num_initial_states = np.where(initial_labels)[0].shape[0]
    print(V[initial_labels == 1.0])
    return np.dot(V, initial_labels)/num_initial_states

td = int(sys.argv[1]) 

mdp = np.load('constant_generated/mdp_%d_td.npy' % td, allow_pickle=True).item()
state_action_pairs = list(mdp.keys())

bad_labels = np.load('constant_generated/unsafe_%d_td.npy' % td, allow_pickle=True)
initial_labels = np.load('constant_generated/initial_%d_td.npy' % td, allow_pickle=True)

num_states = bad_labels.shape[0]
mdp_states = np.arange(num_states)
num_actions = int(len(state_action_pairs)/num_states)

Vmin = np.zeros((num_states,))
Qmin = np.zeros((num_states, num_actions))
Vmin, Qmin = ValueIteration(Vmin, Qmin, mdp, bad_labels) 
# print(Qmin[np.where(initial_labels)[0]])
max_init_safety = 1-Vmin[np.where(1-bad_labels)[0]]

save_dir = 'constant_generated/%d_td/' % td
os.makedirs(save_dir, exist_ok=True)

delta = 0.95
threshold = 1-delta

current_reachability = ValueIterationForActualSafety(Vmin, Qmin, threshold, mdp, bad_labels, initial_labels)
current_safety = 1 - current_reachability
print(current_safety)