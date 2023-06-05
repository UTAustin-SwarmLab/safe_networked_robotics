import os
import sys
import numpy as np 
from decimal import Decimal

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

        # print("max error : ", max_diff)

    return V, Q

def ValueIterationForMinSafety(Vmin, Qmin, threshold, mdp, labels, eps=1e-10):
    num_states = Qmin.shape[0]
    num_actions = Qmin.shape[1]
    V = np.zeros((num_states,))
    Q = np.ones((num_states, num_actions)) # so that while subtracting from 1 for safety, unused actions will have zero safety

    policy = {}
    for st_idx in range(num_states):
        per_state_filtered_actions = []
        for a_idx in range(num_actions):
            if (Qmin[(st_idx,a_idx)] <= threshold or Qmin[(st_idx,a_idx)] == Vmin[st_idx]):
                per_state_filtered_actions.append(a_idx)
        # policy[st_idx] = [int(max(per_state_filtered_actions)),]
        policy[st_idx] = per_state_filtered_actions
    # Start value iteration
    guarantee = False
    while not guarantee:
        max_diff = 0
    
        for state in range(num_states):
            old_state_value = V[state]

            state_action_values_list = []  
            allowed_actions = policy[state]
            for a in allowed_actions:
                state_action_pair = (state, a)
                next_states = mdp[state_action_pair]
                next_state_values = [V[next_state] for next_state in next_states]
                state_action_value = sum(next_state_values) / len(next_state_values)
                state_action_values_list.append(state_action_value)
                Q[state][a] = state_action_value

            state_value = max(state_action_values_list)
            V[state] = state_value

            if labels[state] == 1:
                V[state] = 1.0 
                Q[state] = [1.0,]*num_actions
            
            diff = abs(old_state_value - V[state])
            max_diff = max(max_diff, diff)
        
        if max_diff < eps:
            guarantee = True
             
    return policy, V, Q

def ValueIterationForActualSafety(Vmin, Qmin, threshold, mdp, labels, eps=1e-10):
    num_states = Qmin.shape[0]
    num_actions = Qmin.shape[1]
    V = np.zeros((num_states,))
    abstracted_dnn_policy = np.load('abstracted_dnn_policy.npy', allow_pickle=True).item()
    # print(abstracted_dnn_policy)

    hybrid_policy = np.zeros((num_states,))
    for st_idx in range(num_states):
        per_state_filtered_actions = []
        for a_idx in range(num_actions):
            if (Qmin[(st_idx,a_idx)] <= threshold or Qmin[(st_idx,a_idx)] == Vmin[st_idx]):
                per_state_filtered_actions.append(a_idx)
        if abstracted_dnn_policy[st_idx] in per_state_filtered_actions:
            hybrid_policy[st_idx] = int(abstracted_dnn_policy[st_idx])
        else:
            print("here", st_idx)
            hybrid_policy[st_idx] = int(max(per_state_filtered_actions))
    
    # Start value iteration
    guarantee = False
    while not guarantee:
        max_diff = 0
    
        for state in range(num_states):
            old_state_value = V[state]
            next_states = mdp[(state, hybrid_policy[state])]
            next_state_values = [V[next_state] for next_state in next_states]
            V[state] = sum(next_state_values) / len(next_state_values)

            if labels[state] == 1:
                V[state] = 1.0 
            
            diff = abs(old_state_value - V[state])
            max_diff = max(max_diff, diff)

        if max_diff < eps:
            guarantee = True 
        
    return hybrid_policy, V

def construct_shield(Vmin, Qmin, threshold):
    num_states = Qmin.shape[0]
    num_actions = Qmin.shape[1]
    shield = np.zeros((num_states, num_actions))
    for state_action_pair in state_action_pairs:
        state = state_action_pair[0]
        action = state_action_pair[1]
        if Qmin[state_action_pair] <= threshold or Qmin[state_action_pair] == Vmin[state]: 
            shield[state][action] = 1 
        else:
            shield[state][action] = 0

    return shield

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

# obtaining the maximum safety probability alias minimum reachability
Vmin, Qmin = ValueIteration(mdp, bad_labels)

delta = float(sys.argv[2])
threshold = 1-delta

# obtaining the minimum safety probability alias maximum reachability
min_policy, Vmax, Qmax = ValueIterationForMinSafety(Vmin, Qmin, threshold, mdp, bad_labels)
print(Vmax)
# print(min_policy)
# max_reachability = np.dot(Vmax, initial_labels) / num_initial_states
# min_safety = 1-max_reachability
# print("the minimum safety for delta=%f is %f" % (delta, min_safety))

# save_dir = 'constant_generated/%d_td/' % td
# os.makedirs(save_dir, exist_ok=True)

# obtaining the actual safety probability for the controller that we use - here the controller is move as fast as possible
act_policy, Vactual = ValueIterationForActualSafety(Vmin, Qmin, threshold, mdp, bad_labels)
# print(act_policy)

# for state_id in range(num_states):
#     print(state_id, min_policy[state_id], Vmax[state_id], Qmax[state_id][min_policy[state_id]], act_policy[state_id], Vactual[state_id])
actual_reachability = np.dot(Vactual, initial_labels) / num_initial_states
actual_safety = 1-actual_reachability 
print("the actual safety for delta=%f is %f" % (delta, actual_safety))

# print(mdp[(31,1)])
# print(mdp[(32,3)])

# print("checking if 31 is a safe state")
# current_states = [31,]
# for i in range(10000000):
#     next_states = []
#     for st in current_states:
#         # allowed_actions = [act_policy[st],]
#         allowed_actions = min_policy[st]
#         for act in allowed_actions:
#             next_states.append(mdp[st, act])
#     next_states = list(np.unique(next_states))
#     print("iteration", i)  
#     print("next states : ", next_states, V[next_states])
#     current_states = next_states
