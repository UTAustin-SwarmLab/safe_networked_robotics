import os
import sys
import numpy as np 

def ValueIteration(mdp, bad_labels, goal_labels, eps=1e-8):
    actions_dict = {0:'stay', 1:'up', 2:'right', 3:'down', 4:'left'}
    actions_list = list(actions_dict.keys())
    num_actions = len(actions_list)

    num_states = bad_labels.shape[0]
    
    V = np.zeros((num_states,))
    Q = np.zeros((num_states, num_actions)) # so that while subtracting from 1 for safety, unused actions will have zero safety

    # Start value iteration
    guarantee = False
    while not guarantee:
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

            state_value = max(state_action_values_list)
            V[state] = state_value

            if bad_labels[state] == 1:
                V[state] = 0.0 
                Q[state] = [0.0,]*num_actions
            
            if goal_labels[state] == 1:
                V[state] = 1.0
                Q[state] = [1.0,]*num_actions
            
            diff = abs(old_state_value - V[state])
            max_diff = max(max_diff, diff)
        
        if max_diff < eps:
            guarantee = True

        print("max error : ", max_diff)
             
    return V, Q

def ValueIterationForActualSafety(Vmax, Qmax, delta, mdp, bad_labels, goal_labels, policy, eps=1e-8):
    num_states = Qmax.shape[0]
    num_actions = Qmax.shape[1]
    V = np.zeros((num_states,))

    hybrid_policy = {}
    for st_idx in range(num_states):
        per_state_filtered_actions = []
        for a_idx in range(num_actions):
            if (Qmax[st_idx][a_idx] >= delta or Qmax[st_idx][a_idx] == Vmax[st_idx]):
                per_state_filtered_actions.append(a_idx)

        task_values = [policy[st_idx][act_ii] for act_ii in range(num_actions)]
        indexed_list = list(enumerate(task_values))
        sorted_list = sorted(indexed_list, key=lambda x: x[1], reverse=True)
        act_indices = [index for index, _ in sorted_list]
        
        for act_idx in act_indices:
            if act_idx in per_state_filtered_actions:
                hybrid_policy[st_idx] = act_idx 
                break

    # Start policy evaluation
    guarantee = False
    while not guarantee:
        max_diff = 0
    
        for state in range(num_states):
            old_state_value = V[state]

            action = hybrid_policy[state]
            state_action_pair = (state, action)
            next_states = mdp[state_action_pair]
            next_state_values = [V[next_state] for next_state in next_states]
            V[state] = sum(next_state_values) / len(next_state_values)

            if bad_labels[state] == 1:
                V[state] = 0.0 
            
            if goal_labels[state] == 1:
                V[state] = 1.0
            
            diff = abs(old_state_value - V[state])
            max_diff = max(max_diff, diff)
        
        if max_diff < eps:
            guarantee = True

        print("max error : ", max_diff)
             
    return policy, V
             


td = int(sys.argv[1]) 

"""
loading the mdp, unsafe and initial state labels
"""

mdp = np.load('constant_generated_backup/mdp_%d_td.npy' % td, allow_pickle=True).item()
state_action_pairs = list(mdp.keys())

bad_labels = np.load('constant_generated_backup/unsafe_%d_td.npy' % td, allow_pickle=True)
initial_labels = np.load('constant_generated_backup/initial_%d_td.npy' % td, allow_pickle=True)
goal_labels = np.load('constant_generated_backup/goal_%d_td.npy' % td, allow_pickle=True)

# obtaining some basic quantities
num_states = bad_labels.shape[0]
num_state_action_pairs = len(list(mdp.keys()))
num_actions = int(num_state_action_pairs/num_states)
num_initial_states = np.where(initial_labels)[0].shape[0]

print(num_states, num_state_action_pairs, num_actions, num_initial_states)

# obtaining the maximum safety probability alias minimum reachability
Vmax, Qmax= ValueIteration(mdp, bad_labels, goal_labels)
max_safety = np.dot(Vmax, initial_labels) / num_initial_states
print("the maximum safety achievable is ", max_safety)

delta = float(sys.argv[2])

abstracted_policy_loc = 'constant_generated_backup/abstracted_q_policy_%d_td.npy' % td 
policy = np.load(abstracted_policy_loc, allow_pickle=True)
# print(policy)

# save_dir = 'constant_generated/%d_td/' % td
# os.makedirs(save_dir, exist_ok=True)

# obtaining the actual safety probability for the controller that we use - here the controller is move as fast as possible
act_policy, Vactual = ValueIterationForActualSafety(Vmax, Qmax, delta, mdp, bad_labels, goal_labels, policy)

actual_safety = np.dot(Vactual, initial_labels) / num_initial_states
print("the actual safety for delta=%f is %.16f" % (delta, actual_safety))
print(np.where(initial_labels))
