import os
import sys
import numpy as np 


def ValueIteration(mdp, bad_labels, invalid_labels, num_actions, eps=1e-10):
    
    num_states = bad_labels.shape[0]
    V = np.zeros((num_states,))
    Q = np.zeros((num_states, num_actions))

    # Start value iteration
    guarantee = False
    while not guarantee:
        # print("iteration : %d" % i)
        max_diff = 0
    
        for state in range(num_states):
            if invalid_labels[state]: # donot involve with the invalid states
                continue
            old_state_value = V[state]

            state_action_values_list = []    
            for a in range(num_actions): # donot involve with the invalid action
                state_action_pair = (state, a+1)
                next_states = mdp[state_action_pair]
                next_state_values = [V[next_state] for next_state in next_states]
                state_action_value = sum(next_state_values) / len(next_state_values)
                state_action_values_list.append(state_action_value)
                Q[state][a] = state_action_value

            state_value = min(state_action_values_list)
            V[state] = state_value

            if bad_labels[state] == 1:
                V[state] = 1.0 
                Q[state] = [1.0,]*num_actions
            
            diff = abs(old_state_value - V[state])
            max_diff = max(max_diff, diff)

        if max_diff < eps:
            guarantee = True 

        print("max error : ", max_diff)

    return V, Q

def ValueIterationForMinSafety(Vmin, Qmin, threshold, mdp, bad_labels, invalid_labels, num_actions, eps=1e-10):
    num_states = Qmin.shape[0]
    V = np.zeros((num_states,))

    policy = {}
    for st_idx in range(num_states):
        per_state_filtered_actions = []
        for a_idx in range(num_actions):
            if (Qmin[(st_idx,a_idx)] <= threshold or Qmin[(st_idx,a_idx)] == Vmin[st_idx]):
                per_state_filtered_actions.append(a_idx)
        policy[st_idx] = per_state_filtered_actions
    
    # Start value iteration
    guarantee = False
    while not guarantee:
        max_diff = 0
    
        for state in range(num_states):
            if invalid_labels[state]:
                continue
            old_state_value = V[state]

            state_action_values_list = []  
            allowed_actions = policy[state]  
            for a in allowed_actions:
                state_action_pair = (state, a+1)
                next_states = mdp[state_action_pair]
                next_state_values = [V[next_state] for next_state in next_states]
                state_action_value = sum(next_state_values) / len(next_state_values)
                state_action_values_list.append(state_action_value)

            V[state] = max(state_action_values_list)

            if bad_labels[state] == 1:
                V[state] = 1.0 
            
            diff = abs(old_state_value - V[state])
            max_diff = max(max_diff, diff)
        
        if max_diff < eps:
            guarantee = True

        print("max error : ", max_diff)
            
    return V

def ValueIterationForActualSafety(Vmin, Qmin, threshold, mdp, bad_labels, invalid_labels, num_actions, policy, eps=1e-10):
    num_states = Qmin.shape[0]
    V = np.zeros((num_states,))

    hybrid_policy = {}
    for st_idx in range(num_states):
        # first we obtain the delta shield as in our RSS paper
        per_state_filtered_actions = []
        for a_idx in range(num_actions):
            if (Qmin[(st_idx,a_idx)] <= threshold or Qmin[(st_idx,a_idx)] == Vmin[st_idx]):
                per_state_filtered_actions.append(a_idx)
        
        if policy[st_idx] in per_state_filtered_actions:
            hybrid_policy[st_idx] = int(policy[st_idx])
        else:
            hybrid_policy[st_idx] = int(max(per_state_filtered_actions))

    # Start value iteration
    guarantee = False
    while not guarantee:
        max_diff = 0
    
        for state in range(num_states):
            if invalid_labels[state]:
                continue
            old_state_value = V[state]

            action = hybrid_policy[state]
            state_action_pair = (state, action+1)
            next_states = mdp[state_action_pair]
            next_state_values = [V[next_state] for next_state in next_states]
            state_action_value = sum(next_state_values) / len(next_state_values)
            V[state] = state_action_value

            if bad_labels[state] == 1:
                V[state] = 1.0 
            
            diff = abs(old_state_value - V[state])
            max_diff = max(max_diff, diff)

        if max_diff < eps:
            guarantee = True 

        print("max error : ", max_diff)
        
    return V

def construct_shield(Vmin, Qmin, threshold):
    num_states = Qmin.shape[0]
    num_actions = Qmin.shape[1]
    shield = np.zeros((num_states, num_actions))
    for state in range(num_states):
        for action in range(num_actions):
            if Qmin[state][action] <= threshold or Qmin[state][action] == Vmin[state]: 
                shield[state][action] = 1 
            else:
                shield[state][action] = 0

    return shield

max_td = int(sys.argv[1]) 

delta = float(sys.argv[3])
threshold = 1-delta

"""
loading the mdp, unsafe and initial state labels
"""

mdp = np.load('random_generated/random_mdp_%d_td.npy' % max_td, allow_pickle=True).item()
prob = np.load('random_generated/random_mdp_prob_%d_td.npy' % max_td, allow_pickle=True).item()
state_action_pairs = list(mdp.keys())
# print(mdp)

bad_labels = np.load('random_generated/random_mdp_unsafe_%d_td.npy' % max_td, allow_pickle=True)
initial_labels = np.load('random_generated/random_mdp_initial_%d_td.npy' % max_td, allow_pickle=True)
invalid_labels = np.load('random_generated/random_mdp_invalid_%d_td.npy' % max_td, allow_pickle=True)

# obtaining some basic quantities
num_actions = 4 # do not forget to update this
num_initial_states = np.where(initial_labels)[0].shape[0]

# obtaining the maximum safety probability alias minimum reachability
Vmin, Qmin = ValueIteration(mdp, bad_labels, invalid_labels, num_actions)

min_reachability = np.dot(Vmin, initial_labels) / num_initial_states
max_safety = 1-min_reachability

print("the maximum safety for delta=%f is %f" % (delta, max_safety))

save_dir = 'random_generated/%d_td/' % max_td
os.makedirs(save_dir, exist_ok=True)

max_safety_values_path = os.path.join(save_dir, 'max_safety_state_values.npy')
np.save(max_safety_values_path, 1-Vmin)

# obtaining the minimum safety probability alias maximum reachability
# Vmax = ValueIterationForMinSafety(Vmin, Qmin, threshold, mdp, bad_labels, invalid_labels, num_actions)

# max_reachability = np.dot(Vmax, initial_labels) / num_initial_states
# min_safety = 1-max_reachability

# print("the minimum safety for delta=%f is %f" % (delta, min_safety))

# obtaining the actual safety probability for the controller that we use - here the controller is the abstracted deep neural network 
abstracted_dnn_policy = np.load('random_generated/abstracted_dnn_policy_%d_td.npy' % max_td, allow_pickle=True).item()
Vactual = ValueIterationForActualSafety(Vmin, Qmin, threshold, mdp, bad_labels, invalid_labels, num_actions, abstracted_dnn_policy)
actual_reachability = np.dot(Vactual, initial_labels) / num_initial_states
actual_safety = 1-actual_reachability 
print("the actual safety for delta=%f is %f" % (delta, actual_safety))

shield = construct_shield(Vmin, Qmin, threshold)

save_loc = os.path.join(save_dir, 'shield_%s_prob.npy' % sys.argv[2])
print(save_loc)
np.save(save_loc, shield)