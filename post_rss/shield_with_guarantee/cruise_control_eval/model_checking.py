import os
import sys
import numpy as np 

def ValueIteration(V, Q, mdp, mdp_states, labels, num_actions, max_iter=10000, delta=1e-6):

    # Start value iteration
    for i in range(max_iter):
        print("iteration : %d" % i)
        max_diff = 0
    
        for mdp_state in mdp_states:
            old_state_value = V[mdp_state]

            if labels[mdp_state] == 1:
                V[mdp_state] = 1.0 
                continue    

            state_action_values_list = []    
            for a in range(num_actions):
                state_action_pair = (mdp_state, a)
                next_states = mdp[state_action_pair]
                next_state_values = [V[next_state] for next_state in next_states]
                state_action_value = sum(next_state_values) / len(next_state_values)
                state_action_values_list.append(state_action_value)
                Q[state_action_pair] = state_action_value

            state_value = min(state_action_values_list)
            V[mdp_state] = state_value
            
            diff = abs(old_state_value - state_value)
            max_diff = max(max_diff, diff)

        if max_diff < delta:
            break 
        print("max error : ", max_diff)

    return V, Q

def ValueIterationForMinSafety(V, Q, Vmin, Qmin, threshold, mdp, mdp_states, labels, num_actions, max_iter=10000, delta=1e-6):

    # Start value iteration
    for i in range(max_iter):
        #print("iteration : %d" % i)
        max_diff = 0
    
        for mdp_state in mdp_states:
            old_state_value = V[mdp_state]

            if labels[mdp_state] == 1:
                V[mdp_state] = 1.0 
                continue    

            state_action_values_list = []    
            for a in range(num_actions):
                state_action_pair = (mdp_state, a)
                if not (Qmin[state_action_pair] <= threshold or Qmin[state_action_pair] == Vmin[mdp_state]):
                    continue
                next_states = mdp[state_action_pair]
                next_state_values = [V[next_state] for next_state in next_states]
                state_action_value = sum(next_state_values) / len(next_state_values)
                state_action_values_list.append(state_action_value)
                Q[state_action_pair] = state_action_value

            state_value = max(state_action_values_list)
            V[mdp_state] = state_value
            
            diff = abs(old_state_value - state_value)
            max_diff = max(max_diff, diff)

        if max_diff < delta:
            break 
        # print("max error : ", max_diff)

    return V, Q

td = int(sys.argv[1]) 
num_actions = 5
inc = num_actions**td

mdp = np.load('constant_generated/mdp_%d_td.npy' % td, allow_pickle=True).item()
state_action_pairs = list(mdp.keys())
mdp_states = [state_action_pairs[i][0] for i in range(0, len(state_action_pairs), num_actions)]
num_states = len(mdp_states)

bad_labels = {}
for mdp_state in mdp_states:
    if mdp_state <= 22*6*inc:
        bad_labels[mdp_state] = 1
    else:
        bad_labels[mdp_state] = 0
#print(bad_labels)

Vmin = {mdp_state:0 for mdp_state in mdp_states}
Qmin = {state_action_pair:0 for state_action_pair in state_action_pairs}
Vmin, Qmin = ValueIteration(Vmin, Qmin, mdp, mdp_states, bad_labels, num_actions) 

threshold_min = 0.0 
threshold_max = 1.0 
threshold = (threshold_min + threshold_max)/2
opt_threshold = threshold

req_reachability = 1 - float(sys.argv[2])
guarantee = False 
old_reachability = 1.0
eps = 1e-6

save_dir = 'constant_generated/%d_td/' % td
os.makedirs(save_dir, exist_ok=True)

if req_reachability == 1:
    shield = np.ones((num_states, num_actions))
    save_loc = os.path.join(save_dir, 'shield_%s_prob.npy' % sys.argv[2])
    np.save(save_loc, shield)
else:
    while not guarantee:
        Vmax = {mdp_state:0 for mdp_state in mdp_states}
        Qmax = {state_action_pair:0 for state_action_pair in state_action_pairs}
        Vmax, Qmax = ValueIterationForMinSafety(Vmax, Qmax, Vmin, Qmin, threshold, mdp, mdp_states, bad_labels, num_actions)
        current_reachability = Vmax[num_states-11*inc]
        if current_reachability > req_reachability:
            threshold_max = threshold 
        else:
            threshold_min = threshold 

        if current_reachability < req_reachability:
            guarantee = True
            opt_threshold = threshold
        else:
            threshold = (threshold_min + threshold_max)/2

        print("the current reachability is %.6f and the current value of threshold is %.6f" % (current_reachability, threshold))

    num_states = len(mdp_states)
    shield = np.zeros((num_states, num_actions))
    for state_action_pair in state_action_pairs:
        state = state_action_pair[0]
        action = state_action_pair[1]
        if Qmin[state_action_pair] <= opt_threshold or Qmin[state_action_pair] == Vmin[state]: 
            shield[state][action] = 1 
        else:
            shield[state][action] = 0
    
    
    save_loc = os.path.join(save_dir, 'shield_%s_prob.npy' % sys.argv[2])
    np.save(save_loc, shield)

