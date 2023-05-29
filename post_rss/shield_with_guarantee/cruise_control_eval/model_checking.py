import os
import sys
import numpy as np 

def ValueIteration(V, Q, mdp, labels, max_iter=1000, delta=1e-6):
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
        print("max error : ", max_diff)

    return V, Q

def ValueIterationForMinSafety(Vmin, Qmin, threshold, mdp, labels, initial_labels, max_iter=100, delta=1e-8):
    num_states = Qmin.shape[0]
    num_actions = Qmin.shape[1]
    V = np.zeros((num_states,))
    sa_pairs = list(mdp.keys())
    # Q = {sa_pair:0.0 for sa_pair in sa_pairs}
    
    # Start value iteration
    sat = False 
    iter = 0
    while not sat:
        # print("iteration : %d" % i)
        max_diff = 0
    
        for state in range(num_states):
            old_state_value = V[state]

            if labels[state] == 1:
                V[state] = 1.0 
                # Q[state] = [1.0,]*num_actions
                continue    

            state_action_values_list = []    
            for a in range(num_actions):
                state_action_pair = (state, a)
                if not (Qmin[state_action_pair] <= threshold or Qmin[state_action_pair] == Vmin[state]):
                    continue
                next_states = mdp[state_action_pair]
                next_state_values = [V[next_state] for next_state in next_states]
                state_action_value = sum(next_state_values) / len(next_state_values)
                state_action_values_list.append(state_action_value)
                # Q[state_action_pair] = state_action_value

            state_value = max(state_action_values_list) 
            V[state] = state_value
            
            diff = abs(old_state_value - state_value)
            max_diff = max(max_diff, diff)

        if max_diff < delta:
            sat = True 
        
        iter += 1
        if iter > max_iter:
            print("execeded maximum number of iterations")
            print(V[np.where(initial_labels)[0]])
            return 1.0

    #     print("max error : ", max_diff, np.dot(V, initial_labels))
    # print(np.where(np.logical_and(Q < 0.3, Q > 0.5)))
    num_initial_states = np.where(initial_labels)[0].shape[0]
    Vinit = np.dot(V, initial_labels)/num_initial_states
    print(V[np.where(initial_labels)[0]])
    print(Vinit)
    # print(Q)
    return Vinit

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
# ValueIterationForMinSafety(Vmin, Qmin, 0.0001, mdp, bad_labels, initial_labels)

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
        current_reachability = ValueIterationForMinSafety(Vmin, Qmin, threshold, mdp, bad_labels, initial_labels)
        print("the current reachability is %.6f and the current value of threshold is %.6f" % (current_reachability, threshold))
    
        if current_reachability > req_reachability:
            threshold_max = threshold 
        else:
            threshold_min = threshold 

        # if req_reachability - current_reachability < 0.1 and abs(req_reachability - current_reachability) < 0.1:
        if req_reachability > current_reachability:
            guarantee = True
            opt_threshold = threshold
        else: 
            threshold = (threshold_min + threshold_max)/2


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

