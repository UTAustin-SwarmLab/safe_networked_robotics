import sys
import numpy as np 

def ValueIteration(mdp, prob, mdp_states, zero_td_bad_labels, zero_td_goal_labels, zero_td_init_labels, td, eps=1e-5):
    actions_dict = np.load('actions.npy', allow_pickle=True).item()
    actions_list = list(actions_dict.keys())
    num_actions = len(actions_list)

    V = {state:0 for state in mdp_states}

    # Start value iteration
    guarantee = False
    while not guarantee:
        # print("iteration : %d" % i) 
        max_diff = 0
    
        for state in mdp_states:
            old_state_value = V[state]

            state_action_values_list = []     
            for a in range(num_actions): 
                state_action_pair = (state, a)
                next_states = mdp[state_action_pair]
                next_state_values = [V[next_states[ii]] for ii in range(len(next_states))]
                next_state_probs = prob[state_action_pair] 
                assert round(sum(next_state_probs),15) == 1.0
                state_action_value = sum([nsv*nsp for nsv,nsp in zip(next_state_values, next_state_probs)])
                state_action_values_list.append(state_action_value)

            state_value = max(state_action_values_list)
            V[state] = state_value

            physical_state = (state[0],)
            if zero_td_bad_labels[physical_state] == 1:
                V[state] = 0.0
            
            if zero_td_goal_labels[physical_state] == 1:
                V[state] = 1.0
            
            diff = abs(old_state_value - V[state])
            max_diff = max(max_diff, diff)

        if max_diff < eps:
            guarantee = True 

        print("max error : ", max_diff)

    V_init = {}
    init_ustate = (-1,)*td
    for state in mdp_states:
        physical_state = (state[0],)
        ustate = state[1:-1]            
        itm = state[-1] 
        if zero_td_init_labels[physical_state] == 1 and ustate == init_ustate and itm == 0:
            V_init[state] = V[state] 

    return V, V_init

max_td = int(sys.argv[1]) 
 
"""
loading the mdp, unsafe and initial state labels
"""

mdp = np.load('random_generated/mdp_transitions_%d_td.npy' % max_td, allow_pickle=True).item()
prob = np.load('random_generated/mdp_probabilities_%d_td.npy' % max_td, allow_pickle=True).item()
print("here")

mdp_states = list(np.load('random_generated/states_%d_td.npy' % max_td, allow_pickle=True).item().values())

zero_td_bad_labels = np.load('constant_generated/unsafe_0_td.npy', allow_pickle=True).item()
zero_td_goal_labels = np.load('constant_generated/goal_0_td.npy', allow_pickle=True).item()
zero_td_initial_labels = np.load('constant_generated/initial_0_td.npy', allow_pickle=True).item()

# # obtaining the maximum safety probability alias minimum reachability
print("here")
Vmax, Vmax_init = ValueIteration(mdp, prob, mdp_states, zero_td_bad_labels, zero_td_goal_labels, zero_td_initial_labels, max_td)
print(Vmax_init)
init_states_safety_values = list(Vmax_init.values())
max_safety = sum(init_states_safety_values)/len(init_states_safety_values) 

print("the maximum safety achievable is ", max_safety)

save_loc = 'random_generated/Vmax_values_%d_td' % max_td
np.save(save_loc, Vmax) 

