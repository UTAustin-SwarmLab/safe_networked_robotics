import sys
import numpy as np 

def ValueIteration(mdp, mdp_states, labels, num_actions, max_iter=10000, delta=1e-6):
    V = {tuple(mdp_state):0 for mdp_state in mdp_states}

    # Start value iteration
    for i in range(max_iter):
        print("iteration : %d" % i)
        max_diff = 0
    
        for mdp_state in mdp_states:
            mdp_state = tuple(mdp_state) 
            old_pmin_state_value = V[mdp_state]

            if labels[mdp_state] == 1:
                V[mdp_state] = 1.0 
                continue    

            state_action_values_list = []    
        
            for a in range(num_actions):
                #print(mdp_state,a)
                state_action_pair = (mdp_state, a)
                next_states = mdp[state_action_pair]
                
                next_state_values = [V[next_state] for next_state in next_states]
                #state_action_value = min(next_state_values)
                state_action_value = sum(next_state_values) / len(next_state_values)
                #pmax_state_value = max(pmax_state_value, state_action_value)
                state_action_values_list.append(state_action_value)

            pmin_state_value = min(state_action_values_list)
            diff = abs(old_pmin_state_value - pmin_state_value)
            V[mdp_state] = pmin_state_value 

            max_diff = max(max_diff, diff)

        if max_diff < delta:
            break
        print("max error : ", max_diff)

    return V

td = int(sys.argv[1])
num_actions = 5

mdp = np.load('constant_generated/mdp_%d_td.npy' % td, allow_pickle=True).item()
state_action_pairs = list(mdp.keys())
mdp_states = [state_action_pairs[i][0] for i in range(0, len(state_action_pairs), num_actions)]
#print(state_action_pairs)
# print(mdp_states)



bad_labels = {}
for mdp_state in mdp_states:
    if mdp_state[0] < 132:
        bad_labels[mdp_state] = 1
    else:
        bad_labels[mdp_state] = 0

state_pmax_values = ValueIteration(mdp, mdp_states, bad_labels, num_actions) 
print(state_pmax_values)

# np.save('constant_generated_2/state_values_%d_td' % td, state_pmax_values)

state_action_values_dict = {}

# for mdp_state in mdp_states:
#     mdp_state = tuple(mdp_state)
#     state_action_values_list = []
#     for action in range(1, num_actions+1):
#         state_action_pair = (mdp_state, action)
#         next_states = mdp[state_action_pair]
        
#         next_states_pmax_values = [state_pmax_values[pnms] for pnms in next_states]
#         state_action_value = sum(next_states_pmax_values) / len(next_states_pmax_values)
#         state_action_values_dict[state_action_pair] = state_action_value

# np.save('constant_generated_2/state_action_values_%d_td' % td, state_action_values_dict)

