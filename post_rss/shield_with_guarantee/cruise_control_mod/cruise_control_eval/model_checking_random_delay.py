import sys
sys.path.remove('/usr/lib/python3/dist-packages')
import numpy as np 

def ValueIteration(V, Q, mdp, mdp_states, labels, num_actions, prob, max_iter=10000, delta=1e-2):

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
                next_prob = np.array(prob[state_action_pair])
                #print(next_prob)
                next_state_values = np.array([V[next_state] for next_state in next_states])
                
                state_action_value = np.sum(next_prob*next_state_values)

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
 
max_td = 3
num_actions = 5

mdp = np.load('random_generated/random_mdp_%d_td.npy' % max_td, allow_pickle=True).item()
state_action_pairs = list(mdp.keys())
mdp_states = [state_action_pairs[i][0] for i in range(0, len(state_action_pairs), num_actions)]

bad_labels = {}
for mdp_state in mdp_states:
    if mdp_state[0] <= 22*6:
        bad_labels[mdp_state] = 1
    else:
        bad_labels[mdp_state] = 0

"""
probabilities
"""
prob = np.load('random_generated/random_mdp_prob_%d_td.npy' % max_td, allow_pickle=True).item()
V = {tuple(mdp_state):0 for mdp_state in mdp_states}
Q = {tuple(state_action_pair):0 for state_action_pair in state_action_pairs}
V, Q = ValueIteration(V, Q, mdp, mdp_states, bad_labels, num_actions, prob) 
np.save('random_generated/state_values_%d_td' % max_td, V)
np.save('random_generated/state_action_values_%d_td' % max_td, Q)
