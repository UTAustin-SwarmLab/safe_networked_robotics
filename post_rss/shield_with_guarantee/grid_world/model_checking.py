import sys 
sys.path.remove('/usr/lib/python3/dist-packages')
import numpy as np 

def ValueIteration(V, Q, mdp, mdp_states, labels, num_actions, max_iter=10000, delta=1e-8):

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
        print("state value : ",V[(7,7,0,0,0,0)])

        if max_diff < delta:
            break
        print("max error : ", max_diff)

    return V, Q

td = int(sys.argv[1])
num_actions = 5

mdp = np.load('constant_generated/mdp_%d_td.npy' % td, allow_pickle=True).item()
state_action_pairs = list(mdp.keys())
mdp_states = [state_action_pairs[i][0] for i in range(0, len(state_action_pairs), num_actions)]

#print(state_action_pairs)
#print(mdp_states)

bad_labels = {}
for mdp_state in mdp_states:
    if mdp_state[0] == mdp_state[2] and mdp_state[1] == mdp_state[3]:
        bad_labels[mdp_state] = 1
    else:
        bad_labels[mdp_state] = 0

V = {tuple(mdp_state):0 for mdp_state in mdp_states}
Q = {tuple(state_action_pair):0 for state_action_pair in state_action_pairs}
V, Q = ValueIteration(V, Q, mdp, mdp_states, bad_labels, num_actions) 
np.save('constant_generated/state_values_%d_td' % td, V)
np.save('constant_generated/state_action_values_%d_td' % td, Q)
print(V)