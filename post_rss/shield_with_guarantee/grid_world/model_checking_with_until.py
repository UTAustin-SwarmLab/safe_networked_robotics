import sys 
import numpy as np 

def ValueIteration(V, Q, mdp, mdp_states, bad_labels, good_labels, num_actions, eps=1e-6):
    prev = 1.0
    optimal = False

    while not optimal:
        max_diff = 0
     
        for mdp_state in mdp_states:
            old_state_value = V[mdp_state]

            if bad_labels[mdp_state] == 1:
                V[mdp_state] = 0.0 
                continue

            if good_labels[mdp_state] == 1:
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

            state_value = max(state_action_values_list)
            V[mdp_state] = state_value
            
            diff = abs(old_state_value - state_value)
            max_diff = max(max_diff, diff)

        if max_diff < eps:
            optimal = True
        print("max error : ", max_diff)

    return V, Q

def ValueIterationforMinSafety(V, Q, Vmax, Qmax, mdp, mdp_states, bad_labels, good_labels, threshold, num_actions, eps=1e-6):
    prev = 1.0
    optimal = False

    while not optimal:
        max_diff = 0
     
        for mdp_state in mdp_states:
            old_state_value = V[mdp_state]

            if bad_labels[mdp_state] == 1:
                V[mdp_state] = 0.0 
                continue

            if good_labels[mdp_state] == 1:
                V[mdp_state] = 1.0 
                continue    

            state_action_values_list = []    
            for a in range(num_actions):
                state_action_pair = (mdp_state, a)
                if not (Qmax[state_action_pair] >= threshold or Qmax[state_action_pair] == Vmax[mdp_state]):
                    continue
                next_states = mdp[state_action_pair]
                next_state_values = [V[next_state] for next_state in next_states]
                state_action_value = sum(next_state_values) / len(next_state_values)
                state_action_values_list.append(state_action_value)
                Q[state_action_pair] = state_action_value

            state_value = min(state_action_values_list)
            V[mdp_state] = state_value
            
            diff = abs(old_state_value - state_value)
            max_diff = max(max_diff, diff)

        if max_diff < eps:
            optimal = True
        print("max error : ", max_diff)

    return V, Q

td = int(sys.argv[1])
num_actions = 4
xmax = 7

mdp = np.load('constant_generated/mdp_%d_td.npy' % td, allow_pickle=True).item()
state_action_pairs = list(mdp.keys())
mdp_states = [state_action_pairs[i][0] for i in range(0, len(state_action_pairs), num_actions)]

# print(state_action_pairs)
# print(mdp_states)

bad_labels = {}
for mdp_state in mdp_states:
    if mdp_state[0] == mdp_state[2] and mdp_state[1] == mdp_state[3]:
        bad_labels[mdp_state] = 1        
    else:
        bad_labels[mdp_state] = 0
        
good_labels = {}
for mdp_state in mdp_states:
    good_labels[mdp_state] = 0
    if bad_labels[mdp_state]:
        good_labels[mdp_state] = 0
    else:
        if mdp_state[0] == xmax and mdp_state[1] == xmax:
            good_labels[mdp_state] = 1

initial_states = []
for mdp_state in mdp_states:
    if mdp_state[0] > xmax/2 and mdp_state[2] < xmax/2 and mdp_state[1] < xmax/2 and mdp_state[3] < xmax/2 and mdp_state[-1]:
        initial_states.append(mdp_state)

print(initial_states)

Vmax = {tuple(mdp_state):1 for mdp_state in mdp_states}
Qmax = {tuple(state_action_pair):1 for state_action_pair in state_action_pairs}
Vmax, Qmax = ValueIteration(Vmax, Qmax, mdp, mdp_states, bad_labels, good_labels, num_actions)

Vmax_init = [Vmax[init_state] for init_state in initial_states]
print(sum(Vmax_init)/len(Vmax_init))

delta = 0.999

Vmin = {tuple(mdp_state):1 for mdp_state in mdp_states}
Qmin = {tuple(state_action_pair):1 for state_action_pair in state_action_pairs}
Vmin, Qmin = ValueIterationforMinSafety(Vmin, Qmin, Vmax, Qmax, mdp, mdp_states, bad_labels, good_labels, delta, num_actions)

Vmin_init = [Vmin[init_state] for init_state in initial_states]
print(sum(Vmin_init)/len(Vmin_init))


# np.save('constant_generated/state_values_%d_td' % td, V)
# np.save('constant_generated/state_action_values_%d_td' % td, Q)
#print(V)
