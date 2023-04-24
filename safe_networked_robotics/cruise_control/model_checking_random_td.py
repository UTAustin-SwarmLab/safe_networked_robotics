import sys
import csv
import numpy as np 

def ValueIteration(mdp, mdp_states, labels, num_actions, max_iter=10000, delta=1e-2):
    V = {tuple(mdp_state):1 for mdp_state in mdp_states}

    # Start value iteration
    for i in range(max_iter):
        print("iteration : %d" % i)
        max_diff = 0
    
        for mdp_state in mdp_states:
            mdp_state = tuple(mdp_state) 
            pmax_state_value = 0
            old_pmax_state_value = V[mdp_state]

            if labels[mdp_state] == 1:
                V[mdp_state] = pmax_state_value 
                continue        
        
            for a in range(1,num_actions+1):
                #print(mdp_state,a)
                state_action_pair = (mdp_state, a)
                transition = mdp[state_action_pair]
                #print(state_action_pair, transition)
                next_states = [trans[0] for trans in transition]
                next_states_prob = [trans[1] for trans in transition]
                next_state_values = [V[next_state] for next_state in next_states]
                #state_action_value = sum(next_state_values) / len(next_state_values)
                #print(next_state_values, next_states_prob, np.multiply(next_state_values, next_states_prob), sum(np.multiply(next_state_values, next_states_prob)))
                state_action_value = sum(np.multiply(next_state_values, next_states_prob))
                pmax_state_value = max(pmax_state_value, state_action_value)
            
            diff = abs(old_pmax_state_value - pmax_state_value)
            V[mdp_state] = pmax_state_value 

            max_diff = max(max_diff, diff)

        if max_diff < delta:
            print("max error : ", max_diff)
            break
        print("max error : ", max_diff)

    return V

max_td = 4
num_actions = 5
 
random_td_mdp_csv = open("random_generated/mdp_max_td_%d.csv" % max_td, mode ='r')
csvFile = csv.reader(random_td_mdp_csv)

difference = np.array([(max_td+1)*2*((num_actions+1)**(i+1)) for i in reversed(range(max_td))] + [(max_td+1)*2, 2, 1])

def get_mdp_state_from_id(mdp_state_id):
    mdp_state = []
    for diff in difference:
        quo = int(mdp_state_id/diff)
        mdp_state_id = mdp_state_id%diff
        mdp_state.append(quo)
    return tuple(mdp_state)

random_td_mdp = {}
for lines in csvFile:
    #print(lines)
    #next_states = [np.int64(i) for i in lines]
    next_states = lines
    mdp_state = get_mdp_state_from_id(np.int64(next_states[0]))
    #print(mdp_state)
    abstract_action = np.int64(next_states[1])

    next_mdp_states = []
    num_next_states = int((len(lines) - 2)/2)  
    for i in range(2, 2+num_next_states):
        next_state_id = np.int64(next_states[i])
        next_mdp_state = get_mdp_state_from_id(next_state_id)
        next_prob = float(next_states[i+num_next_states])
        next_mdp_states.append((next_mdp_state, next_prob))

    #print(mdp_state, abstract_action)

    random_td_mdp[(mdp_state, abstract_action)] = next_mdp_states

state_action_pairs = list(random_td_mdp.keys())
random_td_mdp_states = [state_action_pairs[i][0] for i in range(0, len(state_action_pairs), num_actions)]

bad_labels = {}
for mdp_state in random_td_mdp_states:
    if mdp_state[0] <= 30:
        bad_labels[mdp_state] = 1
    else:
        bad_labels[mdp_state] = 0

state_pmax_values = ValueIteration(random_td_mdp, random_td_mdp_states, bad_labels, num_actions) 
np.save('generated/random_td_state_values_%d_td' % max_td, state_pmax_values)

state_action_values_dict = {}

for random_td_mdp_state in random_td_mdp_states:
    state_action_values_list = []
    for action in range(1, num_actions+1):
        state_action_pair = (random_td_mdp_state, action)
        transition = random_td_mdp[state_action_pair]
        next_states = [trans[0] for trans in transition]
        next_states_prob = [trans[1] for trans in transition]        
        next_states_pmax_values = [state_pmax_values[pnms] for pnms in next_states]
        state_action_value = sum(np.multiply(next_states_pmax_values, next_states_prob))
        state_action_values_dict[state_action_pair] = state_action_value

np.save('generated/random_td_state_action_values_%d_td' % max_td, state_action_values_dict)