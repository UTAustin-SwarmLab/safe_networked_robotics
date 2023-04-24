import sys
sys.path.remove('/usr/lib/python3/dist-packages')
import numpy as np 
import csv 

invalid = 9
td = 4
Q = np.load('random_generated/state_action_values_%d_td.npy' % td, allow_pickle=True).item()


state_action_pairs = Q.keys()

f = open('random_generated/state_action_safety_values_%d_rand_td.csv' % td, 'w', newline='')
writer = csv.writer(f)
for state_action_pair in state_action_pairs:    
    string = ""
    abstract_state = state_action_pair[0][0]
    ustate = state_action_pair[0][1:-1]
    itm = state_action_pair[0][-1]
    if itm > 0:
        continue
    action = state_action_pair[1]

    string += str(abstract_state)
    string += '-'

    for t in range(td):
        if ustate[t] != -1:
            string += str(ustate[t])
        else:
            string += str(invalid)
        string += '-'
    
    string += str(action)
    safety_val = 1-Q[state_action_pair]
    print(state_action_pair, string, safety_val)

    data = [string, safety_val]
    writer.writerow(data)