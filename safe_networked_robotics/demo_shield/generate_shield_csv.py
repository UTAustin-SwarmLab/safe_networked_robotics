import sys
sys.path.remove('/usr/lib/python3/dist-packages')
import numpy as np 
import csv 

td = 4
Q = np.load('constant_generated/state_action_values_%d_td.npy' % td, allow_pickle=True).item()
print(Q)

state_action_pairs = Q.keys()

f = open('constant_generated/state_action_safety_values_%d_td.csv' % td, 'w', newline='')
writer = csv.writer(f)
for state_action_pair in state_action_pairs:
    string = ""
    abstract_state = state_action_pair[0][0]
    ustate = state_action_pair[0][1:]
    action = state_action_pair[1]

    string += str(abstract_state)
    string += '-'

    for t in range(td):
        string += str(ustate[t])
        string += '-'
    
    string += str(action)
    safety_val = 1-Q[state_action_pair]
    print(state_action_pair, string, safety_val)

    data = [string, safety_val]
    writer.writerow(data)