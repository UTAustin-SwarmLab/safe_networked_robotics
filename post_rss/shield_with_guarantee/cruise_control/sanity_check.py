import sys
sys.path.remove('/usr/lib/python3/dist-packages')
import csv
import os 
import numpy as np
import itertools

max_td = 2
num_actions = 5
constant_td_mdp = np.load("generated/mdp_%d_td.npy" % max_td, allow_pickle=True).item()
#print(constant_td_mdp)
 
random_td_mdp_csv = open("generated/mdp_max_td_%d.csv" % max_td, mode ='r')
csvFile = csv.reader(random_td_mdp_csv)

no_td_mdp_path = "generated/mdp_0_td.npy"
zero_td_mdp = np.load(no_td_mdp_path, allow_pickle=True).item()
state_action_pairs = list(zero_td_mdp.keys())
abstract_states = [state_action_pairs[i][0] for i in range(0, len(state_action_pairs), num_actions)]
num_abstract_states = len(abstract_states)

difference = np.array([(max_td+1)*2*((num_actions+1)**(i+1)) for i in reversed(range(max_td))] + [(max_td+1)*2, 2, 1])
print(difference)

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


#print(random_td_mdp) 

###############################################################################################
################### Sanity check for mdp ######################################################
###############################################################################################

abstract_state = 65
ustate = (1,1)
abstract_action = 4

random_td_mdp_state = (abstract_state,) + ustate + (max_td, 0)
random_td_mdp_transition = random_td_mdp[(random_td_mdp_state, abstract_action)]

constant_td_mdp_state = (abstract_state,) + ustate
constant_td_mdp_transition = constant_td_mdp[(constant_td_mdp_state, abstract_action)]

print(random_td_mdp_transition)
print(constant_td_mdp_transition)

###############################################################################################
################### Sanity check for value iteration and pmax values ##########################
###############################################################################################

constant_td_mdp_state_action_values = np.load("generated/state_action_values_%d_td.npy" % max_td, allow_pickle=True).item()
random_td_mdp_state_action_values = np.load("generated/random_td_state_action_values_%d_td.npy" % max_td, allow_pickle=True).item()

val1 = constant_td_mdp_state_action_values[(constant_td_mdp_state, abstract_action)]
val2 = random_td_mdp_state_action_values[(random_td_mdp_state, abstract_action)]

print(val1, val2)


ego_acc_list = [-1, -0.5, 0, 0.5, 1]
num_actions = len(ego_acc_list)
abstract_actions = [i for i in range(1, num_actions+1)]
ustates = list(itertools.product(abstract_actions, repeat=max_td))

constant_td_mdp_state_values = np.load("generated/state_values_%d_td.npy" % max_td, allow_pickle=True).item()
random_td_mdp_state_values = np.load("generated/random_td_state_values_%d_td.npy" % max_td, allow_pickle=True).item()
random_td_mdp_states = list(random_td_mdp_state_values.keys())
"""
for abstract_state in abstract_states:
	for ustate in ustates:
		constant_mdp_state = (abstract_state,) + ustate 
		print(constant_mdp_state)
		random_mdp_states = list(itertools.product([abstract_state], [ustate], [0,1,2], [0,1]))
		random_mdp_states = [(st[0],) + st[1] + tuple(st[2:]) for st in random_mdp_states]
		print(random_mdp_states)
		constant_td_state_value = constant_td_mdp_state_values[constant_mdp_state]
		print(constant_td_state_value)
		random_td_mdp_state_value = [random_td_mdp_state_values[st] for st in random_mdp_states if st in random_td_mdp_states]
		print(random_td_mdp_state_value)
"""
for abstract_state in abstract_states:
	constant_td_ustate = tuple([3 for i in range(max_td)])
	constant_mdp_state = (abstract_state,) + constant_td_ustate
	print(constant_mdp_state)
	constant_td_state_value = constant_td_mdp_state_values[constant_mdp_state]
	print(constant_td_state_value)

	random_td_ustate = tuple([0 for i in range(max_td)])
	random_td_mdp_state = (abstract_state,) + random_td_ustate + (max_td,0)
	print(random_td_mdp_state)
	random_td_mdp_state_value = random_td_mdp_state_values[random_td_mdp_state]
	print(random_td_mdp_state_value)