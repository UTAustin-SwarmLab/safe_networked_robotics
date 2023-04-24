import sys
sys.path.remove('/usr/lib/python3/dist-packages')
import numpy as np 

def ValueIteration(V, mdp, mdp_states, bad_labels, good_labels, num_actions, prob, max_iter=1000, delta=1e-8):

	# Start value iteration
	for i in range(max_iter):
		print("iteration : %d" % i)
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
				next_prob = np.array(prob[state_action_pair])
				next_state_values = np.array([V[next_state] for next_state in next_states])				
				state_action_value = np.sum(next_prob*next_state_values)
				state_action_values_list.append(state_action_value)

			state_value = max(state_action_values_list)
			V[mdp_state] = state_value
			
			diff = abs(old_state_value - state_value)
			max_diff = max(max_diff, diff)

		if max_diff < delta or i > max_iter: 
			print(max_diff)
			break
		print("max error : ", max_diff)
		print(V[(441,-1,-1,-1,0)])
	return V#, Q
 
max_td = 3
xmax = 7
num_actions = 5

mdp = np.load('random_generated/random_mdp_%d_td.npy' % max_td, allow_pickle=True).item()
state_action_pairs = list(mdp.keys())
mdp_states = [state_action_pairs[i][0] for i in range(0, len(state_action_pairs), num_actions)]


bad_labels = {}
for mdp_state in mdp_states:
	#print(mdp_state, mdp_state[0])
	val = mdp_state[0]
	physical_state = []
	for k in range(1,6):
		v = int(val % 10) 
		physical_state.append(v)
		val = np.floor(val / 10)
	physical_state.reverse()
	#print(physical_state)	

	if physical_state[0] == physical_state[2] and physical_state[1] == physical_state[3]:
		bad_labels[mdp_state] = 1
	else:
		bad_labels[mdp_state] = 0
print("generated bad labels")
good_labels = {}
for mdp_state in mdp_states:
	good_labels[mdp_state] = 0
	if bad_labels[mdp_state]:
		continue
	val = mdp_state[0]
	physical_state = []
	for k in range(1,6):
		v = int(val % 10) 
		physical_state.append(v)
		val = np.floor(val / 10)
	physical_state.reverse()
	#print(physical_state)	

	if physical_state[0] == xmax and physical_state[1] == xmax:
		good_labels[mdp_state] = 1
print("generated good labels")
"""
probabilities
"""
prob = np.load('random_generated/random_mdp_prob_%d_td.npy' % max_td, allow_pickle=True).item()
V = {tuple(mdp_state):1 for mdp_state in mdp_states}
#Q = {tuple(state_action_pair):1 for state_action_pair in state_action_pairs}
V = ValueIteration(V, mdp, mdp_states, bad_labels, good_labels, num_actions, prob) 
print("here")
np.save('random_generated/state_values_%d_td' % max_td, V)
#np.save('random_generated/state_action_values_%d_td' % max_td, Q)
