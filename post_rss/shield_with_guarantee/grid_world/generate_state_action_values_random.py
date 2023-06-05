import sys 
sys.path.remove('/usr/lib/python3/dist-packages')
import numpy as np 

max_td = 3
num_actions = 5

mdp = np.load('random_generated/random_mdp_%d_td.npy' % max_td, allow_pickle=True).item()
prob = np.load('random_generated/random_mdp_prob_%d_td.npy' % max_td, allow_pickle=True).item()
state_values = np.load('random_generated/state_values_%d_td.npy' % max_td, allow_pickle=True).item()

state_action_pairs = list(mdp.keys())

state_action_values = {}
print("here")
for state_action_pair in state_action_pairs:
	if state_action_pair[0][0] % 2 == 1 and state_action_pair[0][-1] == 0:
		print(state_action_pair)
		next_states = mdp[state_action_pair]
		next_prob = np.array(prob[state_action_pair])
		next_state_values = np.array([state_values[next_state] for next_state in next_states])
		state_action_value = np.sum(next_prob*next_state_values)

		state_action_values[state_action_pair] = state_action_value

print("here")
np.save('random_generated/state_action_values_%d_td' % max_td, state_action_values)
