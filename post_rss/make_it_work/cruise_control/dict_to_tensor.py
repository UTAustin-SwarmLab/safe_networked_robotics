import os
import numpy as np

mdp_loc = 'constant_generated/mdp_0_td.npy'
mdp = np.load(mdp_loc, allow_pickle=True).item()

num_actions = 5

mdp_keys = list(mdp.keys())
num_state_action_pairs = len(mdp_keys)
print(num_state_action_pairs)

num_states = int(num_state_action_pairs/num_actions)
print(num_states)

transition = np.zeros((num_states, num_actions, num_states))

for key in mdp_keys:
    state = key[0][0]
    action = key[1]
    next_states = mdp[key]
    next_states = [state[0] for state in next_states]
    transition[state][action][next_states] = 1/len(next_states)

os.makedirs('constant_generated/transition_arr', exist_ok=True)
np.save('constant_generated/transition_arr/0_td', transition)
