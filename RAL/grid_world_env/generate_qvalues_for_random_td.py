import numpy as np 

max_td = 3
vmax_loc = 'random_generated/Vmax_values_%d_td.npy' % max_td
Vmax = np.load(vmax_loc, allow_pickle=True).item()

random_td_mdp = np.load('random_generated/mdp_transitions_%d_td.npy' % max_td, allow_pickle=True).item()
random_td_mdp_prob = np.load('random_generated/mdp_probabilities_%d_td.npy' % max_td, allow_pickle=True).item()

mdp_states = list(Vmax.keys())
# print(mdp_states)

actions_dict = np.load('actions.npy', allow_pickle=True).item()
actions_list = list(actions_dict.keys())
num_actions = len(actions_list)

Qmax_values = {}
for state in mdp_states:
    if state[-1] != 0:
        continue 
    print(state)
    for action in range(num_actions):
        state_action_pair = (state, action)
        next_states = random_td_mdp[state_action_pair]
        next_states_prob = random_td_mdp_prob[state_action_pair]
        next_states_values = [Vmax[next_states[ii]] for ii in range(len(next_states))]
        state_action_value = sum([nsv*nsp for nsv,nsp in zip(next_states_values, next_states_prob)])
        Qmax_values[state_action_pair] = state_action_value

qmax_loc = 'random_generated/Qmax_values_%d_td.npy' % max_td
np.save(qmax_loc, Qmax_values)
