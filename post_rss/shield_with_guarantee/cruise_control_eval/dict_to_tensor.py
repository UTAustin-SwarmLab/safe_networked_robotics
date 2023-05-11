import os
import sys
import torch
import numpy as np

td = int(sys.argv[1])

mdp_loc = 'constant_generated/mdp_%s_td.npy' % sys.argv[1]
mdp = np.load(mdp_loc, allow_pickle=True).item()

num_actions = 5 

mdp_keys = list(mdp.keys())
num_state_action_pairs = len(mdp_keys)
print(num_state_action_pairs)

num_states = int(num_state_action_pairs/num_actions)
print(num_states)

device = torch.device("cuda:0")
transition = torch.zeros((num_states*num_actions, num_states), dtype=torch.float16).to(device)

for i, key in enumerate(mdp_keys):
    next_states = mdp[key]
    transition[i][next_states] = 1/len(next_states)

transition = transition.to_sparse()
print(transition)

os.makedirs('constant_generated/%d_td' % td, exist_ok=True)
torch.save(transition, 'constant_generated/%d_td/transition' % td)
