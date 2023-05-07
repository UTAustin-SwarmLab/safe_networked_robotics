"""
note that the entire code is in terms of reachability and not safety - the logics will be just opposite
"""

import numpy as np 
import torch 
import copy

device = torch.device("cuda:0")

transition = np.load('constant_generated/transition_arr/0_td.npy', allow_pickle=True)
transition = torch.tensor(transition, dtype=torch.float32).to(device)
num_states = transition.shape[0]
num_actions = transition.shape[1]

bad_state_values = torch.zeros((num_states,)).to(device)
bad_state_values[:132] = 1.0

req_safety_probaility = 0.9

"""
performing value iteration to compute the pmax safety values
"""

eps = 1e-16
state_values = torch.zeros((num_states,), dtype=torch.float32).to(device) # initialization
old_state_values = copy.deepcopy(state_values)
max_err = 1000
while max_err > eps:
    state_values = torch.min(torch.matmul(transition, state_values), dim=1).values
    state_values[:132] = 1.0
    max_err = torch.max(state_values - old_state_values)
    old_state_values = state_values

delta_min = 0.0 
delta_max = 1.0 
delta = 0.5 

state_action_values = torch.matmul(transition, state_values)
state_values = torch.min(state_action_values, dim=1).values

safest_actions = torch.zeros_like(state_action_values)
delta_safe_actions = torch.zeros_like(state_action_values)

# always the maximally safe action is part of the shield
min_vals, _ = torch.min(state_action_values, dim=1)
mask = state_action_values == min_vals.view(-1, 1)
safest_actions[mask] = True

# now let us add the actions that correspond state action value greater than delta
delta_safe_actions = state_action_values <= delta

allowed_actions = torch.logical_or(safest_actions, delta_safe_actions)

# find pmin safety values 


# print(state_action_values[-22])
# print(state_values[-22])
# print(safest_actions[-22])
# print(safest_actions[-22].bool())
# print(delta_safe_actions[-22])
# print(allowed_actions[-22])


# state_values = state_values.detach().cpu().numpy()
# import matplotlib.pyplot as plt

# num_rel_dist_states = 27
# num_rel_vel_states = 22

# vis_array = np.zeros((num_rel_dist_states, num_rel_vel_states))

# for rel_dist_val in range(num_rel_dist_states):
#     for rel_vel_val in range(num_rel_vel_states):
#         physical_state = rel_dist_val * num_rel_vel_states + rel_vel_val
#         #print(physical_state) 
#         vis_array[rel_dist_val, rel_vel_val] = state_values[physical_state]


# # stay_control_vector
# plt.imshow(vis_array, cmap=plt.cm.Blues, extent=[-10,10,25,0])
# plt.title('Pmax values for the stay control vector', size=12)
# plt.ylabel('Relative Distance (m)', size=12)
# plt.xlabel('Relative velocity (m/s)', size=12)
# plt.legend()
# plt.savefig('constant_generated/stay_control_vector_%d_td.png' % 0)
# plt.clf()
# plt.cla()
# plt.close()

# print(vis_array)
