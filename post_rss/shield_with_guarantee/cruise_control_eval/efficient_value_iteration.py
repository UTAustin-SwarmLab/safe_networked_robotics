"""
note that the entire code is in terms of reachability and not safety - the logics will be just opposite
"""
import sys
import os
import numpy as np 
import torch 
import copy

device = torch.device("cuda:0")

td = int(sys.argv[1])
transition = torch.load('constant_generated/%d_td/transition' % td)
num_states = transition.shape[1]
num_actions = int(transition.shape[0]/num_states)

inc = num_actions**td

req_safety_probability = 1 - float(sys.argv[2])

"""
performing value iteration to compute the pmax safety values
"""

eps = 1e-6
pmin_state_values = torch.zeros((num_states, 1), dtype=torch.float32).to(device) # initialization
old_pmin_state_values = copy.deepcopy(pmin_state_values)
max_err = 1000
while max_err > eps:
    pmin_state_action_values = torch.sparse.mm(transition, pmin_state_values).view(num_states, num_actions)
    print("here")
    pmin_state_values = torch.min(pmin_state_action_values, dim=1).values 
    pmin_state_values[:int(132*inc)] = 1.0
    max_err = torch.max(pmin_state_values - old_pmin_state_values)
    old_pmin_state_values = pmin_state_values

pmin_state_action_values = torch.sparse.mm(transition, pmin_state_values).view(num_states, num_actions)
pmin_state_values = torch.min(pmin_state_action_values, dim=1).values
safest_actions = torch.zeros_like(pmin_state_action_values)

# always the maximally safe action is part of the shield
min_vals, _ = torch.min(pmin_state_action_values, dim=1)
mask = pmin_state_action_values == min_vals.view(-1, 1)
safest_actions[mask] = True

print(pmin_state_values[313])

# """
# performinng binary search due to the non-decreasing property
# """

# delta_min = 0.0 
# delta_max = 1.0 
# delta = (delta_min + delta_max)/2
# guarantee = False 
# old_safety_probability = 1.0
# pmax_state_values = torch.zeros((num_states,), dtype=torch.float32).to(device) # initialization
# pmax_state_action_values = torch.zeros_like(pmin_state_action_values)

# while not guarantee:
#     delta = (delta_min + delta_max)/2 # update delta

#     # now let us add the actions that correspond state action value greater than delta to the set of safe actions
#     delta_safe_actions = pmin_state_action_values <= delta
#     allowed_actions = torch.logical_or(safest_actions, delta_safe_actions) # find the safe actions for a given delta

#     # find pmin safety values 

#     max_err = 1000
#     pmax_state_values = torch.zeros((num_states,), dtype=torch.float32).to(device) # initialization
#     old_pmax_state_values = copy.deepcopy(pmax_state_values)
#     while max_err > eps:
#         pmax_state_action_values = torch.matmul(transition, pmax_state_values) - torch.logical_not(allowed_actions) * 1000
#         pmax_state_values = torch.max(pmax_state_action_values, dim=1).values 
#         pmax_state_values[:int(132*inc)] = 1.0 
#         max_err = torch.max(pmax_state_values - old_pmax_state_values)
#         old_pmax_state_values = pmax_state_values

#     pmax_state_action_values = torch.matmul(transition, pmax_state_values)
#     pmax_state_values = torch.max(pmax_state_action_values, dim=1).values

#     current_safety_probability = pmax_state_values[-int(11*inc)]
#     if current_safety_probability > req_safety_probability:
#         delta_max = delta 
#     else:
#         delta_min = delta

#     print("the current safety probability is %.6f and the current value of delta is %.6f" % (current_safety_probability, delta))

#     if abs(current_safety_probability - old_safety_probability) < eps and current_safety_probability < req_safety_probability:
#         break
    
#     old_safety_probability = current_safety_probability

# shielded_actions = pmin_state_action_values <= delta
# allowed_actions = torch.logical_or(safest_actions, shielded_actions)
# allowed_actions = allowed_actions.detach().cpu().numpy()
# # print(allowed_actions)
# # print(pmin_state_action_values[1])

# save_dir = 'constant_generated/%d_td/' % td
# os.makedirs(save_dir, exist_ok=True)
# save_loc = os.path.join(save_dir, 'shield_%s_prob.npy' % sys.argv[2])
# np.save(save_loc, allowed_actions)

