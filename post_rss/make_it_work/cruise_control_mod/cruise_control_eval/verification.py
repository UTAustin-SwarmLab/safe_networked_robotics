import os
import sys
import copy
import numpy as np 
import torch
from itertools import product

log_dir = 'constant_generated'
td = int(sys.argv[1]) 
mdp_loc = os.path.join(log_dir, 'transition_probabilities_%d_td.pt' % td)
mdp = torch.load(mdp_loc)

mdp_size = mdp.coalesce().size()
num_states = mdp_size[1]
num_actions = int(mdp_size[0]/num_states)
size = torch.Size([num_states*num_actions, num_states])

unsafe_states_loc = os.path.join(log_dir, 'unsafe_states_%d_td.pt' % td)
unsafe_states = torch.load(unsafe_states_loc)

initial_states_loc = os.path.join(log_dir, 'initial_states_%d_td.pt' % td)
initial_states = torch.load(initial_states_loc)
initial_states_indices = torch.where(initial_states)[0]
num_initial_states = initial_states_indices.shape[0]

def efficient_min_reachability(mdp, unsafe_states, eps=1e-6):
    mdp_size = mdp.coalesce().size()
    num_states = mdp_size[1]
    num_actions = int(mdp_size[0]/num_states)
    # print(num_states, num_actions)

    pmin_state_values = torch.zeros((num_states, 1)) # initialization
    pmin_state_action_values = torch.zeros((num_states, num_actions))
    old_pmin_state_values = copy.deepcopy(pmin_state_values)
    max_err = 1000

    while max_err > eps:
        pmin_state_action_values = torch.sparse.mm(mdp, pmin_state_values).view(num_states, num_actions)
        pmin_state_values = torch.min(pmin_state_action_values, dim=1).values 
        pmin_state_values[torch.where(unsafe_states)[0]] = 1.0
        pmin_state_values = pmin_state_values.view(-1,1)
        max_err = torch.max(torch.abs(pmin_state_values - old_pmin_state_values))
        old_pmin_state_values = pmin_state_values
        print("the current max error is ", max_err)

    return pmin_state_values.squeeze(dim=1), pmin_state_action_values

def efficient_max_reachability(mdp, unsafe_states, eps=1e-6):
    mdp_size = mdp.coalesce().size()
    num_states = mdp_size[1]
    num_actions = int(mdp_size[0]/num_states)
    # print(num_states, num_actions)

    pmax_state_values = torch.zeros((num_states, 1)) # initialization
    pmax_state_action_values = torch.zeros((num_states, num_actions))
    old_pmax_state_values = copy.deepcopy(pmax_state_values)
    max_err = 1000

    while max_err > eps:
        pmax_state_action_values = torch.sparse.mm(mdp, pmax_state_values).view(num_states, num_actions)
        pmax_state_values = torch.max(pmax_state_action_values, dim=1).values 
        pmax_state_values[torch.where(unsafe_states)[0]] = 1.0
        pmax_state_values = pmax_state_values.view(-1,1)
        max_err = torch.max(torch.abs(pmax_state_values - old_pmax_state_values))
        old_pmax_state_values = pmax_state_values
        # print("the current max error is ", max_err)

    return pmax_state_values.squeeze(dim=1), pmax_state_action_values

def ValueIterationForMinSafety(Vmin, Qmin, threshold, mdp, labels, max_iter=10000, delta=1e-6):

    V = torch.zeros((num_states,)) # initialization
    Q = torch.zeros((num_states, num_actions))

    # Start value iteration
    for i in range(max_iter):
        max_diff = 0
    
        for mdp_state in range(num_states):
            old_state_value = V[mdp_state].item()
            print("-------------------")
            print(old_state_value)

            if labels[mdp_state] == 1:
                V[mdp_state] = 1.0 
                continue    

            state_action_values_list = []    
            for a in range(num_actions):
                state_action_id = mdp_state*num_actions+a
                # print(mdp[state_action_id].coalesce().indices()[0])
                if not (Qmin[mdp_state][a] <= threshold or Qmin[mdp_state][a] == Vmin[mdp_state]):
                    continue
                next_states = mdp[state_action_id].coalesce().indices()[0]
                next_state_values = [V[next_state] for next_state in next_states]
                state_action_value = sum(next_state_values) / len(next_state_values)
                state_action_values_list.append(state_action_value)
                Q[mdp_state][a] = state_action_value
            
            print(state_action_values_list)
            state_value = max(state_action_values_list)
            V[mdp_state] = state_value
            print(old_state_value, state_value)
            print(state_value)
            diff = abs(old_state_value - state_value.item())
            max_diff = max(max_diff, diff)

        print("max error : ", max_diff)
        
        if max_diff < delta:
            break 
        

    return V, Q

Vmin, Qmin = efficient_min_reachability(mdp, unsafe_states)

safest_actions = torch.zeros_like(Qmin)
mask = Qmin == Vmin.view(-1, 1)
safest_actions[mask] = True


delta_min = 0.0 
delta_max = 1.0 
guarantee = False 
threshold = 0.0

while not guarantee:
    delta = (delta_min+delta_max)/2
    # print(delta)
    delta_safe_actions = Qmin <= 0.000244#delta
    allowed_actions = torch.logical_or(safest_actions, delta_safe_actions)
    allowed_actions = allowed_actions.view(-1)
    allowed_sa_pairs = torch.where(allowed_actions)[0]
    indices = mdp.coalesce().indices()
    values = mdp.coalesce().values()
    allowed_indices = torch.where(torch.isin(indices[0,:], allowed_sa_pairs))[0]
    indices = indices[:, allowed_indices]
    values = values[allowed_indices]
    delta_mdp = torch.sparse_coo_tensor(indices, values, size)

    Vmax, Qmax = efficient_max_reachability(delta_mdp, unsafe_states)
    Vmax_init = Vmax[initial_states_indices]
    Vmax_init = torch.sum(Vmax_init) / num_initial_states
    print(Vmax_init)
    Vmax, Qmax = ValueIterationForMinSafety(Vmin, Qmin, 0.000244, mdp, unsafe_states)
    Vmax_init = Vmax[initial_states_indices]
    Vmax_init = torch.sum(Vmax_init) / num_initial_states

    # if Vmax_init < threshold:
    #     guarantee = True
    # else:    
    #     delta_max = delta 
    print(Vmax_init)
    break
    
