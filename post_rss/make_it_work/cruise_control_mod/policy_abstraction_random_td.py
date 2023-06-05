import os
import sys
import numpy as np
import torch 
import itertools

no_td_abs_dnn_policy = np.load('constant_generated/abstracted_dnn_policy_0_td.npy', allow_pickle=True).item()
num_physical_states = len(list(no_td_abs_dnn_policy.keys()))

ego_acc_values = [-0.5, -0.25, 0.0, 0.25]
num_actions = len(ego_acc_values)

abstract_actions = list(np.arange(num_actions+1))
num_abstract_actions = len(abstract_actions)

max_td = int(sys.argv[1])
ustates = list(itertools.product(abstract_actions, repeat=max_td))
num_ustates = len(ustates)
invalid_action = 0

def decompose_int_to_state(st_idx):
	num_state_features = 2 + max_td 
	divide_by = [max_td+1,] + [num_abstract_actions,]*max_td + [num_physical_states,]
	state = []
	for feat_idx in range(num_state_features):
		rem = st_idx % divide_by[feat_idx]
		st_idx = st_idx // divide_by[feat_idx]
		state.append(rem)
	state.reverse()

	return state

zero_td_mdp = np.load('constant_generated/mdp_0_td.npy', allow_pickle=True).item()
bad_labels = np.load('random_generated/random_mdp_unsafe_%d_td.npy' % max_td, allow_pickle=True)

num_states = bad_labels.shape[0]

abs_dnn_policy = {}
for st_idx in range(num_states):
	state = decompose_int_to_state(st_idx)
	# if state != [219,0,0,0,0]:
	# 	continue
	print(state)

	physical_state = state[0]
	abs_dnn_policy[st_idx] = no_td_abs_dnn_policy[physical_state]

policy_path = 'random_generated/abstracted_dnn_policy_%d_td.npy' % max_td
np.save(policy_path, abs_dnn_policy)