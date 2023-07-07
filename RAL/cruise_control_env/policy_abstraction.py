import os
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F

from no_td_mdp import BasicMDP

class ActorNetwork(nn.Module):

	def __init__(self, state_dim, action_dim, max_action):
		super(ActorNetwork, self).__init__() 

		self.state_dim = state_dim 
		self.action_dim = action_dim 

		self.fc1 = nn.Linear(self.state_dim, 256)
		self.fc2 = nn.Linear(256, 256)
		self.fc_mu_layer = nn.Linear(256, self.action_dim)
		self.fc_log_std_layer = nn.Linear(256, self.action_dim)

		self.max_action = max_action

	def forward(self, state):        
		out = self.fc1(state)
		out = F.relu(out)
		out = self.fc2(out)
		out = F.relu(out) 

		mu = self.fc_mu_layer(out)
		action = torch.tanh(mu) * self.max_action

		return action

policy_path = 'tmp/sac/basic_sac/own_sac_best_policy.pt'
policy = ActorNetwork(2, 1, 0.5)
policy.load_state_dict(torch.load(os.path.join(policy_path))) 

basic_mdp = BasicMDP() 

abs_dnn_policy = {}
for rel_dist_idx in range(basic_mdp.num_rel_dist_indices):
	for rel_vel_idx in range(basic_mdp.num_rel_vel_indices):
		state_rel_dist = basic_mdp.rel_dist_list[rel_dist_idx]
		state_rel_vel = basic_mdp.rel_vel_list[rel_vel_idx]
		physical_state = (rel_dist_idx, rel_vel_idx) # discrete system state
		continuous_state = (state_rel_dist, state_rel_vel) # actual continuous system state
		state_id = basic_mdp.convert_physical_state_to_int(physical_state)
		state = (state_id,)
		print("physical state: ", physical_state, "continuous state: ", continuous_state)

		continuous_state_tensor = torch.tensor(continuous_state, dtype=torch.float32).unsqueeze(0)
		action = policy(continuous_state_tensor)
		action = action.detach().numpy().squeeze(0)[0]
		abs_dnn_policy[state] = action

print(abs_dnn_policy)

policy_path = 'abstracted_dnn_policy.npy'
np.save(policy_path, abs_dnn_policy)