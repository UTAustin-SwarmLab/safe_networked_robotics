import os
import sys
import numpy as np
import torch 
import itertools
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

LOG_STD_MAX = 2
LOG_STD_MIN = -20 

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

	def forward(self, state, deterministic=False):        
		out = self.fc1(state)
		out = F.relu(out)
		out = self.fc2(out)
		out = F.relu(out)

		mu = self.fc_mu_layer(out)
		log_std = self.fc_log_std_layer(out)
		log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
		std = torch.exp(log_std)
  
		action_distribution = Normal(mu, std)
		if deterministic:
			sample_action = mu
		else:
			sample_action = action_distribution.rsample() 

		# logprob_action = action_distribution.log_prob(sample_action).sum(axis=-1)
		# logprob_action -= (2*(np.log(2) - sample_action - F.softplus(-2*sample_action))).sum(axis=1)
		sample_action = mu
		action = torch.tanh(sample_action) * self.max_action

		return action

"""
### relative distance abstraction
"""
min_rel_dist = 5 
max_rel_dist = 25 
del_rel_dist = 1.0  

rel_dist_list = []
for i in range(int((max_rel_dist - min_rel_dist) / del_rel_dist + 1)):
	rel_dist_list.append(min_rel_dist + i * del_rel_dist)

"""
### relative velocity abstraction
"""
min_rel_vel = -5
max_rel_vel = 5
del_vel = 0.5 

rel_vel_list = [] 
for i in range(int((max_rel_vel - min_rel_vel) / del_vel)+1):
	rel_vel_list.append(min_rel_vel + i * del_vel)

num_states = len(rel_dist_list)*len(rel_vel_list)

ego_acc_values = [-0.5, -0.25, 0.0, 0.25]
num_actions = len(ego_acc_values)

policy_path = 'tmp/sac/basic_sac/own_sac_best_policy.pt'
policy = ActorNetwork(2, 1, 1.0)
policy.load_state_dict(torch.load(os.path.join(policy_path))) 

td = int(sys.argv[1])
ustates = list(itertools.product(list(range(len(ego_acc_values))), repeat=td))
num_ustates = len(ustates)

def convert_state_to_int(state):
	increments = [num_actions**k for k in range(td)]
	increments.reverse()
	increments = [num_ustates,] + increments
	return np.sum(np.multiply(list(state), increments))

abs_dnn_policy = {}

for state_rel_dist in rel_dist_list:
	rel_dist_idx = rel_dist_list.index(state_rel_dist)
	for state_rel_vel in rel_vel_list:
		rel_vel_idx = rel_vel_list.index(state_rel_vel)
		physical_state = (rel_dist_idx*len(rel_vel_list)+rel_vel_idx,)
		print("physical state id : ", physical_state)
				
		state = [state_rel_dist, state_rel_vel]
		state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
		action = policy(state_tensor, deterministic=True)
		action = action.detach().numpy().squeeze(0)[0]
		if action <= ego_acc_values[-1]:	
			action_idx = np.where(ego_acc_values >= action)[0][0]
		else:
			action_idx = len(ego_acc_values) - 1
		
		if td == 0:
			state_id = physical_state[0] 
			abs_dnn_policy[state_id] = action_idx
		else:
			for ustate in ustates:
				full_state = physical_state + ustate
				state_id = convert_state_to_int(full_state)
				print("full state id : ", state_id)
				abs_dnn_policy[state_id] = action_idx

		abs_dnn_policy[state_id] = action_idx

print(abs_dnn_policy)

policy_path = 'constant_generated/abstracted_dnn_policy_%d_td.npy' % td
np.save(policy_path, abs_dnn_policy)