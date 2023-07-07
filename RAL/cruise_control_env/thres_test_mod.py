import os 
import pandas as pd
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F

from env import DelayContinuousCruiseCtrlEnv
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


def test(policy, test_env, test_ep_random_seeds):         
	last_distance_remaining_list = []

	num_test_episodes = test_ep_random_seeds.shape[0]
	for test_ep_idx in range(num_test_episodes):
		print("test episode number : %d" % test_ep_idx)
		state = test_env.reset(seed=test_ep_random_seeds[test_ep_idx])
		
		done = False 
		last_distance_remaining = state[0]
		while not done: 
			state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
			action = policy(state_tensor)
			action = action.detach().numpy().squeeze(0)
			state, _, done, info = test_env.step(action)
			last_distance_remaining = info['dis_rem']
			
		print("end distance : %f" % last_distance_remaining)
		last_distance_remaining_list.append(last_distance_remaining) 

	last_distance_remaining_arr = np.array(last_distance_remaining_list)
	return last_distance_remaining_arr
 
if __name__ == "__main__":
	num_test_episodes = 100 
	log_dir = 'tmp/sac/basic_sac'
	random_seed = 0
	max_time_delay = 3

	np.random.seed(random_seed)
	policy = ActorNetwork(2, 1, 0.5)
	policy_path = os.path.join(log_dir, 'own_sac_best_policy.pt')
	policy.load_state_dict(torch.load(os.path.join(policy_path)))

	delta_list = [0.0, 0.2, 0.5, 0.8, 0.9]
	# delta_list = [0.0]
	plotting_list = []

	basic_mdp = BasicMDP()
	
	print("for constant delay")

	for time_delay in range(max_time_delay+1):
		print("############################################################")
		print("time delay : %d" % time_delay) 

		for delta in delta_list:
			print("---------------------------------------------------------------")
			print("safety probability : %f" % delta)		

			test_env = DelayContinuousCruiseCtrlEnv(basic_mdp=basic_mdp, time_delay=time_delay, constant_delay=True, delta=delta, print_=False)
			random_seeds = np.random.choice(10000, size=(num_test_episodes,))
			last_dist_rem_arr_per_td = test(policy, test_env, random_seeds)  
			
			plotting_list = plotting_list + list(zip([delta]*num_test_episodes, \
								['cd = %d' % time_delay]*num_test_episodes, \
								last_dist_rem_arr_per_td))


	print("for random delay")
 
	for delta in delta_list:
		print("---------------------------------------------------------------")
		print("safety probability : %f" % delta)		

		test_env = DelayContinuousCruiseCtrlEnv(basic_mdp=basic_mdp, time_delay=max_time_delay, constant_delay=False, delta=delta, print_=False)
		random_seeds = np.random.choice(10000, size=(num_test_episodes,))
		last_dist_rem_arr_per_td = test(policy, test_env, random_seeds)  
			
		plotting_list = plotting_list + list(zip([delta]*num_test_episodes, \
								['rd = %d' % max_time_delay]*num_test_episodes, \
								last_dist_rem_arr_per_td))

	df = pd.DataFrame (plotting_list, columns = ['Threshold', 'time delay', 'end distance'])
	print(df)
	df.to_pickle("quantitative.pkl") 
		 


	