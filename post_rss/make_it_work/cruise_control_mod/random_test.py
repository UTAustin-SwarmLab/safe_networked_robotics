import sys
import os 
import numpy as np
import torch
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sac import SoftActorCritic 

 
if __name__ == "__main__":
 
	parser = argparse.ArgumentParser()

	parser.add_argument("--lr", type=float, default=3e-4)
	parser.add_argument("--buffer_size", type=int, default=1000000)
	parser.add_argument("--update_every", type=int, default=50)
	parser.add_argument("--batch_size", type=int, default=256)
	parser.add_argument("--gamma", type=float, default=0.99)
	parser.add_argument("--alpha", type=float, default=0.02)
	parser.add_argument("--polyak", type=float, default=0.995)
	parser.add_argument("--total_steps", type=int, default=10000)
	parser.add_argument("--modify_alpha_after", type=int, default=100000)
	parser.add_argument("--num_test_episodes", type=int, default=1)
	
	parser.add_argument("--log_dir", type=str, default='tmp/sac/basic_sac') 
	parser.add_argument('--train', default=False, action='store_true')
	parser.add_argument("--random_seed", type=int, default=0)

	parser.add_argument("--max_time_delay", type=int, default=2)
	parser.add_argument("--delta", type=float, default=0.95)
		
	args = parser.parse_args()   

			
	# creating random seeds for num_test_episodes epsiodes
	np.random.seed(args.random_seed)
	random_seeds = np.random.choice(10000, size=(args.num_test_episodes,))

	min_dist_rem_dict = {}

	log_dir = args.log_dir


	"""
	random delay results
	"""
	args.constant_delay = True  
	soft_actor_critic = SoftActorCritic(args)
	policy_path = os.path.join(args.log_dir, 'own_sac_best_policy.pt')

	_, _, min_dist_rem_arr_per_td = soft_actor_critic.test(policy_path, args, random_seeds)
