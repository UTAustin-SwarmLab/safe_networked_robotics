import sys
sys.path.remove('/usr/lib/python3/dist-packages')

import os 
import numpy as np
import torch
import argparse


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
    parser.add_argument("--total_steps", type=int, default=1000000)
    parser.add_argument("--modify_alpha_after", type=int, default=10000)
    parser.add_argument("--num_test_episodes", type=int, default=1)
    
    parser.add_argument("--log_dir", type=str, default='tmp/sac/basic_sac')
    parser.add_argument('--train', default=True, action='store_true')
    parser.add_argument("--random_seed", type=int, default=0)

    parser.add_argument("--time_delay", type=int, default=1)
    parser.add_argument("--delta", type=float, default=0.99)
    
    
    args = parser.parse_args()  
    
    #args.log_dir = os.path.join(args.log_dir, str(args.time_delay), 'shield_' + str(args.delta))
    #os.makedirs(args.log_dir, exist_ok=True)
    #print(args.log_dir)

    # creating an instance for soft actor critic, the env is created along with the instance
    print(args)
    
    soft_actor_critic = SoftActorCritic(args)
    #soft_actor_critic.learn(args)

    policy_path = os.path.join('tmp/sac/basic_sac', 'own_sac_best_policy.pt')
    np.random.seed(args.random_seed)
    random_seeds = np.random.choice(10000, size=(args.num_test_episodes,))
    episode_returns_arr_per_td, last_dist_rem_arr_per_td, min_dist_rem_arr_per_td = soft_actor_critic.test(policy_path, args, random_seeds)  

    # training the soft actor critic 
    """
    if args.train:
        soft_actor_critic.learn(args)
    else:
        policy_path = 'tmp/sac/basic_sac/own_sac_last_policy.pt'
        soft_actor_critic.test(policy_path, args)
    """ 