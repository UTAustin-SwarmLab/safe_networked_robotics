import sys
import os 
import numpy as np
import torch
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models import ActorNetwork
from random_td_env import RandTdContinuousCruiseCtrlEnv

import seaborn as sns
import seaborn
import pandas as pd

FONT_SIZE = 20
FONT_SIZE = 18 
sns.set_color_codes() 
seaborn.set()

plt.rc('text', usetex=False)
font = {'family' : 'normal', 
		'weight' : 'bold',
		'size'   : FONT_SIZE}
plt.rc('font', **font)
# plt.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rcParams["text.latex.preamble"].join([
		r"\usepackage{dashbox}",              
		r"\setmainfont{xcolor}",
])
LEGEND_FONT_SIZE = 10
XTICK_LABEL_SIZE = 10
YTICK_LABEL_SIZE = 10

plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["font.weight"] = "bold"
# matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
import matplotlib.pylab as pylab
params = {'legend.fontsize': LEGEND_FONT_SIZE,
		 'axes.labelsize': FONT_SIZE,
		 'axes.titlesize': FONT_SIZE,
		 'xtick.labelsize': XTICK_LABEL_SIZE,
		 'ytick.labelsize': YTICK_LABEL_SIZE,
		 'figure.autolayout': True}
pylab.rcParams.update(params)
plt.rcParams["axes.labelweight"] = "bold"

def test(policy, test_env, test_ep_random_seeds):        
	last_distance_remaining_list = []

	num_test_episodes = test_ep_random_seeds.shape[0]
	for test_ep_idx in range(num_test_episodes):
		print("test episode number : %d" % test_ep_idx)
		state = test_env.reset(seed=test_ep_random_seeds[test_ep_idx])
		
		done = False 
		# print('sai writes "highly unstructured code" - Sai')
		min_dist_rem = 100
		last_distance_remaining = state[0]
		while not done: 
			state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
			action, _ = policy(state_tensor, deterministic=True)
			action = action.detach().numpy().squeeze(0)
			state, reward, done, info = test_env.step(action)
			last_distance_remaining = info['dis_rem']
			
		print("end distance : %f" % last_distance_remaining)
		last_distance_remaining_list.append(last_distance_remaining)

	last_distance_remaining_arr = np.array(last_distance_remaining_list)
	return last_distance_remaining_arr
 
if __name__ == "__main__":
 
	parser = argparse.ArgumentParser()

	parser.add_argument("--num_test_episodes", type=int, default=100)	
	parser.add_argument("--log_dir", type=str, default='tmp/sac/basic_sac')
	parser.add_argument('--train', default=False, action='store_true')
	parser.add_argument("--random_seed", type=int, default=0)
	parser.add_argument("--max_time_delay", type=int, default=3)
		
	args = parser.parse_args()   

	np.random.seed(args.random_seed)
	device = torch.device("cuda")
	dummy_env = RandTdContinuousCruiseCtrlEnv(time_delay=args.max_time_delay, train=False, constant_td=True, delta=0.0)
	policy = ActorNetwork(dummy_env.observation_space, dummy_env.action_space, dummy_env.action_space.high[0], device)
	policy_path = os.path.join(args.log_dir, 'own_sac_best_policy.pt')
	policy.load_state_dict(torch.load(os.path.join(policy_path)))

	thres_list = [0.0, 0.2, 0.5, 0.8, 0.9, 0.95]
	# thres_list = [0.0]
	plotting_list = []
	
	print("for constant delay")

	for time_delay in range(0,args.max_time_delay+1):
		print("############################################################")
		print("time delay : %d" % time_delay) 
		env_time_delay = time_delay		

		for thres in thres_list:
			print("---------------------------------------------------------------")
			print("threshold : %f" % thres)		
			args.delta = thres

			test_env = RandTdContinuousCruiseCtrlEnv(time_delay=env_time_delay, train=False, constant_td=True, delta=args.delta)
			random_seeds = np.random.choice(10000, size=(args.num_test_episodes,))
			last_dist_rem_arr_per_td = test(policy, test_env, random_seeds)  
			
			plotting_list = plotting_list + list(zip([thres]*args.num_test_episodes, \
								['cd = %d' % env_time_delay]*args.num_test_episodes, \
								last_dist_rem_arr_per_td))


	print("for random delay")
 
	for thres in thres_list:
		print("---------------------------------------------------------------")
		print("threshold : %f" % thres)		
		args.delta = thres

		test_env = RandTdContinuousCruiseCtrlEnv(time_delay=args.max_time_delay, train=False, constant_td=False, delta=args.delta)
		random_seeds = np.random.choice(10000, size=(args.num_test_episodes,))
		last_dist_rem_arr_per_td = test(policy, test_env, random_seeds)  
			
		plotting_list = plotting_list + list(zip([thres]*args.num_test_episodes, \
								['rd = %d' % args.max_time_delay]*args.num_test_episodes, \
								last_dist_rem_arr_per_td))
		



	df = pd.DataFrame (plotting_list, columns = ['Threshold', 'time delay', 'end distance'])
	df.to_pickle("quantitative.pkl")

	fig = plt.figure()
	plot = sns.boxplot(x='time delay', y='end distance', data=df, order = None, hue = 'Threshold')
	
	plt.ylim(0, 100)
	plt.savefig('end_dis_bp.png')
	plt.clf()
	plt.close()


	