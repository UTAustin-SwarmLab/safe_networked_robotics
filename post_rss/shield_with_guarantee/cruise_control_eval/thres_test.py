import sys
sys.path.remove('/usr/lib/python3/dist-packages')
import os 
import numpy as np
import torch
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sac import SoftActorCritic 

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
plt.rcParams['text.latex.preamble'] = [r'\boldmath']

LEGEND_FONT_SIZE = 14
XTICK_LABEL_SIZE = 14
YTICK_LABEL_SIZE = 14

plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["font.weight"] = "bold"
#matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
import matplotlib.pylab as pylab
params = {'legend.fontsize': LEGEND_FONT_SIZE,
		 'axes.labelsize': FONT_SIZE,
		 'axes.titlesize': FONT_SIZE,
		 'xtick.labelsize': XTICK_LABEL_SIZE,
		 'ytick.labelsize': YTICK_LABEL_SIZE,
		 'figure.autolayout': True}
pylab.rcParams.update(params)
plt.rcParams["axes.labelweight"] = "bold"
 
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
	parser.add_argument("--num_test_episodes", type=int, default=300)	
	parser.add_argument("--log_dir", type=str, default='tmp/sac/basic_sac')
	parser.add_argument('--train', default=False, action='store_true')
	parser.add_argument("--random_seed", type=int, default=0)

	parser.add_argument("--num_time_delays", type=int, default=4)
	parser.add_argument("--delta", type=float, default=0.99)
		
	args = parser.parse_args()   

	thres_list = [0.0, 0.5, 0.8, 0.9, 0.95]
	plotting_list = []
	failure_list = []
	pmax_plotting_list = []

	for thres in thres_list:
		print("---------------------------------------------------------------")
		print("threshold : %d", thres)		
		args.delta = thres
		# creating random seeds for num_test_episodes epsiodes
		np.random.seed(args.random_seed)
		random_seeds = np.random.choice(10000, size=(args.num_test_episodes,))

		episode_returns_arr = []
		min_dist_rem_arr = []
		num_failures_arr = []
		end_dist_rem_arr = []

		log_dir = args.log_dir
		for time_delay in range(args.num_time_delays):
			print("############################################################")
			print("time delay : %d", time_delay)
			args.constant_delay = True
			args.max_time_delay = time_delay
			soft_actor_critic = SoftActorCritic(args)
			policy_path = os.path.join(args.log_dir, 'own_sac_best_policy.pt')

			episode_returns_arr_per_td, last_dist_rem_arr_per_td, min_dist_rem_arr_per_td, pmax_values_per_td = \
				soft_actor_critic.test(policy_path, args, random_seeds)  
			
			num_bad_episodes = np.sum(min_dist_rem_arr_per_td < 5)

			end_dist_rem_arr.append(last_dist_rem_arr_per_td) 
			min_dist_rem_arr.append(min_dist_rem_arr_per_td)
			episode_returns_arr.append(episode_returns_arr_per_td)
			num_failures_arr.append(num_bad_episodes)

			plotting_list = plotting_list + list(zip([thres]*args.num_test_episodes, \
								['constant delay = %d' % time_delay]*args.num_test_episodes, \
								last_dist_rem_arr_per_td, \
								min_dist_rem_arr_per_td, \
								episode_returns_arr_per_td))
			if thres != 0.0:
				pmax_values_per_td = [1-pmax for pmax in pmax_values_per_td]
				pmax_plotting_list = pmax_plotting_list + list(zip([thres]*len(pmax_values_per_td), \
								['delay = %d' % time_delay]*len(pmax_values_per_td), \
								pmax_values_per_td))

			failure_list.append(np.array([thres, time_delay, num_bad_episodes]))

		args.constant_delay = False 
		args.max_time_delay = args.num_time_delays-1
		soft_actor_critic = SoftActorCritic(args)
		
		policy_path = os.path.join(args.log_dir, 'own_sac_best_policy.pt')

		episode_returns_arr_per_td, last_dist_rem_arr_per_td, min_dist_rem_arr_per_td, pmax_values_per_td = \
				soft_actor_critic.test(policy_path, args, random_seeds)   

		plotting_list = plotting_list + list(zip([thres]*args.num_test_episodes, \
								['random delay, max = %d' % args.max_time_delay]*args.num_test_episodes, \
								last_dist_rem_arr_per_td, \
								min_dist_rem_arr_per_td, \
								episode_returns_arr_per_td))

		#_, _, min_dist_rem_arr_per_td = soft_actor_critic.test(policy_path, args, random_seeds)  
		#if thres != 0.0:
		#	pmax_plotting_list = pmax_plotting_list + list(zip([thres]*len(pmax_values_per_td), \
		#							['random delay, max = %d' % args.max_time_delay]*len(pmax_values_per_td), \
		#							pmax_values_per_td))


	df = pd.DataFrame (plotting_list, columns = ['Threshold', 'time delay', 'end distance', 'min distance', 'return'])
	df.to_pickle("quantitative.pkl")

	#fig = plt.figure()
	#plot = sns.boxplot(x='time delay', y='min distance', data=df, order = None, hue = 'Threshold')
	
	#plt.ylim(-10, 30)
	#plt.savefig('min_dis_bp.pdf')
	#plt.clf()
	#plt.close()

	df = pd.DataFrame (pmax_plotting_list, columns = ['Threshold', 'time delay', 'maximum safety prob'])
	df.to_pickle("pmax.pkl")
	#fig = plt.figure()
	#plot = sns.boxplot(x='time delay', y='maximum safety prob', data=df, order = None, hue = 'Threshold')

	#plt.savefig('pmax_plot.png') 
	#plt.clf()
	#plt.close()#

	