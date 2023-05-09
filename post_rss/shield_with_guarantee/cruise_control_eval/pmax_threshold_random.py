import sys
sys.path.remove('/usr/lib/python3/dist-packages')
import itertools
import numpy as np 
import seaborn as sns

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns
import seaborn
import pandas as pd

FONT_SIZE = 20
FONT_SIZE = 18 
sns.set_color_codes() 
seaborn.set()

plt.rc('text', usetex=False)
font = {'weight' : 'bold',
		'size'   : FONT_SIZE}
plt.rc('font', **font)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']

LEGEND_FONT_SIZE = 14
XTICK_LABEL_SIZE = 14
YTICK_LABEL_SIZE = 14

plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["font.weight"] = "bold"
plt.style.use('seaborn-whitegrid')
plt.rcParams["axes.grid"] = False
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]


import matplotlib.pylab as pylab
params = {'legend.fontsize': LEGEND_FONT_SIZE,
		 'axes.labelsize': FONT_SIZE,
		 'axes.titlesize': FONT_SIZE,
		 'xtick.labelsize': XTICK_LABEL_SIZE,
		 'ytick.labelsize': YTICK_LABEL_SIZE,
		 'figure.autolayout': True}
pylab.rcParams.update(params)
plt.rcParams["axes.labelweight"] = "bold"

num_rel_dist_states = 27
num_rel_vel_states = 22

threshold = 0.95
max_delay = 3

"""
obtaining values for max delay (constant)
"""
vis_arr = {}

state_values = np.load('constant_generated/state_values_%d_td.npy' % max_delay, allow_pickle=True).item()
states = list(state_values.keys())

vis_array = np.zeros((num_rel_dist_states, num_rel_vel_states))

for rel_dist_val in range(num_rel_dist_states):
    for rel_vel_val in range(num_rel_vel_states):
        physical_state = rel_dist_val * num_rel_vel_states + rel_vel_val 

        # stay control vector
        stay_control_vector = tuple([2 for t in range(max_delay)])
        state = (physical_state,) + stay_control_vector
        min_reach_prob = state_values[state]
        #max_safety_prob = 1-state_values[state]
        #if max_safety_prob >= threshold:
        if min_reach_prob >= 1-threshold:
            vis_array[rel_dist_val, rel_vel_val] = 0.0#(td+1)*1.0 / num_time_delays
        else:
            vis_array[rel_dist_val, rel_vel_val] = 1.0

vis_arr['constant'] = vis_array

"""
obtaining values for random delay
"""
state_values = np.load('random_generated/state_values_%d_td.npy' % max_delay, allow_pickle=True).item()
states = list(state_values.keys())

vis_array = np.zeros((num_rel_dist_states, num_rel_vel_states))

for rel_dist_val in range(num_rel_dist_states):
    for rel_vel_val in range(num_rel_vel_states):
        physical_state = rel_dist_val * num_rel_vel_states + rel_vel_val 

        # stay control vector
        invalid_control_vector = tuple([-1 for t in range(max_delay)])
        state = (physical_state,) + invalid_control_vector + (0,)
        min_reach_prob = state_values[state]
        #max_safety_prob = 1-state_values[state]
        #if max_safety_prob >= threshold:
        if min_reach_prob >= 1-threshold:
            vis_array[rel_dist_val, rel_vel_val] = 0.0#(td+1)*1.0 / num_time_delays
        else:
            vis_array[rel_dist_val, rel_vel_val] = 1.0

vis_arr['random'] = vis_array

threshold_vis_arr = np.ones((num_rel_dist_states, num_rel_vel_states, 3))

colors_list = sns.color_palette("cubehelix")
#print(colors_list)

for i, td in enumerate(['random', 'constant']):
    print(i,td)
    vis_array = vis_arr[td]
    safe_states = np.where(vis_array)
    if i == 1:
        threshold_vis_arr[safe_states] = np.array(colors_list[3])
    else:
        threshold_vis_arr[safe_states] = np.array(colors_list[i])

fig, ax = plt.subplots()

plt.imshow(threshold_vis_arr, extent=[-10,10,25,0])
plt.ylabel('Relative Distance (m)')
plt.xlabel('Relative velocity (m/s)')


plt.savefig('random_generated/pmax_plot_combined.pdf')

