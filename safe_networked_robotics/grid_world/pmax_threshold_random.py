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

sns.set_theme(style="darkgrid", palette="deep", color_codes='true')
FONT_SIZE = 20
LEGEND_FONT_SIZE = 16
TICK_LABEL_SIZE = 14
plt.rc('text', usetex=False)
params = {'legend.fontsize'     : LEGEND_FONT_SIZE,
          'legend.title_fontsize': LEGEND_FONT_SIZE,
		  'axes.labelsize'      : FONT_SIZE,
		  'axes.titlesize'      : FONT_SIZE,
		  'xtick.labelsize'     : TICK_LABEL_SIZE,
		  'ytick.labelsize'     : TICK_LABEL_SIZE,
		  'figure.autolayout'   : True,
          'axes.labelweight'    : 'bold',
          'font.weight'         : 'normal'
         }
plt.rcParams.update(params)

def convert_state_to_int(state):
	int_val = 0
	state_len = len(state)
	for i, j in enumerate(reversed(range(state_len))):
		int_val += state[j] * 10 ** i 
	return int_val

xmax = 8
ymax = 8

threshold = 0.95
max_delay = 3

random_state_values = np.load('random_generated/state_values_%d_td.npy' % max_delay, allow_pickle=True).item()
random_states = list(random_state_values.keys())

constant_state_values = np.load('constant_generated/state_values_%d_td.npy' % max_delay, allow_pickle=True).item()
constant_states = list(constant_state_values.keys())

a_inits = [[4,4], [6,7]]
pmax_arrays = []


for a_init in a_inits:
    ax_init = a_init[0]
    ay_init = a_init[1]

    per_td_vis_arr = {}

    vis_array = np.zeros((xmax, ymax))

    for x in range(xmax):
        for y in range(ymax):
            physical_state = (x, y) + (ax_init, ay_init) + (1,)
            physical_state = convert_state_to_int(physical_state)
            invalid_control_vector = tuple([-1 for t in range(max_delay)])
            state = (physical_state,) + invalid_control_vector + (0,)
            min_reach_prob = random_state_values[state]
            if min_reach_prob >= 1-threshold:
                    vis_array[ymax-1-y, x] = 0.0
            else:
                vis_array[ymax-1-y, x] = 1.0

    per_td_vis_arr['random'] = vis_array

    vis_array = np.zeros((xmax, ymax))

    for x in range(xmax):
        for y in range(ymax):
            physical_state = (x, y) + (ax_init, ay_init)
            stay_control_vector = tuple([0 for t in range(max_delay)])
            state = physical_state + stay_control_vector + (1,)
            min_reach_prob = constant_state_values[state]
            if min_reach_prob >= 1-threshold:
                    vis_array[ymax-1-y, x] = 0.0
            else:
                vis_array[ymax-1-y, x] = 1.0

    per_td_vis_arr['constant'] = vis_array


    threshold_vis_arr = np.ones((xmax, ymax, 3))

    colors_list = sns.color_palette("deep")

    for i, td in enumerate(['random', 'constant']):
        print(i,td)
        vis_arr = per_td_vis_arr[td]
        safe_states = np.where(vis_arr)
        if i == 1:
            threshold_vis_arr[safe_states] = np.array(colors_list[7])
        else:
            threshold_vis_arr[safe_states] = np.array(colors_list[4])

    pmax_arrays.append(threshold_vis_arr)

fig, ax = plt.subplots(len(a_inits),1)

for i, a_init in enumerate(a_inits):
    ax[i].imshow(pmax_arrays[i])
    #ax[i].set_xticks(np.arange(0, 8, 1))
    #ax[i].set_yticks(np.arange(0, 8, 1))

    # Labels for major ticks
    #ax[i].set_xticklabels(np.arange(0, 8, 1))
    #ax[i].set_yticklabels(np.arange(0, 8, 1))

    # Minor ticks
    ax[i].set_xticks(np.arange(-.5, 8, 1), minor=True)
    ax[i].set_yticks(np.arange(-.5, 8, 1), minor=True)

    # Gridlines based on minor ticks
    ax[i].grid(which='minor', color='w', linestyle='-', linewidth=1)

    # Remove minor ticks
    #ax[i].tick_params(which='minor', bottom=False, left=False)

plt.savefig('random_generated/pmax_plot_combined.pdf')
