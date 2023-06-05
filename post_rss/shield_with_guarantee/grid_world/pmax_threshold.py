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

plt.rc('text', usetex=True)
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

xmax = 8
ymax = 8

ax_init = 4
ay_init = 4

threshold = 0.95
per_td_vis_arr = {}

num_time_delays = 4

for td in range(num_time_delays):
    state_values = np.load('constant_generated/state_values_%d_td.npy' % td, allow_pickle=True).item()
    states = list(state_values.keys())

    vis_array = np.zeros((xmax, ymax))

    for x in range(xmax):
        for y in range(ymax):
            physical_state = (x, y) + (ax_init, ay_init)
            stay_control_vector = tuple([0 for t in range(td)])
            #relevant_states = [s for s in states if s[:4] == physical_state]
            #relevant_states = [s for s in relevant_states if s[-1] == 1]
            #relevant_states_values = [state_values[s] for s in relevant_states]
            state = physical_state + stay_control_vector + (1,)
            min_reach_prob = state_values[state]
            if min_reach_prob < threshold:
                    vis_array[ymax-1-y, x] = 0.0#(td+1)*1.0 / num_time_delays
            else:
                vis_array[ymax-1-y, x] = 1.0

    per_td_vis_arr[td] = vis_array


threshold_vis_arr = np.ones((xmax, ymax, 3))

colors_list = sns.color_palette("cubehelix")
#print(colors_list)

for td in range(num_time_delays):
    vis_array = per_td_vis_arr[td]
    safe_states = np.where(vis_array)
    #print(safe_states)
    threshold_vis_arr[safe_states] = np.array(colors_list[td])

fig, ax = plt.subplots()

plt.imshow(threshold_vis_arr, extent=[0,xmax,0,xmax])
#plt.axhline(25, color='black', linewidth=1)
#plt.axvline(-10, color='black', linewidth=1)
#plt.ylabel('Relative Distance (m)')
#plt.xlabel('Relative velocity (m/s)')


plt.savefig('constant_generated/pmax_plot_combined_%d_%d.pdf'%(ax_init, ay_init))

