import sys
import itertools
import numpy as np 
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd

# SETTING GLOBAL PLOTTING PARAMETERS
sns.set_theme(style="whitegrid", palette="deep", color_codes='true')
FONT_SIZE = 28
LEGEND_FONT_SIZE = 22
TICK_LABEL_SIZE = 22
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

# LOADING DATA
xmax = 8
ymax = 8
num_xbins = 8

def convert_state_to_int(state):
	increments = [(num_xbins**3)*2, (num_xbins**2)*2, num_xbins*2, 2, 1]
	return np.sum(np.multiply(list(state), increments))

ax_init = 4
ay_init = 4

delta = 0.95
per_td_vis_arr = {}

num_time_delays = 4

for td in range(num_time_delays):
    state_values = np.load('constant_generated/Vmax_values_%d_td.npy' % td, allow_pickle=True).item()
    states = list(state_values.keys())

    vis_array = np.zeros((xmax, ymax))

    for x in range(xmax): 
        for y in range(ymax):
            physical_state = (x, y) + (ax_init, ay_init) + (1,)
            physical_state_id = convert_state_to_int(physical_state)
            stay_control_vector = tuple([0 for t in range(td)])
            state = (physical_state_id,) + stay_control_vector
            max_safety_prob = state_values[state]
            if max_safety_prob >= delta:
                    vis_array[ymax-1-y, x] = 1.0
            else:
                vis_array[ymax-1-y, x] = 0.0

    per_td_vis_arr[td] = vis_array


threshold_vis_arr = np.ones((xmax, ymax, 3))

colors_list = sns.color_palette("deep")
colors_list[0], colors_list[1] = colors_list[1], colors_list[0]
colors_list[2], colors_list[3] = colors_list[3], colors_list[2]

for td in range(num_time_delays):
    vis_array = per_td_vis_arr[td]
    safe_states = np.where(vis_array)
    threshold_vis_arr[safe_states] = np.array(colors_list[td])

fig, ax = plt.subplots(figsize=(6,6))
ax.grid(which="minor")
ax.imshow(threshold_vis_arr, extent=[0,xmax,0,xmax], aspect='equal')
ax.set_ylabel('Y-Coordinate')
ax.set_xlabel('X-Coordinate')

sdp = sns.color_palette("deep")
b1 = mpatches.Patch(edgecolor = 'none', facecolor= sdp[0])
o1 = mpatches.Patch(edgecolor = 'none', facecolor= sdp[1])
g1 = mpatches.Patch(edgecolor = 'none', facecolor= sdp[2])
r1 = mpatches.Patch(edgecolor = 'none', facecolor= sdp[3])
categories = ['  Constant Delay: 0','  Constant Delay: 1','  Constant Delay: 2','  Constant Delay: 3']
# ax.legend(handles=[o1,b1,r1,g1,b1,b1,r1,g1,r1,r1,r1,g1,g1,g1,g1,g1], labels=12*['']+categories, 
#           loc = 'center', ncol = 4, bbox_to_anchor=(0.5,1.25),
#           handletextpad=0.0, handlelength=1.0, columnspacing=-0.0,
#           frameon=True, title='Shield Type',
#           title_fontproperties={'weight':'bold'}, alignment='right')

plt.tight_layout()
plt.savefig('gw_cd_pmax.pdf',dpi=300)
plt.savefig('gw_cd_pmax.eps',dpi=300)

