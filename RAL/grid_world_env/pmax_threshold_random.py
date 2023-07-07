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

delta = 0.95
max_delay = 3

random_state_values = np.load('random_generated/Vmax_values_%d_td.npy' % max_delay, allow_pickle=True).item()
random_states = list(random_state_values.keys())

constant_state_values = np.load('constant_generated/Vmax_values_%d_td.npy' % max_delay, allow_pickle=True).item()
constant_states = list(constant_state_values.keys())

a_init = [4,4] 
ax_init = 4
ay_init =4

per_td_vis_arr = {}

vis_array = np.zeros((xmax, ymax))
for x in range(xmax):
    for y in range(ymax):
        physical_state = (x, y) + (ax_init, ay_init) + (1,)
        physical_state_id = convert_state_to_int(physical_state)
        print(physical_state, physical_state_id)
        invalid_control_vector = tuple([-1 for t in range(max_delay)])
        state = (physical_state_id,) + invalid_control_vector + (0,)
        max_safety_prob = random_state_values[state]
        if max_safety_prob >= delta:
                vis_array[ymax-1-y, x] = 1.0
        else:
            vis_array[ymax-1-y, x] = 0.0

per_td_vis_arr['random'] = vis_array

vis_array = np.zeros((xmax, ymax))
for x in range(xmax):
    for y in range(ymax):
        physical_state = (x, y) + (ax_init, ay_init) + (1,)
        physical_state_id = convert_state_to_int(physical_state)
        stay_control_vector = tuple([0 for t in range(max_delay)])
        state = (physical_state_id,) + stay_control_vector
        max_safety_prob = constant_state_values[state]
        if max_safety_prob >= delta:
                vis_array[ymax-1-y, x] = 1.0
        else:
            vis_array[ymax-1-y, x] = 0.0

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


fig, ax = plt.subplots(figsize=(6,6))
ax.grid(which="minor")
ax.imshow(threshold_vis_arr, extent=[0,xmax,0,xmax], aspect='equal')
ax.set_ylabel('Y-Coordinate')
ax.set_xlabel('X-Coordinate')
sdp = sns.color_palette("deep")
p1 = mpatches.Patch(edgecolor = 'none', facecolor= sdp[4])
a1 = mpatches.Patch(edgecolor = 'none', facecolor= sdp[7])
# ax.legend(handles=[p1,a1,a1,a1], labels=['\n','\n','  Random\n  Delay: 3 (max)','  Constant\n  Delay: 3'], 
#           loc = 'center', ncol = 2, bbox_to_anchor=(0.5,1.25),
#           handletextpad=0.0, handlelength=2.0, columnspacing=-0.0,
#           frameon=True, title='Shield Type',
#           title_fontproperties={'weight':'bold'}, alignment='right')
plt.tight_layout()
plt.savefig('gw_rd_pmax.pdf',dpi=300)
plt.savefig('gw_rd_pmax.eps',dpi=300)