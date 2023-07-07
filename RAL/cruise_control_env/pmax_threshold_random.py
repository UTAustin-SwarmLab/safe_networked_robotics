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
from no_td_mdp import BasicMDP

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
basic_mdp = BasicMDP()
num_rel_dist_states = basic_mdp.num_rel_dist_indices
num_rel_vel_states = basic_mdp.num_rel_vel_indices

delta = 0.95
max_delay = 3

"""
obtaining values for max delay (constant)
"""
vis_arr = {}

state_values = np.load('constant_generated/Vmax_values_%d_td.npy' % max_delay, allow_pickle=True).item()
states = list(state_values.keys())

vis_array = np.zeros((num_rel_dist_states, num_rel_vel_states))

for rel_dist_val in range(num_rel_dist_states):
    for rel_vel_val in range(num_rel_vel_states):
        physical_state = rel_dist_val * num_rel_vel_states + rel_vel_val 

        # stay control vector
        stay_control_vector = tuple([2 for t in range(max_delay)])
        state = (physical_state,) + stay_control_vector
        max_safety_prob = state_values[state]
        if max_safety_prob >= delta:
            vis_array[rel_dist_val, rel_vel_val] = 1    # Unsafe states
        else:
            vis_array[rel_dist_val, rel_vel_val] = 0    # Safe states

vis_arr['constant'] = vis_array

"""
obtaining values for random delay
"""
state_values = np.load('random_generated/Vmax_values_%d_td.npy' % max_delay, allow_pickle=True).item()
states = list(state_values.keys())

vis_array = np.zeros((num_rel_dist_states, num_rel_vel_states))

for rel_dist_val in range(num_rel_dist_states):
    for rel_vel_val in range(num_rel_vel_states):
        physical_state = rel_dist_val * num_rel_vel_states + rel_vel_val 

        # stay control vector
        invalid_control_vector = tuple([-1 for t in range(max_delay)])
        state = (physical_state,) + invalid_control_vector + (0,)
        max_safety_prob = state_values[state]
        if max_safety_prob >= delta:
            vis_array[rel_dist_val, rel_vel_val] = 1    # Unsafe states
        else:
            vis_array[rel_dist_val, rel_vel_val] = 0    # Safe states

vis_arr['random'] = vis_array

threshold_vis_arr = np.ones((num_rel_dist_states, num_rel_vel_states, 3))

colors_list = sns.color_palette("deep")
#print(colors_list)

for i, td in enumerate(['random', 'constant']):
    print(i,td)
    vis_array = vis_arr[td]
    safe_states = np.where(vis_array)
    if i == 1:
        threshold_vis_arr[safe_states] = np.array(colors_list[7])
    else:
        threshold_vis_arr[safe_states] = np.array(colors_list[4])

fig, ax = plt.subplots(figsize=(6,7.5))
ax.grid(which="minor")
ax.imshow(threshold_vis_arr, extent=[-5.5,5.5,26,4],aspect='auto')
ax.set_ylabel('Relative Distance (m)')
ax.set_xlabel('Relative Velocity (m/s)')

sdp = sns.color_palette("deep")
p1 = mpatches.Patch(edgecolor = 'none', facecolor= sdp[4])
a1 = mpatches.Patch(edgecolor = 'none', facecolor= sdp[7])
ax.legend(handles=[p1,a1,a1,a1], labels=['\n','\n','  Random\n  Delay: 3 (max)','  Constant\n  Delay: 3'], 
          loc = 'center', ncol = 2, bbox_to_anchor=(0.5,1.25),
          handletextpad=0.0, handlelength=2.0, columnspacing=-0.0,
          frameon=True, title='Shield Type', title_fontproperties={'weight':'bold'}, alignment='right')

plt.tight_layout()
plt.savefig('cc_rd_pmax.eps',dpi=300)
plt.savefig('cc_rd_pmax.pdf',dpi=300)

