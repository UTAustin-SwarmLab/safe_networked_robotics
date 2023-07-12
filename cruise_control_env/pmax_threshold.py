import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
import matplotlib
from no_td_mdp import BasicMDP
import matplotlib.ticker as plticker
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerTuple
matplotlib.use('Agg')


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
ego_acc_list = np.load('actions.npy', allow_pickle=True)

delta = 0.95
num_time_delays = 4
per_td_vis_arr = {}

for td in range(num_time_delays):
    print(td)
    state_values = np.load('constant_generated/Vmax_values_%d_td.npy' % td, allow_pickle=True).item()
    states = list(state_values.keys())

    vis_array = np.zeros((num_rel_dist_states, num_rel_vel_states))

    for rel_dist_val in range(num_rel_dist_states):
        for rel_vel_val in range(num_rel_vel_states):
            physical_state = rel_dist_val * num_rel_vel_states + rel_vel_val 

            stay_control_vector = tuple([2 for t in range(td)])
            state = (physical_state,) + stay_control_vector
            max_safety_prob = state_values[state]
            if max_safety_prob >= delta:
                vis_array[rel_dist_val, rel_vel_val] = 1    # Unsafe states
            else:
                vis_array[rel_dist_val, rel_vel_val] = 0    # Safe states

    per_td_vis_arr[td] = vis_array

threshold_vis_arr = np.ones((num_rel_dist_states, num_rel_vel_states, 3))
colors_list = sns.color_palette("deep")
colors_list[0], colors_list[1] = colors_list[1], colors_list[0]
colors_list[2], colors_list[3] = colors_list[3], colors_list[2]
for td in range(num_time_delays):
    print(td)
    vis_array = per_td_vis_arr[td]
    safe_states = np.where(vis_array)
    threshold_vis_arr[safe_states] = np.array(colors_list[td])

# PLOTTING DATA
fig, ax = plt.subplots(figsize=(6,7.5))
ax.grid(which="minor")
ax.imshow(threshold_vis_arr, extent=[-5.5,5.5,26,4],aspect='auto')
ax.set_ylabel('Relative Distance (m)')
ax.set_xlabel('Relative Velocity (m/s)')

sdp = sns.color_palette("deep")
b1 = mpatches.Patch(edgecolor = 'none', facecolor= sdp[0])
o1 = mpatches.Patch(edgecolor = 'none', facecolor= sdp[1])
g1 = mpatches.Patch(edgecolor = 'none', facecolor= sdp[2])
r1 = mpatches.Patch(edgecolor = 'none', facecolor= sdp[3])
categories = ['  Constant Delay: 0','  Constant Delay: 1','  Constant Delay: 2','  Constant Delay: 3']
ax.legend(handles=[o1,b1,r1,g1,b1,b1,r1,g1,r1,r1,r1,g1,g1,g1,g1,g1], labels=12*['']+categories, 
          loc = 'center', ncol = 4, bbox_to_anchor=(0.5,1.25),
          handletextpad=0.0, handlelength=1.0, columnspacing=-0.0,
          frameon=True, title='Shield Type', title_fontproperties={'weight':'bold'}, alignment='right')

# VIEWING
plt.tight_layout()
fig.savefig('cc_cd_pmax.eps',dpi=300)
fig.savefig('cc_cd_pmax.pdf',dpi=300)

