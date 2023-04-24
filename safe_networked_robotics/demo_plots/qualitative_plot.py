import sys 
sys.path.remove('/usr/lib/python3/dist-packages')
import os 
import itertools 
import numpy as np
import seaborn as sns


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd

FONT_SIZE = 20
FONT_SIZE = 18 
sns.set_color_codes() 
sns.set(style='ticks')
sns.set_style("darkgrid")

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
plt.rcParams["axes.grid"] = True
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

rand_td_csv_files_loc = 'log/2023-jan-29/rand_td/'
const_td_csv_files_loc = 'log/2023-jan-29/const_td/'
rand_td_csv_files = [rand_td_csv_files_loc + csv_file for csv_file in os.listdir(rand_td_csv_files_loc)]
const_td_csv_files = [const_td_csv_files_loc + csv_file for csv_file in os.listdir(const_td_csv_files_loc)]
csv_files = rand_td_csv_files + const_td_csv_files

required_csv_files = [csv_file for csv_file in csv_files if '/run_loop' in csv_file]
print(required_csv_files)


latency_list = []
latency_df = pd.DataFrame()

"""
constant delay curated qualitative
"""
fig, ax = plt.subplots()
dataset = []
plotting_pd = []
j = 0
for i, csv_file in enumerate(required_csv_files):
    if i not in [19, 20]:
        continue
    print(i, csv_file)
    df = pd.read_csv(csv_file)
    df = df[150:]
    latency = list(df['Network Latency'])
    state = list(df[' State'])
    opt_action = list(df[' Optimal Action'])
    shielded_action = list(df[' Shielded Action '])

    if 'rand_td' in csv_file and max(latency) > 300:
        continue
    index = list(np.arange(0,len(latency)*0.1, 0.1))
    safety = [0.2,]*len(index)
    state = [s+0.2 for s in state]

    shielded = []
    for k in range(len(opt_action)):
        if opt_action[k] != shielded_action[k]:
            shielded.append("Shielded") 
        else:
            shielded.append("Unshielded")
    
    run = ['run_%d' % j]*len(index)
    j += 1
    dataset = dataset + list(zip(index, latency, state, safety, shielded, run))
    
    #time_col_name = 'time_run_%d' % j 
    #latency_col_name = 'latency_run_%d' % j
    #distance_col_name = 'distance_run_%d' % j
    #safety_col_name = 'safety_run_%d' % j
    #shielded_col_name = 'shielded_run_%d' % j

    #plotting_pd[time_col_name] = index
    #plotting_pd[latency_col_name] = latency 
    #plotting_pd[distance_col_name] = state 
    #plotting_pd[safety_col_name] = safety
    #plotting_pd[shielded_col_name] = shielded

    
#print(new_df)
new_df = pd.DataFrame(dataset, columns=['Time (s)', 'Latency (ms)', 'Distance (m)', 'Safety', 'Shielded', 'Run'])

sns.scatterplot( x="Time (s)", y="Distance (m)", data=new_df, hue='Shielded', palette="pastel")    
sns.lineplot(data=new_df, x="Time (s)", y="Distance (m)", hue="Run", palette="flare")

sns.lineplot(x = "Time (s)", y = "Safety", data=new_df)

#sns.lineplot(y = "Safety", x = "Time (s)", data=new_df, legend=False)
plt.ylim(0, 3)

sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=4, title=None, frameon=False)
loc = 'const_delay_qualitative.png'
plt.savefig(loc)
plt.close()

print(plt.rcParams["figure.figsize"])

"""
random delay curated qualitative
"""

#fig, axes = plt.subplots(2,1, figsize=(6.4,9.6))
fig, ax = plt.subplots()
dataset = []
plotting_pd = []
j = 0
for i, csv_file in enumerate(required_csv_files):
    if i not in [5, 8]:
        continue
    print(i, csv_file)
    df = pd.read_csv(csv_file)
    df = df[150:]
    latency = list(df['Network Latency'])
    state = list(df[' State'])
    opt_action = list(df[' Optimal Action'])
    shielded_action = list(df[' Shielded Action '])

    if 'rand_td' in csv_file and max(latency) > 300:
        continue
    index = list(np.arange(0,len(latency)*0.1, 0.1))
    safety = [0.2,]*len(index)
    state = [s+0.2 for s in state]

    shielded = []
    for k in range(len(opt_action)):
        if opt_action[k] != shielded_action[k]:
            shielded.append("Shielded") 
        else:
            shielded.append("Unshielded")
    
    run = ['run_%d' % j]*len(index)
    j += 1
    dataset = dataset + list(zip(index, latency, state, safety, shielded, run))

    
new_df = pd.DataFrame(dataset, columns=['Time (s)', 'Latency (ms)', 'Distance (m)', 'Safety', 'Shielded', 'Run'])

#sns.scatterplot(ax=axes[0], x="Time (s)", y="Distance (m)", data=new_df, hue='Shielded', palette="pastel")    
#sns.lineplot(ax=axes[0], data=new_df, x="Time (s)", y="Distance (m)", hue="Run", palette="flare")
#sns.lineplot(ax=axes[0], x = "Time (s)", y = "Safety", data=new_df)
#axes[0].set_ylim(0, 3)
#sns.move_legend(axes[0], "lower center", bbox_to_anchor=(.5, 1), ncol=4, title=None, frameon=False)

#sns.lineplot(ax=axes[1], data=new_df, x="Time (s)", y="Latency (ms)", hue="Run", palette="flare")
#axes[1].set_ylim(0,200)
#sns.move_legend(axes[1], "lower center", bbox_to_anchor=(.5, 1), ncol=2, title=None, frameon=False)

sns.scatterplot(ax=ax, x="Time (s)", y="Distance (m)", data=new_df, hue='Shielded', palette="pastel")    
sns.lineplot(ax=ax, data=new_df, x="Time (s)", y="Distance (m)", hue="Run", palette="flare")
sns.lineplot(ax=ax, x = "Time (s)", y = "Safety", data=new_df)
ax.set_ylim(0, 3)
sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=4, title=None, frameon=False)

ax2 = ax.twinx()

sns.lineplot(ax=ax2, data=new_df, x="Time (s)", y="Latency (ms)", hue="Run", palette="flare", linestyle='--', legend=False)
ax2.set_ylim(50,150)


loc = 'random_delay_qualitative.png'
plt.savefig(loc)
plt.close()


