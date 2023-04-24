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
from scipy import stats

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
noshield_td_csv_files_loc = 'log/2023-jan-29/no_shield/'
rand_td_csv_files = [rand_td_csv_files_loc + csv_file for csv_file in os.listdir(rand_td_csv_files_loc)]
const_td_csv_files = [const_td_csv_files_loc + csv_file for csv_file in os.listdir(const_td_csv_files_loc)]
noshield_td_csv_files = [noshield_td_csv_files_loc + csv_file for csv_file in os.listdir(noshield_td_csv_files_loc)]
csv_files = rand_td_csv_files + const_td_csv_files + noshield_td_csv_files

required_csv_files = [csv_file for csv_file in csv_files if '/run_loop' in csv_file]
print(required_csv_files)

fig, ax = plt.subplots(figsize=(4.4,4.6))
dataset = []
for i, csv_file in enumerate(required_csv_files):
    df = pd.read_csv(csv_file)
    df = df[150:]
    latency = list(df['Network Latency'])
    state = list(df[' State'])
    opt_action = list(df[' Optimal Action'])
    shielded_action = list(df[' Shielded Action '])
    #print(i, csv_file)
    #print(max(latency), min(state), max(state))
    if 'rand_td' in csv_file and max(latency) > 300:
        continue
    
    

    index = list(np.arange(0,len(latency)*0.1, 0.1))
    safety = [0.2,]*len(index)
    state = [s+0.2 for s in state]
    if 'no_shield' not in csv_file:
        if min(state) < 0.5 or max(state) > 4.0:
            continue    

    print(i, csv_file)

    shielded = []
    for k in range(len(opt_action)):
        if opt_action[k] != shielded_action[k]:
            shielded.append("Shielded") 
        else:
            shielded.append("Unshielded")
    if 'const_td' in csv_file:
        run = ['Constant']*len(index)
    if 'rand_td' in csv_file:
        run = ['Random']*len(index)
    if 'no_shield' in csv_file:
        run = ['No Shield']*len(index)

    dataset = dataset + list(zip(index, latency, state, safety, shielded, run))

df = pd.DataFrame (dataset, columns = ['Time (s)', 'Latency (ms)', 'Distance (m)', 'Safety', 'Shielded', 'Shield Type'])
sns.boxplot(x='Shield Type', y='Distance (m)', data=df)
#sns.lineplot(x = "Shield Type", y = "Safety", data=df)
#ax.set_ylim(-1.0, 4)

#plt.savefig('demo_quantitative.png')
#plt.clf()
#plt.close()

print(df['Distance (m)'])
print(df['Shield Type'])

df1 = df[df['Shield Type'] == "Random"]
df2 = df[df['Shield Type'] == "Constant"]

print(df1, df2)

random_dist = list(df1['Distance (m)'])
constant_dist = list(df2['Distance (m)'])

print(len(random_dist), len(constant_dist))

print(stats.wilcoxon(random_dist[:1000], constant_dist[:1000]))
