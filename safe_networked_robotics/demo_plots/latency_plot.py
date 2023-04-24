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

# SETTING GLOBAL PLOTTING PARAMETERS
sns.set_theme(style="darkgrid", palette="deep", color_codes='true')
FONT_SIZE = 20
LEGEND_FONT_SIZE = 16
TICK_LABEL_SIZE = 18
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

latency_csv_files_loc = 'log/2023-jan-29/rand_td/'
csv_files = os.listdir(latency_csv_files_loc)

required_csv_files = [csv_file for csv_file in csv_files if csv_file[:3] == 'run']
print(required_csv_files)

latency_list = []
latency_df = pd.DataFrame()

for csv_file in required_csv_files:
    csv_file_path = os.path.join(latency_csv_files_loc, csv_file)
    df = pd.read_csv(csv_file_path)
    df = df[150:]
    latency = list(df['Network Latency'])
    if max(latency) > 300:
        continue
    index = list(np.arange(0,len(latency)*0.1, 0.1))
    latency_list += list(zip(index, latency))

latency_csv_files_loc = 'log/2023-jan-29/no_shield/'
csv_files = os.listdir(latency_csv_files_loc)

required_csv_files = [csv_file for csv_file in csv_files if csv_file[:3] == 'run']
print(required_csv_files)

for csv_file in required_csv_files:
    csv_file_path = os.path.join(latency_csv_files_loc, csv_file)
    df = pd.read_csv(csv_file_path)
    df = df[150:]
    latency = list(df['Network Latency'])
    if max(latency) > 300:
        continue
    index = list(np.arange(0,len(latency)*0.1, 0.1))
    latency_list += list(zip(index, latency))

latency_df = pd.DataFrame(latency_list, columns=['Time (s)', 'Latency (ms)'])
sns.relplot(data=latency_df, x="Time (s)", y="Latency (ms)", kind="line") 
plt.xlim(0, 32)
plt.savefig('latency.pdf')   



ts = 100 
max_delay = 2
sys_lat = 2
delay_intervals = [(i*100, (i+1)*100) for i in range(max_delay+1)]
print(delay_intervals)
skip_val = 1

latency_list = latency_df['Latency (ms)']
abstract_latency_list = []
for lat in latency_list:
    abstract_lat = 0
    for j in range(max_delay+1):
        pot_delay_interval = delay_intervals[j]
        if lat >= pot_delay_interval[0] and lat <= pot_delay_interval[1]:
            abstract_lat = j
            break 

    abstract_latency_list.append(abstract_lat)

td_dist = np.zeros((max_delay+1, max_delay+1))
for i in range(len(abstract_latency_list)-skip_val):
    current_lat = abstract_latency_list[i]
    next_lat = abstract_latency_list[i+skip_val]
    #print(current_lat, next_lat)
    td_dist[current_lat, next_lat] += 1

for td in range(max_delay+1):
    if np.sum(td_dist[td]) == 0:
        continue
    td_dist[td] = td_dist[td]/np.sum(td_dist[td])

print(td_dist)


sns.heatmap(td_dist, annot=True, fmt=".3f",annot_kws={"fontsize":20})
plt.xlabel('Time Delay (steps)')
plt.ylabel('Time Delay (steps)')
plt.savefig('latency_matrix.pdf')  
