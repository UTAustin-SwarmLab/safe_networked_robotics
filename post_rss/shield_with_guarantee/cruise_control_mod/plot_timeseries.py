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


csv_file = pd.read_csv('timeseries.csv')
print(csv_file)


csv_file['timeseries'] = csv_file['120.436761000']
csv_file['index'] = list(range(len(csv_file['timeseries'])))
csv_file[ 'rolling_avg' ] = csv_file.timeseries.rolling(50).mean()
print(csv_file)

plt.figure()
sns.lineplot( x = 'index',
			 y = 'rolling_avg',
			 data = csv_file, linewidth = 3.5, color='orange')

plt.savefig('timeseries.png')
