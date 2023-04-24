import sys 
sys.path.remove('/usr/lib/python3/dist-packages')
from scipy import stats
import pandas as pd 

import os 
import itertools 
import numpy as np
import seaborn as sns

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.read_csv("blood_pressure.csv")
print(df[['bp_before','bp_after']].describe())

