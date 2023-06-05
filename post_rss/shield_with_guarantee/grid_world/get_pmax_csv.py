import sys
sys.path.remove('/usr/lib/python3/dist-packages')
import os 
import numpy as np
import pandas as pd

loc = 'pmax_values_'
csv_files = os.listdir(loc)

pmax_plotting_list = []

for csv_file in csv_files:
    csv_file_loc = os.path.join(loc, csv_file)
    print(csv_file_loc)
    df = pd.read_csv(csv_file_loc)
    #print(df)
    
    if 'cd' in csv_file_loc:
        shield_type = 'constant delay = %d' % int(csv_file[15])
    else:
        shield_type = 'random delay, max = %d' % int(csv_file[15])

    #print(shield_type)

    threshold = float(csv_file.split('_')[-1][:-4])
    #print(threshold)

    pmax_values = list(df[list(df)[0]])
    
    if threshold == 1.0:
        continue

    num_pmax_values = len(pmax_values)
    shield_type_list = [shield_type]*num_pmax_values
    threshold_list = [threshold]*num_pmax_values

    pmax_plotting_list = pmax_plotting_list + list(zip(threshold_list, shield_type_list, pmax_values))

df = pd.DataFrame (pmax_plotting_list, columns = ['Threshold', 'time delay', 'maximum safety prob'])
print(df)
df.to_pickle("pmax.pkl")

