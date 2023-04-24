import sys 
sys.path.remove('/usr/lib/python3/dist-packages')
import os 
import csv 
import numpy as np 

ts = 100 
max_delay = 4
sys_lat = 2
delay_intervals = [(i*100, (i+1)*100) for i in range(max_delay)]
skip_val = 2


csv_loc = sys.argv[1]

csv_files = os.listdir(csv_loc)

req_csv_files = [f for f in csv_files if f[:4]=='spin']

td_dist = np.zeros((max_delay+1, max_delay+1))

for req_csv_file in req_csv_files:
    csv_file_loc = os.path.join(csv_loc, req_csv_file)
    
    f = open(csv_file_loc, 'r')
    csv_reader = csv.reader(f) 
    rowcount = 0
    for row in csv_reader:
        rowcount += 1
    rowcount -= 1
    f = open(csv_file_loc, 'r')

    abstract_lat_list = []
    csv_reader = csv.reader(f)
    for i, row in enumerate(csv_reader):
        if  i == 0:
            continue
        lat = float(row[0])     
        abstract_lat = 0
        for j in range(max_delay):
            pot_delay_interval = delay_intervals[j]
            if lat >= pot_delay_interval[0] and lat <= pot_delay_interval[1]:
                abstract_lat = j + sys_lat
                break 
        abstract_lat_list.append(abstract_lat)

    for i in range(len(abstract_lat_list)-skip_val):
        current_lat = abstract_lat_list[i]
        next_lat = abstract_lat_list[i+skip_val]
        td_dist[current_lat, next_lat] += 1
        #print(current_lat, next_lat)

for td in range(max_delay):
    if np.sum(td_dist[td]) == 0:
        continue
    td_dist[td] = td_dist[td]/np.sum(td_dist[td])

td_dist[4,3] = 0.99
td_dist[4,4] = 0.01
print(td_dist)
np.save('random_generated/td_dist', td_dist)