import os
import random
import itertools
import numpy as np 
import time 

abstract_actions = [1,2,3,4,5,6,7,8]
num_actions = len(abstract_actions)
max_td = 3
ustates_array = np.zeros((1, max_td), dtype=np.uint8)

start = time.time()
for td_val in range(1,max_td+1):
    print("current td val : ", td_val)
    num_zeros = max_td - td_val 
    zero_array = np.zeros((num_actions**td_val, num_zeros), dtype=np.uint8)
    beg = time.time()
    actions_list = list(itertools.product(abstract_actions, repeat=td_val))
    fin = time.time()
    print("time for product : ", fin-beg)
    actions_array = np.asarray(actions_list, dtype=np.uint8)
    actions_array = np.concatenate((actions_array, zero_array), axis=1)
    ustates_array = np.concatenate((ustates_array, actions_array), axis=0)

for ustate in ustates_array:
    print(ustate)

end = time.time()
print("time : ", end - start)
#print(ustates_array)


"""
abstract_actions_indices = [k for k in range(len(abstract_actions))]
start = time.time()
ustates_list = list(itertools.product(abstract_actions, repeat=max_td))
ustates_list_copy = ustates_list.copy()
ustates_array = np.asarray(ustates_list_copy)

for ustate in ustates_list:
	flag = False 
	bad_state = False 
	for i in range(max_td):
		if ustate[i] == -1:
			flag = True 
		if flag and ustate[i] != -1:
			bad_state = True 
			break 
	if bad_state:
		ustates_list_copy.remove(ustate)
ustates_list = ustates_list_copy

end = time.time()
print("time : ", end - start)
"""