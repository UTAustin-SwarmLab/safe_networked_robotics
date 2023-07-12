import pickle 
import collections 
import numpy as np 
from no_td_mdp import BasicMDP

policy_path = 'policy.pkl'
filehandler = open(policy_path, 'rb') 
Q = pickle.load(filehandler)
Q = collections.defaultdict(lambda: np.zeros(5), Q)

basic_mdp_obj = BasicMDP()

policy = {}
for rx in range(basic_mdp_obj.xmin, basic_mdp_obj.xmax+1):
	for ry in range(basic_mdp_obj.ymin, basic_mdp_obj.ymax+1): 
		for ax in range(basic_mdp_obj.xmin, basic_mdp_obj.xmax+1):
			for ay in range(basic_mdp_obj.ymin, basic_mdp_obj.ymax+1):
				query = (rx,ry,ax,ay)
				# print(Q[query])
				sa_values = list(Q[query])
				# print(sa_values, query)
				
				for flag in [0, 1]:
					physical_state = (rx, ry, ax, ay, flag)
					state_id = basic_mdp_obj.convert_physical_state_to_int(physical_state)
					state = (state_id,)
					policy[state] = sa_values

# print(policy)

np.save('abstracted_dnn_policy', policy)