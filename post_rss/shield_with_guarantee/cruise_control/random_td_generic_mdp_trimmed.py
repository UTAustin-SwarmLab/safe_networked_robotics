import os
import random
import itertools
import numpy as np 

class TimeDelayedMDP:
	def __init__(self, env, max_td, dist, zero_td_mdp, constant_td=False):
		self.env = env
		self.abstract_states = env.abstract_states  
		self.actions = env.abstract_actions
		self.abstract_actions = env.abstract_actions
		self.max_td = max_td
		self.constant_td = constant_td 
		self.td_dist = dist
		self.zero_td_mdp = zero_td_mdp
		self.difference = np.array([12960, 2160, 360, 60, 10, 2, 1])
		self.ustate_diff = np.array([2160, 360, 60, 10])
 
		self.ustates_list = []
		self.states = {}
		self.mdp = {}
		self.prob = {}
		
		#print(self.zero_td_mdp)
		#print(self.abstract_states)
		#print(self.abstract_actions)
		self.generate_ustates_list()
		#print(self.ustates_list)
		#print(self.states)
		self.get_one_step_transitions()

	
	def generate_ustates_list(self):
		self.abstract_actions = self.abstract_actions + [0]
		self.ustates_list = list(itertools.product(self.abstract_actions, repeat=self.max_td))

	def convert_mdp_state_to_id(self, state):
		state_feat = [state[0]] + list(state[1]) + [state[2]] + [state[3]]
		state_feat = np.array(state_feat)
		state_id = np.sum(self.difference * state_feat)
		return state_id

	
	def get_one_step_transitions(self):

		for abstract_state in self.abstract_states:
			print("abstract state : ", abstract_state)
			
			for ustate in self.ustates_list:
				ustate = np.array(ustate)
				num_valid_actions = len([u for u in ustate if u != 0])
				num_invalid_actions = self.max_td - num_valid_actions

				# invalid (-1, here 0) actions occur only at the end, not inbetween valid actions 
				if 0 in ustate[:num_valid_actions]:
					continue
			
				for td in range(self.max_td+1):
				
					for itm in [0, 1]:	
						if itm:
							# it has to be a non intermediate state 
							if td >= num_valid_actions:
								continue

						mdp_state = np.concatenate((np.array([abstract_state, td, itm]), ustate))
						#print(mdp_state)
						for abstract_action in self.actions:
							sa_pair = np.concatenate((mdp_state, np.array([abstract_action,])))
							#print(sa_pair)
