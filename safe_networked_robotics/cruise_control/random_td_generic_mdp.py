import os
import csv
import random
import itertools
import numpy as np 

class TimeDelayedMDP:
	def __init__(self, env, max_td, td_dist):
		self.env = env
		self.abstract_states = env.abstract_states  
		self.actions = env.abstract_actions # 1,2,3,4,5
		self.abstract_actions = env.abstract_actions # 1,2,3,4,5
		self.num_actions = len(self.actions) + 1 #6
		self.max_td = max_td
		self.zero_td_mdp = env.zero_td_mdp
		self.td_dist = td_dist
		self.difference = np.array([(self.max_td+1)*2*(self.num_actions**(i+1)) for i in reversed(range(self.max_td))] + [(self.max_td+1)*2, 2, 1])
 		
		self.ustates_list = []
		self.states = {}
		self.mdp = {}
		self.prob = {}

		self.f_mdp = open('random_generated/mdp_max_td_%d.csv'%(self.max_td), 'w')
		self.mdp_writer = csv.writer(self.f_mdp)
		
		self.generate_ustates_list()
		#print(self.ustates_list)
		self.get_one_step_transitions()
		#self.generate_one_step_transitions([54, (5, 2), 1, 1], 2)
	
	def generate_ustates_list(self):
		self.abstract_actions = self.abstract_actions + [0]
		self.ustates_list = list(itertools.product(self.abstract_actions, repeat=self.max_td))

	def convert_mdp_state_to_id(self, state):
		state_feat = [state[0]] + list(state[1]) + [state[2]] + [state[3]]
		state_feat = np.array(state_feat)
		state_id = np.sum(self.difference * state_feat)
		return state_id

	
	def get_one_step_transitions(self):
		"""
		A random td state has 4 components
		1. The abstract env state
		2. Action buffer
		3. current value of time delay
		4. whether the state is intermediate or not
		"""
		for abstract_state in self.abstract_states:
			print("abstract state : ", abstract_state)
			
			for ustate in self.ustates_list:
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

						mdp_state = (abstract_state, ustate, td, itm)
						mdp_state_id = self.convert_mdp_state_to_id(mdp_state)

						for abstract_action in self.actions:
							state_action_pair = (mdp_state_id, abstract_action)
							#print("state action pair : ", state_action_pair)

							# if the state is intermediate
							if itm:
								control_command = ustate[0] 
								query = (abstract_state, control_command)
								next_abstract_states = self.zero_td_mdp[query]
								next_ustate = ustate[1:] + (0,)
								num = len([u for u in next_ustate if u != 0])
								if num > td: 
									next_itm = 1
								else:
									next_itm = 0	

							# if the state is not intermediate
							else: 
								if td > num_valid_actions:
									next_abstract_states = [abstract_state]
									next_ustate = ustate[:num_valid_actions] + (abstract_action,) + (0,)*(num_invalid_actions-1)
									next_itm = 0 
								else:
									next_ustate = ustate[:num_valid_actions] + (abstract_action,) + (0,)*num_invalid_actions
									control_command = next_ustate[0]
									next_ustate = next_ustate[1:]
									num = len([u for u in next_ustate if u != 0])
									if num > td: 
										next_itm = 1
									else:
										next_itm = 0
									query = (abstract_state, control_command)
									next_abstract_states = self.zero_td_mdp[query]
									

							next_td = []
							if next_itm:
								next_td = [td]
							else:
								next_td = [k for k in range(self.max_td+1)]

							if next_itm:
								prob_list = [1/len(next_abstract_states)] * len(next_abstract_states)
							else:
								prob_list = list(self.td_dist.values())
								prob_list = [prob_list[prob_idx] / len(next_abstract_states) for prob_idx in range(len(prob_list))]
								prob_list = prob_list * len(next_abstract_states)

							next_mdp_states = list(itertools.product(next_abstract_states, [next_ustate], next_td, [next_itm]))
							next_mdp_states_id = [self.convert_mdp_state_to_id(next_mdp_state) for next_mdp_state in next_mdp_states]
							transition = [mdp_state_id, abstract_action] + next_mdp_states_id + prob_list

							self.mdp_writer.writerow(transition)


	def generate_one_step_transitions(self, mdp_state, abstract_action):
		abstract_state = mdp_state[0]
		ustate = mdp_state[1]
		td = mdp_state[2]
		itm = mdp_state[3]

		num_valid_actions = len([u for u in ustate if u != 0])
		num_invalid_actions = self.max_td - num_valid_actions

		# invalid (-1, here 0) actions occur only at the end, not inbetween valid actions 
		if 0 in ustate[:num_valid_actions]:
			print("invalid state, 0 cannot occur before a valid action")
			return 0

		if itm:
			# it has to be a non intermediate state 
			if td >= num_valid_actions:
				print("If it is an intermediate state, the time delay should be less than the number of valid actions in the buffer")
				return 0

		mdp_state_id = self.convert_mdp_state_to_id(mdp_state)

		state_action_pair = (mdp_state_id, abstract_action)
		#print("state action pair : ", state_action_pair)

		# if the state is intermediate
		if itm:
			control_command = ustate[0] 
			query = (abstract_state, control_command)
			next_abstract_states = self.zero_td_mdp[query]
			next_ustate = ustate[1:] + (0,)
			num = len([u for u in next_ustate if u != 0])
			if num > td: 
				next_itm = 1
			else:
				next_itm = 0	

		# if the state is not intermediate
		else: 
			if td > num_valid_actions:
				next_abstract_states = [abstract_state]
				next_ustate = ustate[:num_valid_actions] + (abstract_action,) + (0,)*(num_invalid_actions-1)
				next_itm = 0 
			else:
				next_ustate = ustate[:num_valid_actions] + (abstract_action,) + (0,)*num_invalid_actions
				control_command = next_ustate[0]
				next_ustate = next_ustate[1:]
				num = len([u for u in next_ustate if u != 0])
				if num > td: 
					next_itm = 1
				else:
					next_itm = 0
				query = (abstract_state, control_command)
				next_abstract_states = self.zero_td_mdp[query]
									
		next_td = []
		if next_itm:
			next_td = [td]
		else:
			next_td = [k for k in range(self.max_td+1)]

		print(next_abstract_states)
		next_mdp_states = list(itertools.product(next_abstract_states, [next_ustate], next_td, [next_itm]))
		print(next_mdp_states)

		if next_itm:
			prob_list = [1/len(next_abstract_states)] * len(next_abstract_states)
		else:
			prob_list = list(self.td_dist.values())
			prob_list = [prob_list[prob_idx] / len(next_abstract_states) for prob_idx in range(len(prob_list))]
			prob_list = prob_list * len(next_abstract_states)

		print(prob_list)
		return 0