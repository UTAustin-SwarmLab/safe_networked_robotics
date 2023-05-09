import os 
import csv
import random 
import itertools
import numpy as np  

class TimeDelayedMDP:
	def __init__(self, env, max_td, td_dist):
		self.env = env
		self.abstract_states = env.abstract_states  
		self.abstract_actions = env.abstract_actions # 0,1,2,3,4,5
		self.num_actions = len(self.abstract_actions) #6
		self.invalid_action = min(self.abstract_actions)
		self.max_td = max_td
		self.zero_td_mdp = env.zero_td_mdp
		self.td_dist = td_dist

		self.ustates_list = []
		self.states = {} 
		self.mdp = {} 
		self.prob = {}
		
		self.generate_ustates_list()
		#print(self.ustates_list)
		self.get_one_step_transitions()
		#self.generate_one_step_transitions((201, 3, 4, -1, 2), 4)
		#self.generate_one_step_transitions((201, 3, 4, 1, 0), 3)
	
	def generate_ustates_list(self):
		self.ustates_list = list(itertools.product(self.abstract_actions, repeat=self.max_td))

	def get_one_step_transitions(self):
		"""
		A random td state has 3 components
		1. The abstract env state
		2. Action buffer
		3. whether the state is intermediate or not 
		"""
		for abstract_state in self.abstract_states:
			print("abstract state : ", abstract_state)

			for ustate in self.ustates_list:
				num_valid_actions = len([u for u in ustate if u != self.invalid_action])
				num_invalid_actions = self.max_td - num_valid_actions
				
				# invalid (self.invalid_action (5)) actions occur only at the end, not inbetween valid actions 
				if self.invalid_action in ustate[:num_valid_actions]:
					continue 
				
				current_td = num_valid_actions
		
				for itm in range(num_valid_actions+1):
					# itm cannot be greater than the number of valid actions
					# 0,...,num_valid_actions
					state = abstract_state + ustate + (itm,)

					for action in self.abstract_actions:
						if action == self.invalid_action:
							continue 
						state_action_pair = (state, action)
						
						#if state_action_pair != ((133, -1, -1, -1, 0), 0):
						#	continue
						#print(state_action_pair)
						transitions = []
						probabilities = []
						
						if itm > 0:
							next_itm = itm-1
							control_command = ustate[0]
							query = (abstract_state, control_command)
							next_abstract_states = self.zero_td_mdp[query]
							next_ustate = ustate[1:] + (self.invalid_action,)
							for st in next_abstract_states:
								next_st = st + next_ustate + (next_itm,)
								transitions.append(next_st)
								probabilities.append(1/len(next_abstract_states))
							
						else:
							for next_td in range(self.max_td+1):
								if next_td > current_td+1:
									break
								if next_td <= current_td:
									next_itm = current_td - next_td
									next_ustate = ustate[:num_valid_actions] + (action,) + (self.invalid_action,)*num_invalid_actions
									control_command = next_ustate[0]
									next_ustate = next_ustate[1:]
									query = (abstract_state, control_command)
									next_abstract_states = self.zero_td_mdp[query]
									for st in next_abstract_states:
										next_st = st + next_ustate + (next_itm,)
										transitions.append(next_st)
										probabilities.append(self.td_dist[current_td, next_td] / len(next_abstract_states))
								else:
									next_itm = 0
									next_abstract_states = [abstract_state]
									next_ustate = ustate[:num_valid_actions] + (action,) + (self.invalid_action,)*(num_invalid_actions-1)
									for st in next_abstract_states:
										next_st = st + next_ustate + (next_itm,)
										transitions.append(next_st)
										probabilities.append(self.td_dist[current_td, next_td])

						#print(transitions)
						self.mdp[state_action_pair] = transitions
						self.prob[state_action_pair] = probabilities

		os.makedirs('random_generated', exist_ok=True)
		np.save('random_generated/random_mdp_%d_td' % self.max_td, self.mdp)
		np.save('random_generated/random_mdp_prob_%d_td' % self.max_td, self.prob)

	def generate_one_step_transitions(self, state, action):
		abstract_state = state[0]
		ustate = state[1:-1]
		itm = state[-1]

		print("abstract state : ", abstract_state)
		print("ustate : ", ustate)
		print("itermediate status : ", itm)
		print("action : ", action)

		num_valid_actions = len([u for u in ustate if u != self.invalid_action]) # the current value of time delay
		num_invalid_actions = self.max_td - num_valid_actions

		# there cannot be any invalid action inbetween valid actions
		if self.invalid_action in ustate[:num_valid_actions]:
			print("invalid ustate")
			return 0

		current_td = num_valid_actions
		print("current time delay : ", current_td)

		if action == self.invalid_action:
			print("invalid action")
			return 0

		# the value of itm indicates how many actions to be executed before exiting intermediate status.
		# if itm is 0, the state is not intermediate
		if num_valid_actions < itm:
			print("invalid itm")
			return 0

		state_action_pair = (state, action)

		transitions = []
						
		if itm > 0:
			print("In an intermediate state")
			next_itm = itm-1
			print("number of steps left to leave the intermediate state : ", next_itm)
			control_command = ustate[0]
			print("control command to be executed : ", control_command)
			query = ((abstract_state,), control_command)
			next_abstract_states = self.zero_td_mdp[query]
			print("next abstract states : ", next_abstract_states)
			next_ustate = ustate[1:] + (self.invalid_action,)
			print("next ustate : ", next_ustate)
			for st in next_abstract_states:
				next_st = st + next_ustate + (next_itm,)
				transitions.append(next_st)
		else:
			print("In a non intermediate state")
			print("Possible values of the next delay : ", list(range(min(current_td+2, self.max_td+1))))
			
			print("Looking into each next delay transition")
			for next_td in range(self.max_td+1):
				if next_td > current_td+1:
					break
				print("-------------------------------------------------")
				print("next delay candidate : ", next_td)
				if next_td <= current_td:
					next_itm = current_td - next_td
					print("next state's intermediate status : ", next_itm)
					next_ustate = ustate[:num_valid_actions] + (action,) + (self.invalid_action,)*num_invalid_actions
					control_command = next_ustate[0]
					print("control command to be executed : ", control_command)
					next_ustate = next_ustate[1:]
					query = ((abstract_state,), control_command)
					next_abstract_states = self.zero_td_mdp[query]
					print("next abstract states : ", next_abstract_states)
					print("next ustate : ", next_ustate)

					for st in next_abstract_states:
						next_st = st + next_ustate + (next_itm,)
						transitions.append(next_st)
				else:
					next_itm = 0
					print("next state's intermediate status : ", next_itm)
					next_abstract_states = [(abstract_state,)]
					next_ustate = ustate[:num_valid_actions] + (action,) + (self.invalid_action,)*(num_invalid_actions-1)
					print("no control commands to be executed")
					print("next abstract states : ", next_abstract_states)
					print("next ustate : ", next_ustate)
					
					for st in next_abstract_states:
						next_st = st + next_ustate + (next_itm,)
						transitions.append(next_st)

		print(transitions)
