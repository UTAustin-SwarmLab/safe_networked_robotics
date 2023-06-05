import os 
import csv
import random 
import itertools
import numpy as np  

class TimeDelayedMDP:
	def __init__(self, env, max_td, td_dist):
		self.env = env
		self.abstract_states = env.abstract_states  
		self.abstract_actions = env.abstract_actions # 0,1,2,3,4
		self.num_actions = len(self.abstract_actions) #5
		self.invalid_action = 0
		self.initial_action = self.invalid_action
		self.max_td = max_td
		self.zero_td_mdp = env.zero_td_mdp
		self.td_dist = td_dist

		self.ustates_list = []
		self.states = {} 
		self.mdp = {} 
		self.prob = {}
		
		self.generate_ustates_list()
		self.num_ustates = len(self.ustates_list)
		# print(self.ustates_list)
		# print(len(self.ustates_list))
		# print(self.convert_state_to_int((219, 4, 2, 3, 3)))

		self.get_one_step_transitions()
		# self.generate_one_step_transitions((201, -1, -1, -1, 0), 3)
		#self.generate_one_step_transitions((201, 3, 4, 1, 0), 3)
	
	def generate_ustates_list(self):
		self.ustates_list = list(itertools.product(self.abstract_actions, repeat=self.max_td))

	def convert_state_to_int(self, state):
		# print(state)
		increments = [1,] 
		increments += [(self.max_td+1)*self.num_actions**k for k in range(self.max_td)]
		increments += [(self.max_td+1)*self.num_ustates,]
		increments.reverse()
		# print(increments)
		return np.sum(np.multiply(list(state), increments))

	def get_one_step_transitions(self):
		"""
		A random td state has 3 components
		1. The abstract env state 
		2. Action buffer
		3. whether the state is intermediate or not 
		"""

		mdp_unsafe_states = []
		mdp_initial_states = []
		mdp_invalid_states = []

		for abstract_state in self.abstract_states:
			print("abstract state : ", abstract_state)

			for ustate in self.ustates_list:
				num_valid_actions = len([u for u in ustate if u != self.invalid_action])
				num_invalid_actions = self.max_td - num_valid_actions
				
				# invalid (self.invalid_action (5)) actions occur only at the end, not inbetween valid actions 
				# if self.invalid_action in ustate[:num_valid_actions]:
					# continue 
				
				current_td = num_valid_actions
		
				for itm in range(self.max_td+1):
					state = (abstract_state,) + ustate + (itm,)
					state_id = self.convert_state_to_int(state)
					
					# identify if the state is invalid 
					invalid_state = self.invalid_action in ustate[:num_valid_actions] or itm > num_valid_actions

					if invalid_state:
						mdp_invalid_states.append(1.0)
					else:
						mdp_invalid_states.append(0.0)

					if self.env.zero_td_unsafe_states[abstract_state] and not invalid_state:
						mdp_unsafe_states.append(1.0)
					else:
						mdp_unsafe_states.append(0.0) 

					if self.env.zero_td_initial_states[abstract_state] and list(ustate) == [self.invalid_action,]*self.max_td and not invalid_state:
						mdp_initial_states.append(1.0)
					else:
						mdp_initial_states.append(0.0)

					if invalid_state:
						continue

					for action in self.abstract_actions:
						if action == self.invalid_action:
							continue

						state_action_pair = (state_id, action)
						# print(state_action_pair)

						transitions = []
						probabilities = []
							
						if itm > 0:
							# print("here, valid state action pair, in intermediate state")
							next_itm = itm-1
							control_command = ustate[0]-1 # because we added one action in front of allowable legal actions (the invalid action)
							# print(control_command)
							query = (abstract_state, control_command)
							# print(query)
							next_physical_states = self.zero_td_mdp[query]
							# print(next_physical_states)
							next_ustate = ustate[1:] + (self.invalid_action,)
							for st in next_physical_states:
								next_st = (st,) + next_ustate + (next_itm,)
								next_state_id = self.convert_state_to_int(next_st) 
								transitions.append(next_state_id)
								probabilities.append(1/len(next_physical_states))
								
						else:
							# print("here, valid state action pair, in non intermediate state")
							for next_td in range(self.max_td+1):
								if next_td > current_td+1:
									break
								if next_td <= current_td:
									next_itm = current_td - next_td
									next_ustate = ustate[:num_valid_actions] + (action,) + (self.invalid_action,)*num_invalid_actions
									control_command = next_ustate[0]-1 # because we added one action in front of allowable legal actions (the invalid action)
									next_ustate = next_ustate[1:]
									query = (abstract_state, control_command)
									next_abstract_states = self.zero_td_mdp[query]
									for st in next_abstract_states:
										next_st = (st,) + next_ustate + (next_itm,)
										# print(next_st)
										next_state_id = self.convert_state_to_int(next_st)
										transitions.append(next_state_id)
										probabilities.append(self.td_dist[current_td, next_td] / len(next_abstract_states))
								else:
									next_itm = 0
									next_abstract_states = [abstract_state,]
									next_ustate = ustate[:num_valid_actions] + (action,) + (self.invalid_action,)*(num_invalid_actions-1)
									for st in next_abstract_states:
										next_st = (st,) + next_ustate + (next_itm,)
										# print(next_st)
										next_state_id = self.convert_state_to_int(next_st)
										transitions.append(next_state_id)
										probabilities.append(self.td_dist[current_td, next_td])

						# print(transitions)
						self.mdp[state_action_pair] = transitions
						self.prob[state_action_pair] = probabilities
		
		# print(self.mdp)

		os.makedirs('random_generated', exist_ok=True)
		np.save('random_generated/random_mdp_%d_td' % self.max_td, self.mdp)
		np.save('random_generated/random_mdp_prob_%d_td' % self.max_td, self.prob)
		np.save('random_generated/random_mdp_unsafe_%d_td' % self.max_td, mdp_unsafe_states)
		np.save('random_generated/random_mdp_initial_%d_td' % self.max_td, mdp_initial_states)
		np.save('random_generated/random_mdp_invalid_%d_td' % self.max_td, mdp_invalid_states)

	def generate_one_step_transitions(self, state, action):
		abstract_state = state[0]
		ustate = state[1:-1]
		itm = state[-1]

		print("state : ", state)
		print("abstract state : ", abstract_state)
		print("ustate : ", ustate)
		print("itermediate status : ", itm)
		print("action : ", action)

		num_valid_actions = len([u for u in ustate if u != self.invalid_action]) # the current value of time delay
		num_invalid_actions = self.max_td - num_valid_actions
		# print(num_valid_actions, num_invalid_actions)

		# there cannot be any invalid action inbetwen valid actions
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
			query = (abstract_state, control_command)
			next_abstract_states = self.zero_td_mdp[query]
			print("next abstract states : ", next_abstract_states)
			next_ustate = ustate[1:] + (self.invalid_action,)
			print("next ustate : ", next_ustate)
			for st in next_abstract_states:
				next_st = (st,) + next_ustate + (next_itm,)
				transitions.append(next_st)
			# print(transitions)
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
					query = (abstract_state, control_command)
					next_abstract_states = self.zero_td_mdp[query]
					print("next abstract states : ", next_abstract_states)
					print("next ustate : ", next_ustate)

					for st in next_abstract_states:
						next_st = (st,) + next_ustate + (next_itm,)
						transitions.append(next_st)
				else:
					next_itm = 0
					print("next state's intermediate status : ", next_itm)
					next_abstract_states = [abstract_state,]
					next_ustate = ustate[:num_valid_actions] + (action,) + (self.invalid_action,)*(num_invalid_actions-1)
					print("no control commands to be executed")
					print("next abstract states : ", next_abstract_states)
					print("next ustate : ", next_ustate)
					
					for st in next_abstract_states:
						next_st = (st,) + next_ustate + (next_itm,)
						transitions.append(next_st)

		print(transitions)
