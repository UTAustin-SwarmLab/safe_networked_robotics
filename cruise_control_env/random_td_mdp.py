import sys
import os
import itertools
import numpy as np 

class NetworkedMDPRandomDelay:
	def __init__(self, basic_mdp, basic_mdp_states, num_actions, td_dist, log_dir=None):
		"""
		stuff with respect to basic mdp
		"""
		self.basic_mdp = basic_mdp 
		self.basic_mdp_states = basic_mdp_states 
		self.num_actions = num_actions

		"""
		stuff with respect to time delay
		"""
		self.td_dist = td_dist  
		self.max_td = td_dist.shape[0]-1

		"""
		stuff with respect to buffer states
		"""
		self.invalid_action = -1
		self.buffer_actions = [-1,] + list(range(num_actions)) 
		self.buffer_states = self.generate_buffer_states()
		assert len(self.buffer_states) == len(self.buffer_actions) ** self.max_td

		self.log_dir = log_dir
		self.mdp_states = {}
		self.mdp_transitions = {}
		self.mdp_transition_probabilities = {}

	def generate_buffer_states(self):
		return list(itertools.product(self.buffer_actions, repeat=self.max_td))
	
	def is_invalid(self, state):
		intermediate_status = state[-1]
		buffer_state = state[1:-1]
		
		num_valid_actions_in_buffer = len([u for u in buffer_state if u != self.invalid_action])
		invalid_state = self.invalid_action in buffer_state[:num_valid_actions_in_buffer] or intermediate_status > num_valid_actions_in_buffer
		return invalid_state
	
	def update_intermediate_state(self, state):
		"""
		if an intermediate state, then the action should have no effect 
		the next inter should reduce by 1
		the buffer state should have one less valid action
		and the first action in the buffer state should act on the basic mdp state
		"""
		transitions = []
		probabilities = []

		# extracting relevant information from the state action pair
		basic_mdp_state = (state[0],)
		current_buffer_state = state[1:-1]
		inter = state[-1]
		assert inter != 0 # an intermediate state has a non zero inter value
		assert len([u for u in current_buffer_state if u != self.invalid_action]) >= inter # if this is not the case, then the state is invalid
				
		# update the intermediate status
		next_inter = inter - 1 

		# update the buffer state
		assert current_buffer_state[0] != -1
		control_command = current_buffer_state[0]
		next_buffer_state = current_buffer_state[1:] + (self.invalid_action,)

		# update the basic mdp states
		query = (basic_mdp_state, control_command)
		next_basic_mdp_states = self.basic_mdp[query][0]
		next_basic_mdp_states_probabilities = self.basic_mdp[query][1]

		for next_basic_mdp_state, next_basic_mdp_state_probability in zip(next_basic_mdp_states, next_basic_mdp_states_probabilities):
			next_state = next_basic_mdp_state + next_buffer_state + (next_inter,)
			transitions.append(next_state)
			probabilities.append(next_basic_mdp_state_probability)

		assert len(transitions) == len(next_basic_mdp_states)
		assert len(probabilities) == len(next_basic_mdp_states_probabilities)
		assert sum(probabilities) == 1.0

		return transitions, probabilities
	
	def update_non_intermediate_state_for_non_increasing_delay(self, basic_mdp_state, current_buffer_state, action, next_delay):
		"""
		if the delay has reduced, then we need to enter intermediate state
		the buffer state should be adjusted accordingly such that there are current_delay - next_delay number of actions left
		the basic mdp states should be updated by executing the first action from the buffer state on the currently available basic mdp state
		"""
		transitions = []
		probabilities = []

		num_valid_actions = len([u for u in current_buffer_state if u != self.invalid_action])
		current_delay = num_valid_actions

		# update the intermediate status
		next_inter = current_delay - next_delay

		# update the buffer state
		num_invalid_actions = self.max_td - num_valid_actions
		next_buffer_state = current_buffer_state[:num_valid_actions] + (action,) + (self.invalid_action,)*num_invalid_actions
		control_command = next_buffer_state[0]
		next_buffer_state = next_buffer_state[1:]
						
		# update the basic mdp states
		query = (basic_mdp_state, control_command)
		next_basic_mdp_states = self.basic_mdp[query][0]
		next_basic_mdp_states_probabilities = self.basic_mdp[query][1]

		for next_basic_mdp_state, next_basic_mdp_state_probability in zip(next_basic_mdp_states, next_basic_mdp_states_probabilities):
			next_state = next_basic_mdp_state + next_buffer_state + (next_inter,)
			transitions.append(next_state)
			probabilities.append(next_basic_mdp_state_probability*self.td_dist[current_delay, next_delay])

		assert len(transitions) == len(next_basic_mdp_states)
		assert len(probabilities) == len(next_basic_mdp_states_probabilities)
		assert sum(probabilities) == self.td_dist[current_delay, next_delay]

		return transitions, probabilities
	
	def update_non_intermediate_state_for_increasing_delay(self, basic_mdp_state, current_buffer_state, action, next_delay):
		"""
		if the delay has increased, it means no new observation has been received from the cloud
		so the system does not enter into an intermediate state
		the basic state remains the same
		the action get appended to the action buffer
		"""
		transitions =[]
		probabilities = []

		# update the intermediate status
		next_inter = 0

		# update the buffer state 
		num_valid_actions = len([u for u in current_buffer_state if u != self.invalid_action])
		current_delay = num_valid_actions
		num_invalid_actions = self.max_td - num_valid_actions
		next_buffer_state = current_buffer_state[:num_valid_actions] + (action,) + (self.invalid_action,)*(num_invalid_actions-1)
		assert len([u for u in next_buffer_state if u != self.invalid_action]) == num_valid_actions + 1

		# update the basic mdp states 
		next_basic_mdp_states = [basic_mdp_state,]
		next_basic_mdp_states_probabilities = [1.0,]

		for next_basic_mdp_state, next_basic_mdp_state_probability in zip(next_basic_mdp_states, next_basic_mdp_states_probabilities):
			next_state = next_basic_mdp_state + next_buffer_state + (next_inter,)
			transitions.append(next_state)
			probabilities.append(next_basic_mdp_state_probability*self.td_dist[current_delay, next_delay])

		assert len(transitions) == 1
		assert len(probabilities) == 1
		assert sum(probabilities) == self.td_dist[current_delay, next_delay]

		return transitions, probabilities

	def update_non_intermediate_state(self, state, action, next_delay):
		# retriving basic mdp state
		assert state[-1] == 0 # to make sure that the state is non intermediate 

		basic_mdp_state = (state[0],)
		current_buffer_state = state[1:-1]

		# update the intermediate status
		num_valid_actions = len([u for u in current_buffer_state if u != self.invalid_action])
		current_delay = num_valid_actions

		if next_delay < current_delay + 1: # for the case where the delay is not increasing
			return self.update_non_intermediate_state_for_non_increasing_delay(basic_mdp_state, current_buffer_state, action, next_delay)
		elif next_delay == current_delay + 1: # for the case where the delay is increasing
			return self.update_non_intermediate_state_for_increasing_delay(basic_mdp_state, current_buffer_state, action, next_delay)
		else:
			return None, None # invalid next delay
	
	def dynamics(self, state_action_pair):
		# retriving information about the intermediate status
		state = state_action_pair[0]
		current_delay = len([u for u in state[1:-1] if u != self.invalid_action])
		assert len(state) == self.max_td + 2 # buffer state size is self.max_td, and there is the basic mdp state and the intermediate state
		inter = state[-1]

		action = state_action_pair[1]

		if inter > 0:
			intermediate_state_transitions, intermediate_state_transition_probabilities = self.update_intermediate_state(state)

			assert len(intermediate_state_transitions) == len(intermediate_state_transition_probabilities)
			assert sum(intermediate_state_transition_probabilities) == 1.0

			return intermediate_state_transitions, intermediate_state_transition_probabilities
		else:
			non_intermediate_state_transitions = []
			non_intermediate_state_transition_probabilities = []
			for next_delay in range(self.max_td+1):
				per_delay_state_transitions, per_delay_state_transition_probabilities = self.update_non_intermediate_state(state, action, next_delay)
				
				if per_delay_state_transitions != None:
					assert next_delay <= current_delay + 1
					assert len(per_delay_state_transitions) == len(per_delay_state_transition_probabilities)
					assert sum(per_delay_state_transition_probabilities) == self.td_dist[current_delay, next_delay]

					non_intermediate_state_transitions += per_delay_state_transitions
					non_intermediate_state_transition_probabilities += per_delay_state_transition_probabilities

			return non_intermediate_state_transitions, non_intermediate_state_transition_probabilities

	def generate_networked_mdp(self):
		state_count = 0
		for basic_mdp_state in self.basic_mdp_states:
			for buffer_state in self.buffer_states:
				for itm in range(self.max_td+1):
					state = basic_mdp_state + buffer_state + (itm,)

					invalid_state = self.is_invalid(state)
					if invalid_state:
						continue 

					print("networked mdp state : ", state) 
					self.mdp_states[state_count] = state 
					state_count += 1

					for action in range(self.num_actions):
						state_action_pair = (state, action)
						state_transitions, state_transition_probabilities = self.dynamics(state_action_pair)
						self.mdp_transitions[state_action_pair] = state_transitions
						self.mdp_transition_probabilities[state_action_pair] = state_transition_probabilities

	def save_mdp(self):
		os.makedirs(self.log_dir, exist_ok=True)

		mdp_states_loc = os.path.join(self.log_dir, 'states_%d_td' % self.max_td)		
		np.save(mdp_states_loc, self.mdp_states)

		mdp_transitions_loc = os.path.join(self.log_dir, 'mdp_transitions_%d_td' % self.max_td)
		np.save(mdp_transitions_loc, self.mdp_transitions)

		mdp_transition_probabilities_loc = os.path.join(self.log_dir, 'mdp_probabilities_%d_td' % self.max_td)
		np.save(mdp_transition_probabilities_loc, self.mdp_transition_probabilities)

		# print(self.mdp)

if __name__ == "__main__":
	basic_mdp = np.load('constant_generated/mdp_0_td.npy', allow_pickle=True).item() # for querying
	basic_mdp_states = list(np.load('constant_generated/states_0_td.npy', allow_pickle=True).item().values()) # for basic mdp states
	ego_acc_list = np.load('actions.npy', allow_pickle=True)
	num_actions = len(ego_acc_list)
	
	td_dist = np.array([[0.9, 0.1, 0.0, 0.0], [0.8, 0.1, 0.1, 0.0], [0.7, 0.1, 0.1, 0.1], [0.7, 0.1, 0.1, 0.1]])
	# td_dist = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
	log_dir = 'random_generated'
	
	networked_mdp_obj = NetworkedMDPRandomDelay(basic_mdp, basic_mdp_states, num_actions, td_dist, log_dir)
	networked_mdp_obj.generate_networked_mdp()
	networked_mdp_obj.save_mdp()
	# state = (211, 1, 1, -1, 2)
	# print(networked_mdp_obj.is_invalid(state))
	# action = 2
	# state_action_pair = (state, action)
	# print(networked_mdp_obj.dynamics(state_action_pair))