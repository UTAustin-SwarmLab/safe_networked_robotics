import sys
import os
import itertools
import numpy as np 

class NetworkedMDPConstantDelay:
	def __init__(self, basic_mdp, basic_mdp_states, num_actions, td, log_dir=None):
		self.basic_mdp = basic_mdp 
		self.basic_mdp_states = basic_mdp_states 
		self.num_actions = num_actions 
		self.td = td 
		self.log_dir = log_dir

		self.buffer_states = self.generate_buffer_states()
		# print(self.buffer_states)

		self.mdp = {}
		self.mdp_states = {}

	def generate_buffer_states(self):
		return list(itertools.product(list(range(self.num_actions)), repeat=self.td)) 
	
	def generate_networked_mdp(self):
		state_counter = 0
		for basic_mdp_state in self.basic_mdp_states:
			robot_turn = basic_mdp_state[0]%2

			for buffer_state in self.buffer_states:
				state = basic_mdp_state + buffer_state
				print("networked mdp state : ", state) 
				self.mdp_states[state_counter] = state 
				state_counter += 1

				for action in range(num_actions):
					state_action_pair = (state, action)
					
					query = (basic_mdp_state, buffer_state[0])
					assert query == ((state[0],), state[1])

					next_basic_states = self.basic_mdp[query]
					if robot_turn:
						next_buffer_state = buffer_state[1:] + (action,)
					else:
						next_buffer_state = buffer_state
					
					transitions = [next_basic_states[idx] + next_buffer_state for idx in range(len(next_basic_states))]
					# print(state_action_pair, query, transitions)
					
					self.mdp[state_action_pair] = transitions

	def save_mdp(self):
		os.makedirs(self.log_dir, exist_ok=True)

		mdp_states_loc = os.path.join(self.log_dir, 'states_%d_td' % self.td)		
		np.save(mdp_states_loc, self.mdp_states)

		mdp_loc = os.path.join(self.log_dir, 'mdp_%d_td' % self.td)
		np.save(mdp_loc, self.mdp)
		# print(self.mdp)

if __name__ == "__main__":
	basic_mdp = np.load('constant_generated/mdp_0_td.npy', allow_pickle=True).item()
	basic_mdp_states = list(np.load('constant_generated/states_0_td.npy', allow_pickle=True).item().values())
	actions_dict = np.load('actions.npy', allow_pickle=True).item()
	actions_list = list(actions_dict.keys())
	num_actions = len(actions_list)
	
	td = int(sys.argv[1])
	log_dir = 'constant_generated'
	
	networked_mdp_obj = NetworkedMDPConstantDelay(basic_mdp, basic_mdp_states, num_actions, td, log_dir)
	networked_mdp_obj.generate_networked_mdp()
	networked_mdp_obj.save_mdp()