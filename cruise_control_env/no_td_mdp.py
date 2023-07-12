import os
import numpy as np 
from collections import Counter

class BasicMDP:
	def __init__(self, log_dir=None):
		self.log_dir = log_dir
		self.del_t = 1.0

		self.min_rel_dist = 5
		self.max_rel_dist = 25 
		self.del_rel_dist = 1.0

		self.rel_dist_list = [0,] # this is like the sink state
		for i in range(int((self.max_rel_dist - self.min_rel_dist) / self.del_rel_dist + 1)):
			self.rel_dist_list.append(self.min_rel_dist + i * self.del_rel_dist)
		self.num_rel_dist_indices = len(self.rel_dist_list)

		self.min_rel_vel = -5 
		self.max_rel_vel = 5 
		self.del_rel_vel = 0.5

		self.rel_vel_list = [-30,] # sink state
		for i in range(int((self.max_rel_vel - self.min_rel_vel) / self.del_rel_vel + 1)):
			self.rel_vel_list.append(self.min_rel_vel + i * self.del_rel_vel)
		self.num_rel_vel_indices = len(self.rel_vel_list)

		self.env_min_fv_acc = -0.2
		self.env_max_fv_acc = 0.2
		self.fv_acc_list = [self.env_min_fv_acc + ((self.env_max_fv_acc - self.env_min_fv_acc) / 501) * i for i in range(501 + 1)]

		self.ego_acc_values = [-0.5, -0.25, 0.0, 0.25, 0.5]
		self.num_actions = len(self.ego_acc_values)

		# print(self.rel_dist_list)
		# print(self.rel_vel_list)
		# print(self.fv_acc_list)
		# print(self.ego_acc_values)

		self.unsafe_dist = 5.0 
		self.initial_dist = 25
		self.initial_vel = 0.0

		self.mdp = {}
		self.mdp_states = {}
		self.mdp_unsafe_states = {}
		self.mdp_initial_states = {}

	def convert_physical_state_to_int(self, physical_state):
		assert len(physical_state) == 2 # rel_dist, rel_vel
		rel_dist_idx = physical_state[0] 
		rel_vel_idx = physical_state[1]
		id = rel_dist_idx * self.num_rel_vel_indices + rel_vel_idx
		return id

	def is_unsafe(self, physical_state):
		assert len(physical_state) == 2
		state_rel_dist = self.rel_dist_list[physical_state[0]]
		if state_rel_dist <= self.unsafe_dist:
			return 1.0
		else:
			return 0.0

	def is_init(self, physical_state):
		state_rel_dist = self.rel_dist_list[physical_state[0]]
		state_rel_vel = self.rel_vel_list[physical_state[1]]
		if state_rel_dist >= self.initial_dist and state_rel_vel >= self.initial_vel:
			return 1.0
		else:
			return 0.0
		
	def dynamics(self, physical_state, action):
		rel_dist_val = self.rel_dist_list[physical_state[0]]
		rel_vel_val = self.rel_vel_list[physical_state[1]]
		ego_acc_val = self.ego_acc_values[action]
		# print("----------------------------------------------------------------------------------")
		# print(physical_state[0], physical_state[1], rel_dist_val, rel_vel_val, action, ego_acc_val)

		next_rel_dist_values = []
		next_rel_vel_values = []
		for fv_acc_val in self.fv_acc_list:
			rel_acc = fv_acc_val - ego_acc_val 
			next_rel_dist_val = rel_dist_val + rel_vel_val * self.del_t + 0.5 * rel_acc * self.del_t ** 2 
			# similar to (fv_dist_val + fv_vel_val * self.del_t + 0.5 * fv_acc_val *self.del_t ** 2) - (ego_dist_val + ego_vel_val * self.del_t + 0.5 * ego_acc_val * self.del_t ** 2)
			next_rel_vel_val = rel_vel_val + rel_acc * self.del_t 
			# similar to (fv_vel_val + fv_acc_val * self.del_t) - (ego_vel_val + ego_acc_val * self.del_t)
			next_rel_dist_values.append(next_rel_dist_val)
			next_rel_vel_values.append(next_rel_vel_val)

		next_rel_dist_idxs = np.digitize(next_rel_dist_values, self.rel_dist_list)
		# print(next_rel_dist_values)
		# print(next_rel_dist_idxs)
		for rdidx in range(next_rel_dist_idxs.shape[0]):
			if next_rel_dist_idxs[rdidx] == 0:
				continue
			else:
				next_rel_dist_idxs[rdidx] -= 1
		# print(next_rel_dist_idxs, [self.rel_dist_list[kdx] for kdx in next_rel_dist_idxs])
		

		next_rel_vel_idxs = np.digitize(next_rel_vel_values, self.rel_vel_list)
		# print(next_rel_vel_values)
		# print(next_rel_vel_idxs)
		for rvidx in range(next_rel_vel_idxs.shape[0]):
			if next_rel_vel_idxs[rvidx] == 0:
				continue
			else:
				next_rel_vel_idxs[rvidx] -= 1
		# print(next_rel_vel_idxs, [self.rel_vel_list[kdx] for kdx in next_rel_vel_idxs])

		next_state_ids = []
		# print(next_rel_dist_idxs)
		# print(next_rel_vel_idxs)
		for next_rel_dist_idx, next_rel_vel_idx in zip(next_rel_dist_idxs, next_rel_vel_idxs):
			# print(next_rel_dist_idx, next_rel_vel_idx)
			next_physical_state = (next_rel_dist_idx, next_rel_vel_idx)
			next_state_id = self.convert_physical_state_to_int(next_physical_state)
			next_state_ids.append(next_state_id) 

		# print(next_state_ids)
		transitions_counter = Counter(next_state_ids)
		unique_ids = list(transitions_counter.keys())
		occurrences = list(transitions_counter.values())
		# print(unique_ids, occurrences)

		total_transitions = len(next_state_ids)
		unique_states = [(unique_ids[unqidx],) for unqidx in range(len(unique_ids))]
		probabilities = [occurrences[occidx]/total_transitions for occidx in range(len(occurrences))]
		# print(unique_states, probabilities)	
		assert round(sum(probabilities), 2) == 1.0 
		return unique_states, probabilities 
	
	def generate_basic_mdp(self):
		state_counter = 0
		for rel_dist_idx in range(self.num_rel_dist_indices):
			for rel_vel_idx in range(self.num_rel_vel_indices):
				physical_state = (rel_dist_idx, rel_vel_idx)
				state_id = self.convert_physical_state_to_int(physical_state)
				state = (state_id,)
				print('state : ', state)
				self.mdp_states[state_counter] = state 
				state_counter += 1 

				self.mdp_unsafe_states[state] = self.is_unsafe(physical_state)
				self.mdp_initial_states[state] = self.is_init(physical_state)

				# print(rel_dist_idx, rel_vel_idx, self.mdp_unsafe_states[state], self.mdp_initial_states[state])

				for action in range(self.num_actions):
					state_action_pair = (state, action)
					# print(state_action_pair, physical_state, action)
					next_states, next_states_probabilities = self.dynamics(physical_state, action)

					self.mdp[state_action_pair] = (next_states, next_states_probabilities) 

	def save_mdp(self):
		os.makedirs(self.log_dir, exist_ok=True)

		mdp_states_loc = os.path.join(self.log_dir, 'states_0_td')		
		np.save(mdp_states_loc, self.mdp_states)

		mdp_initial_states_loc = os.path.join(self.log_dir, 'initial_0_td')		
		np.save(mdp_initial_states_loc, self.mdp_initial_states)

		mdp_unsafe_states_loc = os.path.join(self.log_dir, 'unsafe_0_td')		
		np.save(mdp_unsafe_states_loc, self.mdp_unsafe_states)

		actions_loc = 'actions'
		np.save(actions_loc, self.ego_acc_values)

		mdp_loc = os.path.join(self.log_dir, 'mdp_0_td')
		np.save(mdp_loc, self.mdp)
		# print(self.mdp)

if __name__ == "__main__":
	log_dir = 'constant_generated'
	basic_mdp_obj = BasicMDP(log_dir)
	basic_mdp_obj.generate_basic_mdp()
	basic_mdp_obj.save_mdp()