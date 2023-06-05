	def transition_fn(self, state_action_pair):
		mdp_state = state_action_pair[0]
		action = state_action_pair[1]
	
		delayed_state = mdp_state[0]
		action_buffer = mdp_state[1]  
		td = mdp_state[2]
		is_itm = mdp_state[3]
	
		n = len([u for u in action_buffer if u != -1])

		next_action_buffer = list(action_buffer)

		# each block should have 
		# 1. next_delayed_states
		# 2. next_itm 
		# 3. next_action_buffer
		# 4. control_command and dynamics function call to get the next_delayed_states
		
		if is_itm:
			control_command = next_action_buffer[0] 
			next_delayed_states = self.env.forward(delayed_state, control_command)

			next_action_buffer = next_action_buffer[1:] + [-1]
			
			if len([u for u in next_action_buffer if u != -1]) > td: 
				next_itm = 1
			else:
				next_itm = 0	
		else: 
			if td > n:
				next_delayed_states = [delayed_state]
				next_action_buffer[n] = action
				next_itm = 0 
			else:
				control_command = next_action_buffer[0]
				next_delayed_states = self.env.forward(delayed_state, control_command)

				next_action_buffer += [-1]
				next_action_buffer[n] = action 
				next_action_buffer = next_action_buffer[1:]
				
				if len([u for u in next_action_buffer if u != -1]) > td: 
					next_itm = 1
				else:
					next_itm = 0

		next_td = []
		if next_itm:
			next_td = [td]
		else:
			next_td = [k for k in range(self.max_td+1)]

		next_mdp_states = list(itertools.product(next_delayed_states, [tuple(next_action_buffer)], next_td, [next_itm]))
		next_trans_prob = []
		for next_mdp_state in next_mdp_states:
			if next_mdp_state[3]:
				next_trans_prob.append(0.0)
			else:
				next_trans_prob.append(self.td_dist[next_mdp_state[2]])

		return next_mdp_states, next_trans_prob

	def generate_one_step_transitions(self, mdp_state, action):
		print("---------------------------------------------------------")
		delayed_state = mdp_state[0]
		ustate = mdp_state[1] 
		td = mdp_state[2]
		itm = mdp_state[3]

		abstract_action = action 
		#print("delayed abstract state : ", delayed_state)
		#print("abstract action state : ", abstract_action)
		
		if itm:
			num_valid_actions = len([u for u in ustate if u != -1])
			if td >= num_valid_actions:
				print("invalid state")
				return 0

		state_action_pair = (mdp_state, action)
		print("state action pair : ", state_action_pair)
							
		next_mdp_states = self.transition_fn(state_action_pair)
		print(next_mdp_states)

		next_itm = next_mdp_states[0][-1]
		if next_itm:
			next_mdp_state = random.sample(next_mdp_states, 1)[0]
			print("randomly chosen next mdp state : ", next_mdp_state)
			self.generate_one_step_transitions(next_mdp_state, action)


def generate_mdp_states(self):
		state_counter = 0
		for abstract_state in self.abstract_states:
			
			for ustate in self.ustates_list:

				for td in range(self.max_td+1):

					for itm in [0,1]:

						mdp_state = (abstract_state, ustate, td, itm)
						mdp_state_str = self.convert_mdp_state_to_str(mdp_state)
						
						if itm:
							num_valid_actions = len([u for u in ustate if u != 0])
							if td >= num_valid_actions:
								continue
						self.states[mdp_state] = state_counter
						state_counter += 1		

def generate_ustates_list(self):
		self.abstract_actions = self.abstract_actions + [0]
		self.ustates_list = list(itertools.product(self.abstract_actions, repeat=self.max_td))
		"""
		ustates_list = list(itertools.product(self.abstract_actions, repeat=self.max_td))
		ustates_list_copy = ustates_list.copy()
		for ustate in ustates_list:
			flag = False 
			bad_state = False 
			for i in range(self.max_td):
				if ustate[i] == 0:
					flag = True 
				if flag and ustate[i] != 0:
					bad_state = True 
					break 
			if bad_state:
				ustates_list_copy.remove(ustate)
		self.ustates_list = ustates_list_copy
		"""

def forward(self, abstract_state, abstract_action):
		rel_dist_idx = abstract_state[0]
		rel_vel_idx = abstract_state[1]
			
		rel_dist_tuple = self.rel_dist_tuples[rel_dist_idx]
		rel_vel_tuple = self.rel_vel_tuples[rel_vel_idx]

		min_rel_dist = rel_dist_tuple[0]
		max_rel_dist = rel_dist_tuple[1]

		min_rel_vel = rel_vel_tuple[0]
		max_rel_vel = rel_vel_tuple[1]
		
		action = self.ego_acc_list[abstract_action]

		#print(rel_dist_tuple)
		#print(min_rel_dist, max_rel_dist)
		#print(rel_vel_tuple)
		#print(min_rel_vel, max_rel_vel)
		#print(action)


		#######################################################################################
		########### function to modify acceleration ###########################################
		#######################################################################################

		def modify_acc(vel, acc, max_vel, min_vel, del_t):
			if vel >= max_vel:
				if vel + acc * del_t >= max_vel:
					acc = 0.0 
			else:
				if vel + acc * del_t >= max_vel:
					acc = (max_vel - vel) / del_t 
			if vel <= min_vel:
				if vel + acc * del_t <= min_vel:
					acc = 0.0 
			else:
				if vel + acc * del_t <= min_vel:
					acc = (min_vel - vel) / del_t 
			return acc

		# modifying the min_rel_acc so that the limits of min_rel_vel is maintained
		min_rel_acc = self.env_min_fv_acc - action
		#min_rel_acc = modify_acc(min_rel_vel, min_rel_acc, self.env_max_rel_vel, self.env_min_rel_vel, self.del_t)
		min_rel_dist_traveled = min_rel_vel * self.del_t + 0.5 * min_rel_acc * self.del_t ** 2

		# modifying the max_fv_acc so that the limits of max_fv_vel is maintained
		max_rel_acc = self.env_max_fv_acc - action 
		#max_rel_acc = modify_acc(max_rel_vel, max_rel_acc, self.env_max_rel_vel, self.env_min_rel_vel, self.del_t)
		max_rel_dist_traveled = max_rel_vel * self.del_t + 0.5 * max_rel_acc * self.del_t ** 2

		# calculating the minimum and maximum values for the next state relative distance
		next_min_rel_dist = min_rel_dist + min_rel_dist_traveled
		next_max_rel_dist = max_rel_dist + max_rel_dist_traveled
		#print(next_min_rel_dist, next_max_rel_dist)

		# calculating the minimum and maximum values for the next state ego velocity
		next_min_rel_vel = min_rel_vel + min_rel_acc * self.del_t 
		next_max_rel_vel = max_rel_vel + max_rel_acc * self.del_t
		#print(next_min_rel_vel, next_max_rel_vel)

		###############################################################################################
		###################### Function to get indices ################################################
		###############################################################################################
		def get_indices(min_val, max_val, min_list, max_list):
			# calculate the difference list
			min_diff_list = [min_val - min_list[idx] for idx in range(len(min_list))]
			min_idx = -1 
			# if the min val is lower than the minimum most value possible, min_idx = 0
			if min_val < min_list[0]:
				min_idx = 0
			# if the min val is higher than the maximum most value possible, min_idx = len(min_list)
			elif min_val > min_list[-1]:
				min_idx = len(min_list) - 1
			else:
				for idx in range(len(min_list)):
					if min_diff_list[idx] < 0:
						min_idx = idx - 1
						break 
					elif min_diff_list[idx] == 0:
						min_idx = idx
						break 
		
			max_diff_list = [max_val - max_list[idx] for idx in range(len(max_list))]
			max_idx = -1 
			if max_val < max_list[0]:
				max_idx = 0
			elif max_val > max_list[-1]:
				max_idx = len(max_list) - 1
			else:
				for idx in range(len(max_list)):
					if max_diff_list[idx] <= 0:
						max_idx = idx 
						break 
			indices = list(range(int(min_idx), int(max_idx+1)))
		
			return indices
		
		next_rel_dist_indices = get_indices(next_min_rel_dist, next_max_rel_dist, self.min_rel_dist_list, self.max_rel_dist_list) 
		next_rel_vel_indices = get_indices(next_min_rel_vel, next_max_rel_vel, self.min_rel_vel_list, self.max_rel_vel_list)
		#print(next_rel_dist_indices)
		#print(next_rel_vel_indices)
			
		transitions = []
		for next_rel_dist_idx in next_rel_dist_indices:
			for next_rel_vel_idx in next_rel_vel_indices:
				next_state = (next_rel_dist_idx, next_rel_vel_idx)
				transitions.append(next_state)
		#print(transitions)
		
		return transitions


		
						

		#mdp_loc = os.path.join("generated", "mdp_max_td_%d.npy"%(self.max_td))
		#np.save(mdp_loc, self.mdp)

		#prob_loc = os.path.join("generated", "prob_max_td_%d.npy"%(self.max_td))
		#np.save(prob_loc, self.prob)

		#states_loc = os.path.join("generated", "states_%d.npy"%(self.max_td))
		#np.save(states_loc, self.states)

		#actions_loc = os.path.join("generated", "actions.npy")
		#np.save(actions_loc, self.abstract_actions)
