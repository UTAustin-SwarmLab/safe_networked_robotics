import os
import numpy as np 

class BasicMDP:
	def __init__(self, log_dir=None):
		self.log_dir = log_dir

		self.xmin = 0
		self.xmax = 7
		self.ymin = 0  
		self.ymax = 7
		self.num_xbins = int(self.xmax-self.xmin+1)

		self.rx_init_min = 0
		self.rx_init_max = 1
		
		self.ry_init_min = 0
		self.ry_init_max = 1

		self.ax_init_min = 4
		self.ax_init_max = 6

		self.ay_init_min = 4
		self.ay_init_max = 6

		self.goalx = self.xmax 
		self.goaly = self.ymax

		self.actions_dict = {0:'stay', 1:'up', 2:'right', 3:'down', 4:'left'}
		self.actions_list = list(self.actions_dict.keys())

		self.mdp = {}
		self.mdp_states = {}
		self.mdp_unsafe_states = {}
		self.mdp_initial_states = {}
		self.mdp_goal_states = {}

	def convert_physical_state_to_int(self, physical_state):
		increments = [(self.num_xbins**3)*2, (self.num_xbins**2)*2, self.num_xbins*2, 2, 1]
		return np.sum(np.multiply(list(physical_state), increments))

	def is_unsafe(self, physical_state):
		rx = physical_state[0]
		ry = physical_state[1]
		ax = physical_state[2]
		ay = physical_state[3]

		if rx == ax and ry == ay:
			return 1.0
		else:
			return 0.0

	def is_init(self, physical_state):
		rx = physical_state[0]
		ry = physical_state[1]
		ax = physical_state[2]
		ay = physical_state[3]

		rx_init_flag = rx >= self.rx_init_min and rx <= self.rx_init_max
		ry_init_flag = ry >= self.ry_init_min and ry <= self.ry_init_max
		ax_init_flag = ax >= self.ax_init_min and ax <= self.ax_init_max
		ay_init_flag = ay >= self.ay_init_min and ay <= self.ay_init_max

		if rx_init_flag and ry_init_flag and ax_init_flag and ay_init_flag:
			return 1.0 
		else:
			return 0.0
		
	def is_goal(self, physical_state):
		rx = physical_state[0]
		ry = physical_state[1]
		ax = physical_state[2]
		ay = physical_state[3]

		if rx == self.goalx and ry == self.goaly and (ax !=self.goalx or ay != self.goaly):
			return 1.0
		else:
			return 0.0
		
	def robot_transition(self, physical_state, action):
		rx = physical_state[0]
		ry = physical_state[1]
		ax = physical_state[2]
		ay = physical_state[3]

		if action == 'stay': 
			next_rx = rx 
			next_ry = ry 

		if action == 'up':
			next_rx = rx
			next_ry = min(ry + 1, self.ymax)

		if action == 'down':
			next_rx = rx
			next_ry = max(ry - 1, self.ymin)

		if action == 'left':
			next_rx = max(rx - 1, self.xmin)
			next_ry = ry 

		if action == 'right':
			next_rx = min(rx + 1, self.xmax)
			next_ry = ry 

		next_ax = ax
		next_ay = ay

		next_states = [(next_rx, next_ry, next_ax, next_ay, 0)]
		return next_states
	
	def adversary_transition(self, physical_state):
		rx = physical_state[0]
		ry = physical_state[1]
		ax = physical_state[2]
		ay = physical_state[3]

		next_rx = rx 
		next_ry = ry

		next_states = []
		for act in self.actions_list:
			action = self.actions_dict[act]
		
			if action == 'stay':
				continue

			if action == 'up':
				next_ax = ax
				next_ay = min(ay + 1, self.ymax)

			if action == 'down':
				next_ax = ax
				next_ay = max(ay - 1, self.ymin)

			if action == 'left':
				next_ax = max(ax - 1, self.xmin)
				next_ay = ay  

			if action == 'right':
				next_ax = min(ax + 1, self.xmax)
				next_ay = ay

			next_states.append((next_rx, next_ry, next_ax, next_ay, 1))
		
		return next_states
		
	def dynamics(self, physical_state, act):
		flag = physical_state[-1]
		action = self.actions_dict[act]
		if flag:
			return self.robot_transition(physical_state, action)
		else:
			return self.adversary_transition(physical_state)
			
	def generate_basic_mdp(self):
		state_counter = 0
		for rx in range(self.xmin, self.xmax+1):
			for ry in range(self.ymin, self.ymax+1): 
				for ax in range(self.xmin, self.xmax+1):
					for ay in range(self.xmin, self.xmax+1):
						for flag in [0, 1]:
							physical_state = (rx, ry, ax, ay, flag)
							state_id = self.convert_physical_state_to_int(physical_state)
							state = (state_id,)
							print("state id : ", state_id, "state : ", state)
							self.mdp_states[state_counter] = state
							state_counter += 1

							self.mdp_unsafe_states[state] = self.is_unsafe(physical_state)
							self.mdp_initial_states[state] = self.is_init(physical_state)
							self.mdp_goal_states[state] = self.is_goal(physical_state)

							for act in self.actions_list:
								state_action_pair = (state, act)
								next_physical_states = self.dynamics(physical_state, act)
								next_states = [(self.convert_physical_state_to_int(next_physical_state),) for next_physical_state in next_physical_states]
								print(state_action_pair, physical_state, act, next_physical_states, next_states)
								self.mdp[state_action_pair] = next_states

	def save_mdp(self):
		os.makedirs(self.log_dir, exist_ok=True)

		mdp_states_loc = os.path.join(self.log_dir, 'states_0_td')		
		np.save(mdp_states_loc, self.mdp_states)

		mdp_initial_states_loc = os.path.join(self.log_dir, 'initial_0_td')		
		np.save(mdp_initial_states_loc, self.mdp_initial_states)

		mdp_unsafe_states_loc = os.path.join(self.log_dir, 'unsafe_0_td')		
		np.save(mdp_unsafe_states_loc, self.mdp_unsafe_states)

		mdp_goal_states_loc = os.path.join(self.log_dir, 'goal_0_td')		
		np.save(mdp_goal_states_loc, self.mdp_goal_states)

		actions_loc = 'actions'
		np.save(actions_loc, self.actions_dict)

		mdp_loc = os.path.join(self.log_dir, 'mdp_0_td')
		np.save(mdp_loc, self.mdp)
		# print(self.mdp_states)

if __name__ == "__main__":
	log_dir = 'constant_generated'
	basic_mdp_obj = BasicMDP(log_dir)
	basic_mdp_obj.generate_basic_mdp()
	basic_mdp_obj.save_mdp()

