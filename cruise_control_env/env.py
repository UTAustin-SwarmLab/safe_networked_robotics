import random
import numpy as np
from os import path
from scipy.interpolate import UnivariateSpline
from collections import deque

class DelayContinuousCruiseCtrlEnv():

	def __init__(self, basic_mdp, time_delay=0, constant_delay=False, delta=1.0, print_=False): 

		# environment specifications - continuous and discrete
		self.num_state_features = 2
		self.delt = 1 # 1s time step 
		self.max_episode_steps = 100
		self.fv_max_acc = 0.2  # 1m/s^2
		self.ego_max_acc = 0.5
		
		self.basic_mdp = basic_mdp
		self.ego_acc_values = np.load('actions.npy', allow_pickle=True)
		self.invalid_action = -1

		# shielding stuff
		self.max_delay = time_delay
		self.possible_delays = list(range(self.max_delay+1))
		self.constant_delay = constant_delay
		if self.constant_delay:
			shield_path = 'constant_generated/%d_td/shield_%s_prob.npy' % (self.max_delay, str(delta))  
			self.shield = np.load(shield_path, allow_pickle=True).item()
			self.td_dist = np.zeros((self.max_delay+1, self.max_delay+1))
			self.td_dist[self.max_delay][self.max_delay] = 1.0
		else:
			shield_path = 'random_generated/%d_td/shield_%s_prob.npy' % (self.max_delay, str(delta))
			self.shield = np.load(shield_path, allow_pickle=True).item()
			self.td_dist = np.array([[0.9, 0.1, 0.0, 0.0], [0.8, 0.1, 0.1, 0.0], [0.7, 0.1, 0.1, 0.1], [0.7, 0.1, 0.1, 0.1]])
			# self.td_dist = np.array([[0.25, 0.25, 0.0, 0.0], [0.25, 0.25, 0.25, 0.0], [0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]])

		# print stuff 
		self.print = print_

	###### Initialization ###################################################################

	def initializeFrontVehicleAccelerations(self, num_pts=4, N=100):
		acc_pts = np.random.uniform(-self.fv_max_acc, self.fv_max_acc, num_pts)
		#print(acc_pts)
		acc_pts = np.insert(acc_pts, 0, 0)
		acc_pts = np.append(acc_pts, 0)

		anchor_x = np.random.choice(N, num_pts, replace=False) # contains the time steps
		anchor_x.sort()
		anchor_x = anchor_x / N
		anchor_x = np.insert(anchor_x, 0, 0)
		anchor_x = np.append(anchor_x, 1.0)

		spl = UnivariateSpline(anchor_x, acc_pts)

		fv_acc = []
		for i in range(N):
			acc = spl(i/N)
			if acc > self.fv_max_acc:
				acc = self.fv_max_acc 
			if acc < -self.fv_max_acc:
				acc = -self.fv_max_acc
			fv_acc.append(acc)

		return fv_acc

	def initializeRelDist(self):
		return np.random.uniform(25,50)

	def initializeRelVel(self):
		return np.random.uniform(0,5)

	def initializeBufferstate(self):
		if self.constant_delay:
			return list(np.zeros((self.max_delay,)))
		else:
			return list([])
		
	def initializeSystemForConstantDelay(self):
		self.current_delay = self.max_delay
		self.state_buffer = deque(maxlen=self.max_delay+1)
		self.state_buffer.append(self.state)
		for ii in range(self.max_delay):
			self.state = self.dynamics(self.state, 0, self.fv_acc_list[ii]) # 0 is the stay action
			self.state_buffer.append(self.state)
			self.episode_steps += 1

	def initializeSystemForRandomDelay(self):
		self.current_delay = 0
		self.state_buffer = deque(maxlen=self.max_delay+1)
		self.state_buffer.append(self.state)

	def initializeEnvironmentVariables(self):
		self.episode_steps = 0
		self.done = False

		"""
		### Initialization
		The state is always in continuous space, including the MDP state
		"""
		self.fv_acc_list = self.initializeFrontVehicleAccelerations(num_pts=4, N=200) # initializing the front vehicle accelerations

		self.rel_init_dist = self.initializeRelDist()
		self.rel_init_vel = self.initializeRelVel()
		buffer_state = self.initializeBufferstate() # the buffer state contains continuous action values
		self.networked_mdp_state = [self.rel_init_dist, self.rel_init_vel] + buffer_state # initializing the mdp state - the delayed state (relative distance, relative velocity) + the buffer state

		self.state = [self.rel_init_dist, self.rel_init_vel] # the actual state
		if self.constant_delay:
			self.initializeSystemForConstantDelay() # if constant delay, we evolve the system by a constant delay number of time steps considering the ego vehicle does not move during that time period
		else:
			self.initializeSystemForRandomDelay()

	def reset(self, seed=0):
		np.random.seed(seed)
		self.initializeEnvironmentVariables()
		if self.print:
			print("beginning a new episode")
			print(self.networked_mdp_state, self.state_buffer)
		obs = self.networked_mdp_state[:self.num_state_features].copy()

		return obs 
	
	def dynamics(self, state, ego_acc_value, fv_acc_value):
		rel_pos_value = state[0]
		rel_vel_value = state[1]
		rel_acc_value = fv_acc_value - ego_acc_value	
		rel_pos_value = rel_pos_value + rel_vel_value * self.delt + 0.5 * rel_acc_value * self.delt**2
		rel_vel_value = rel_vel_value + rel_acc_value * self.delt
		return [rel_pos_value, rel_vel_value]
	
	############ discretization ##########################################################
	def convertConstantDelayContinuousBufferStateToDiscrete(self, continuous_buffer_state):
		discrete_buffer_state = []
		num_valid_actions = len(continuous_buffer_state)
		assert num_valid_actions == self.max_delay

		for td_idx in range(num_valid_actions):
			continuous_value = continuous_buffer_state[td_idx]
			assert continuous_value <= self.ego_max_acc and continuous_value >= -self.ego_max_acc
			discrete_value = np.where(self.ego_acc_values >= continuous_value)[0][0]
			# if continuous_value <= self.ego_acc_values[-1]:
			# 	discrete_value = np.where(self.ego_acc_values >= continuous_value)[0][0]
			# else:
			# 	discrete_value = len(self.ego_acc_values)-1
			discrete_buffer_state.append(discrete_value)
		
		assert len(discrete_buffer_state) == self.max_delay
		return discrete_buffer_state		

	def convertRandomDelayContinuousBufferStateToDiscrete(self, continuous_buffer_state):
		num_valid_actions = len(continuous_buffer_state)
		discrete_buffer_state = []
		for td_idx in range(num_valid_actions):
			continuous_value = continuous_buffer_state[td_idx]	
			assert continuous_value <= self.ego_max_acc and continuous_value >= -self.ego_max_acc
			discrete_value = np.where(self.ego_acc_values >= continuous_value)[0][0]
			discrete_buffer_state.append(discrete_value)

		assert len(discrete_buffer_state) == len(continuous_buffer_state)
		num_invalid_actions_required = self.max_delay - len(discrete_buffer_state)		
		discrete_buffer_state = discrete_buffer_state + [self.invalid_action,] * num_invalid_actions_required
		
		assert len(discrete_buffer_state) == self.max_delay
		return discrete_buffer_state

	def convertContinuousBufferStateToDiscrete(self, continuous_buffer_state):
		if self.constant_delay:
			return self.convertConstantDelayContinuousBufferStateToDiscrete(continuous_buffer_state)
		else:
			return self.convertRandomDelayContinuousBufferStateToDiscrete(continuous_buffer_state)

	def convertContinuousNetworkedMDPStateToDiscrete(self, continuous_networked_mdp_state):
		continuous_relative_distance = continuous_networked_mdp_state[0]
		continuous_relative_velocity = continuous_networked_mdp_state[1]

		# converting continuous relative distance to discrete relative distance
		rel_dist_idx = np.digitize(continuous_relative_distance, self.basic_mdp.rel_dist_list)
		if rel_dist_idx != 0:
			rel_dist_idx -= 1

		# converting continuous relative velocity to discrete relative velocity
		rel_vel_idx = np.digitize(continuous_relative_velocity, self.basic_mdp.rel_vel_list)
		if rel_vel_idx != 0:
			rel_vel_idx -= 1
		
		# combining discrete relative distance and discrete relative velocity to obtain the discrete basic mdp state
		basic_mdp_state = [rel_dist_idx * len(self.basic_mdp.rel_vel_list) + rel_vel_idx,]
		
		# obtaining the discrete buffer state from the continuous buffer state
		continuous_buffer_state = continuous_networked_mdp_state[2:]
		discrete_buffer_state = self.convertContinuousBufferStateToDiscrete(continuous_buffer_state)

		# combining the dicrete basic mdp state and the discrete buffer state to obtain the discrete networked mdp state.
		# will contain intermediate variable for the random delay case
		if self.constant_delay:
			discrete_networked_mdp_state = basic_mdp_state + discrete_buffer_state
			assert len(discrete_networked_mdp_state) == self.max_delay + 1
		else:
			discrete_networked_mdp_state = basic_mdp_state + discrete_buffer_state + [0,]
			assert len(discrete_networked_mdp_state) == self.max_delay + 2

		return discrete_networked_mdp_state
	
	def getShield(self, discrete_networked_mdp_state):
		discrete_networked_mdp_state = tuple(discrete_networked_mdp_state)
		safe_actions_indices = self.shield[discrete_networked_mdp_state]
		safe_actions = self.ego_acc_values[safe_actions_indices]
		return safe_actions
	
	def processAction(self, action):
		action = action[0]
		if action > self.ego_max_acc:
			action = self.ego_max_acc
		if action < -self.ego_max_acc:
			action = -self.ego_max_acc
		return action

	def step(self, action):
		action = self.processAction(action)

		if self.print:
			print("-------------------------------------------")
			print("time step : ", self.episode_steps)
			print("current system state : ", self.state[0], self.state[1])
		assert self.state == self.state_buffer[-1]

		assert len(self.state) == 2 # [relative_distance, relative_velocity]
		assert len(self.networked_mdp_state) == len(self.state) + self.current_delay # [relative distance, relative velocity] + buffer state

		current_buffer_state = self.networked_mdp_state[2:] 
		assert len(current_buffer_state) == self.current_delay

		num_valid_actions = len(current_buffer_state) 
		assert num_valid_actions == self.current_delay
		assert self.networked_mdp_state[:self.num_state_features] == self.state_buffer[-1-self.current_delay]

		discrete_networked_mdp_state = self.convertContinuousNetworkedMDPStateToDiscrete(self.networked_mdp_state)
		if self.print:
			print("current networked mdp continuous state : ", self.networked_mdp_state)
			print("current networked mdp discrete state : ", discrete_networked_mdp_state)

		allowed_actions = self.getShield(discrete_networked_mdp_state) # allowed ego acceleration values
		max_allowed_action = max(allowed_actions)
		
		shielded_action = action
		if shielded_action >= max_allowed_action:  
			shielded_action = max_allowed_action  

		ego_acc_val = shielded_action
		fv_acc_val = self.fv_acc_list[self.episode_steps]

		if self.print:
			print("allowed actions", allowed_actions, "unfiltered action : %f, filtered action : %f" % (action, shielded_action), self.state, fv_acc_val, ego_acc_val)

		### State update
		self.state = self.dynamics(self.state, ego_acc_val, fv_acc_val)
		self.state_buffer.append(self.state)

		### get a random value for the next delay
		next_delay = random.choices(self.possible_delays, self.td_dist[self.current_delay])[0]

		### check for the correctness in transitions for delay
		if self.constant_delay:
			assert next_delay == self.current_delay
		else:
			assert next_delay <= self.current_delay + 1

		### update the ustate accordingly
		next_buffer_state = current_buffer_state + [ego_acc_val,]
		if next_delay == 0:
			next_buffer_state = list([])
		else:
			next_buffer_state = next_buffer_state[-next_delay:]

		### update the mdp_state accordingly
		self.networked_mdp_state = self.state_buffer[-1-next_delay] + next_buffer_state

		if self.print:
			print("state buffer : ", self.state_buffer)
			print("current delay : ", num_valid_actions, "next delay : ", next_delay)
			print("next agent state : ", self.state_buffer[-1], self.state)
			print("next mdp continuous state : ", self.networked_mdp_state)
		
		obs = self.networked_mdp_state[:self.num_state_features]
		self.episode_steps += 1
		self.current_delay = next_delay
		
		### Terminating the episode
		if self.state[0] < 0 or self.episode_steps >= self.max_episode_steps:
			self.done = True 

		info = {'dis_rem': self.state[0]}

		return obs, 0, self.done, info

	