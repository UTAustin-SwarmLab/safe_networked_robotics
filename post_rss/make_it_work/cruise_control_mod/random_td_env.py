import gym 
import math
import numpy as np
from os import path
from random import choices 
from collections import deque
from scipy.interpolate import UnivariateSpline

class RandTdContinuousCruiseCtrlEnv(gym.Env):

	def __init__(self, time_delay=0, train=True, constant_td=False, log_file_str=None, delta=1.0): 
		"""
		### Action Space
			The action is a scalar in the range `[-1, 1]` that multiplies the max_acc
			to give the acceleration of the ego vehicle. 
		"""

		self.action_low = -1.0 
		self.action_high = 1.0
		self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,))
		 
		"""
		### Observation Space
			The observation is an ndarray of shape (2,) with each element in the range
			`[-inf, inf]`.   
		"""
		self.num_state_features = 2
		self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_state_features,))

		"""
		### Episodic Task
		"""
		self.max_episode_steps = 100
		self.episode_steps = 0
		self.done = False

		"""
		### Environment Specifications   
		"""
		self.safety_dist = 5	# Required distance between the ego and front vehicle
		self.violating_safety_dist_reward = -10	# Reward for getting too close to the front car 
		self.delt = 1 # 1s time step 
		self.ego_max_acc = 0.25
		self.ego_min_acc = -0.5
		self.fv_max_acc = 0.1

		"""
		shielding stuff
		"""
		self.constant_td = constant_td
		self.max_td = time_delay
		if self.constant_td:
			shield_path = 'constant_generated/%d_td/shield_%s_prob.npy' % (self.max_td, str(delta))
			self.shield = np.load(shield_path, allow_pickle=True)
			self.td_dist = np.zeros((self.max_td+1, self.max_td+1))
			self.td_dist[self.max_td][self.max_td] = 1.0
		else:
			shield_path = 'random_generated/%d_td/shield_%s_prob.npy' % (self.max_td, str(delta))
			self.shield = np.load(shield_path, allow_pickle=True)
			self.td_dist = np.array([[0.9, 0.1, 0.0, 0.0], [0.8, 0.1, 0.1, 0.0], [0.7, 0.1, 0.1, 0.1], [0.7, 0.1, 0.1, 0.1]])

		self.possible_delays = list(range(self.max_td+1))
		

		### relative distance abstraction
		min_rel_dist = 5 
		max_rel_dist = 25 
		del_rel_dist = 1.0  

		self.rel_dist_list = []
		for i in range(int((max_rel_dist - min_rel_dist) / del_rel_dist + 1)):
			self.rel_dist_list.append(min_rel_dist + i * del_rel_dist)

		### relative velocity abstraction
		min_rel_vel = -5 
		max_rel_vel = 5
		del_vel = 0.5

		self.rel_vel_list = []
		for i in range(int((max_rel_vel - min_rel_vel) / del_vel)+1):
			self.rel_vel_list.append(min_rel_vel + i * del_vel)

		self.ego_acc_values = np.array([-0.5, -0.25, 0.0, 0.25])
		self.disc_invalid_action = 0
		self.invalid_action = -10

		### For random seed purposes 
		self.train = train 	# Are we training or validating? For validating, we set the seed to get constant initializations

		"""
		### Initialization
		"""
		self.InitializeEnvironmentVariables()

	def InitializeEnvironmentVariables(self):
		self.episode_steps = 0
		self.done = False

		"""
		### Initial conditions
		"""
		self.rel_init_dist = self.InitializeRelDist()
		self.rel_init_vel = self.InitializeRelVel()
		self.fv_acc_list = self.AccelerationProfile(num_pts=4, N=100)

		self.mdp_state = np.array([self.rel_init_dist, self.rel_init_vel], dtype=np.float32)
		self.ustate = self.InitializeUstate()
		self.mdp_state = np.concatenate((self.mdp_state, self.ustate))
		self.state_buffer = deque(maxlen=self.max_td+1)
		self.state_buffer.append(np.array([self.rel_init_dist, self.rel_init_vel], dtype=np.float32))

		if self.constant_td:
			self.current_delay = self.max_td
		else:
			self.current_delay = 0


	def InitializeRelDist(self):
		if self.train:
			return 100
		else:
			return np.random.uniform(25,50)

	def InitializeRelVel(self):
		if self.train:
			return 0
		else:
			return np.random.uniform(0,5)

	def InitializeUstate(self):
		if self.constant_td:
			return np.zeros((self.max_td,))
		else:
			return np.ones((self.max_td,)) * self.invalid_action


	def GetShield(self, mdp_discrete_state):
		safe_actions_indices = np.where(self.shield[mdp_discrete_state])[0]
		safe_actions = self.ego_acc_values[safe_actions_indices]
		
		return safe_actions

	def AccelerationProfile(self, num_pts=4, N=100):
		acc_pts = np.random.uniform(-self.fv_max_acc, self.fv_max_acc, num_pts)
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
	
	def convert_random_td_mdp_state_to_int(self, state):
		# print(state)
		num_physical_actions = self.ego_acc_values.shape[0]
		num_buffer_actions = num_physical_actions+1
		increments = [1,] 
		increments += [(self.max_td+1)*num_buffer_actions**k for k in range(self.max_td)]
		increments += [(self.max_td+1)*num_buffer_actions**self.max_td,]
		increments.reverse()
		# print(increments)
		return int(np.sum(np.multiply(list(state), increments)))
	
	def convert_constant_td_mdp_state_to_int(self, state):
		# print(state)
		num_physical_actions = self.ego_acc_values.shape[0]
		increments = [num_physical_actions**k for k in range(self.max_td)]
		increments.reverse()
		increments = [num_physical_actions**self.max_td,] + increments
		# print(discrete_state, np.sum(np.multiply(discrete_state, increments)))
		return int(np.sum(np.multiply(state, increments)))

	def ConvertCurrentMDPStateToDiscrete(self, mdp_state):
		cont_rel_dist = self.mdp_state[0]
		cont_rel_vel = self.mdp_state[1]

		disc_rel_dist = 0
		if cont_rel_dist < self.rel_dist_list[0]:
			disc_rel_dist = 0
		elif cont_rel_dist > self.rel_dist_list[-1]:
			disc_rel_dist = len(self.rel_dist_list)-1
		else:
			for list_idx in range(len(self.rel_dist_list)-1):
				if self.rel_dist_list[list_idx] <= cont_rel_dist < self.rel_dist_list[list_idx+1]:
					disc_rel_dist = list_idx
					break


		disc_rel_vel = 0
		if cont_rel_vel < self.rel_vel_list[0]:
			disc_rel_vel = 0
		elif cont_rel_vel > self.rel_vel_list[-1]:
			disc_rel_vel = len(self.rel_vel_list)-1
		else:
			for list_idx in range(len(self.rel_vel_list)-1):
				if self.rel_vel_list[list_idx] <= cont_rel_vel < self.rel_vel_list[list_idx+1]:
					disc_rel_vel = list_idx
					break
		# print(disc_rel_dist, self.rel_dist_list[disc_rel_dist], disc_rel_vel, self.rel_vel_list[disc_rel_vel])
		disc_physical_state = (disc_rel_dist * len(self.rel_vel_list) + disc_rel_vel,)

		cont_ustate = mdp_state[2:]
		disc_ustate = cont_ustate.copy()
		for ui in range(self.max_td):
			cont_val = cont_ustate[ui]
			if cont_val == self.invalid_action:
				disc_val = self.disc_invalid_action
			else:
				if cont_val <= self.ego_acc_values[-1]:
					disc_val = np.where(self.ego_acc_values >= cont_val)[0][0]
				else:
					disc_val = len(self.ego_acc_values)-1
				disc_val += 1
			disc_ustate[ui] = disc_val
		disc_ustate = tuple(disc_ustate)

		if self.constant_td:
			discrete_state = disc_physical_state + disc_ustate
			return self.convert_constant_td_mdp_state_to_int(discrete_state)
		else:
			discrete_state = disc_physical_state + disc_ustate + (0,)
			return self.convert_random_td_mdp_state_to_int(discrete_state)

	def step(self, action):
		# print("-------------------------------------------")
		# print("previous robot state : ", self.state_buffer[-1])
		action = action[0] 
		# constraining the action space
		if action > self.ego_max_acc:
			action = self.ego_max_acc
		if action < self.ego_min_acc:
			action = self.ego_min_acc

		rel_pos = self.mdp_state[0]
		rel_vel = self.mdp_state[1]
		ustate = self.mdp_state[2:]
		num_valid_actions = len([u for u in ustate if u != self.invalid_action])
		# num_invalid_actions = self.max_td - num_valid_actions
		mdp_discrete_state = self.ConvertCurrentMDPStateToDiscrete(self.mdp_state) # contains the delayed state and actions executed hence

		# print("previous time delay : ", num_valid_actions)
		# print("previous mdp continuous state : ", self.mdp_state)
		# print("previous mdp discrete state : ", mdp_discrete_state)

		if self.train:
			ego_acc = action 
		else:
			allowed_actions = self.GetShield(mdp_discrete_state) # allowed action indices
			max_allowed_action = max(allowed_actions)
			if action >= max_allowed_action:
				action = max_allowed_action 
			
		ego_acc = action # action to be executed for the current time step
		# print("current action to be executed : ", ego_acc)

		"""
		### State transition
		"""
		fv_acc = self.fv_acc_list[self.episode_steps]
		fv_acc = fv_acc*self.fv_max_acc

		rel_acc = fv_acc - ego_acc	

		### State update
		rel_dist_trav = rel_vel*self.delt + 0.5*rel_acc*self.delt**2
		rel_pos = rel_pos + rel_dist_trav 
		rel_vel = rel_vel + rel_acc*self.delt

		self.current_delay = choices(self.possible_delays, self.td_dist[self.current_delay])[0]
		self.state_buffer.append(np.array([rel_pos, rel_vel]))

		# print("current robot state : ", self.state_buffer[-1])

		available_obs = self.state_buffer[len(self.state_buffer) - self.current_delay - 1]
		ustate = np.concatenate((ustate[:num_valid_actions], \
								+ np.array((action,))))
		ustate = ustate[ustate.shape[0]-self.current_delay:]
		invalid_actions = np.array((self.invalid_action,)*(self.max_td - ustate.shape[0]))
		ustate = np.concatenate((ustate, invalid_actions))
		self.mdp_state = np.concatenate((available_obs, ustate))

		# print("current time delay : ", self.current_delay)
		# print("current mdp continuous state : ", self.mdp_state)
		# print("current mdp discrete state : ", self.ConvertCurrentMDPStateToDiscrete(self.mdp_state))


		"""
		# Reward function
		"""
		### Reward for moving forward
		reward = -rel_dist_trav/40

		### Reward for being too close to the front vehicle
		if rel_pos < self.safety_dist:
			reward += self.violating_safety_dist_reward
		
		"""
		### Observation
		"""
		obs = self.mdp_state[:self.num_state_features]

		# print("current observation received from the cloud for which the cloud dnn provided the action : ", obs)

		"""
		### Environment handling 
		"""
		self.episode_steps += 1

		### Terminating the episode
		if rel_pos < self.safety_dist or self.episode_steps >= self.max_episode_steps:
			self.done = True 

		info = {'dis_rem': rel_pos}


		return obs, reward, self.done, info

	def reset(self, seed=0):
		if not self.train:
			np.random.seed(seed)
		self.InitializeEnvironmentVariables()
		obs = self.mdp_state[:self.num_state_features].copy()

		# print("-------------------------------------------")
		# print("current time stamp details")
		# print("current time stamp : ", self.episode_steps)
		# print("the current time delay : ", self.current_delay)
		# print("the current state buffer : ", self.state_buffer)
		# print("the current mdp state : ", self.mdp_state)

		return obs 