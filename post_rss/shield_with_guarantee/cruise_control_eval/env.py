import gym 
import math
import numpy as np
from os import path
from scipy.interpolate import UnivariateSpline

class ConstTdContinuousCruiseCtrlEnv(gym.Env):

	def __init__(self, time_delay=0, train=True, log_file_str=None, delta=1.0): 
 
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

		self.num_mdp_state_features = self.num_state_features + time_delay
		# print(self.num_mdp_state_features, time_delay)

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
		self.fv_max_acc = 0.5  # 1m/s^2
		self.delt = 1 # 1s time step 

		"""
		shielding stuff
		"""
		self.td = time_delay
		shield_path = 'constant_generated/%d_td/shield_%.1f_prob.npy' % (self.td, delta)  
		self.shield = np.load(shield_path, allow_pickle=True)

		### relative distance abstraction
		min_rel_dist = 0
		max_rel_dist = 25 
		del_rel_dist = 1.0  

		rel_dist_tuples = []
		for i in range(int((max_rel_dist - min_rel_dist) / del_rel_dist)):
			rel_dist_tuples.append((min_rel_dist + i * del_rel_dist, min_rel_dist + (i + 1) * del_rel_dist))

		neg_large_val = -1000
		pos_large_val = 1000

		self.rel_dist_tuples = [(neg_large_val, min_rel_dist)] + rel_dist_tuples + [(max_rel_dist, pos_large_val)]
		#print(rel_dist_tuples)

		### relative velocity abstraction
		min_rel_vel = -10
		max_rel_vel = 10
		del_vel = 1.0


		rel_vel_tuples = []
		for i in range(int((max_rel_vel - min_rel_vel) / del_vel)):
			rel_vel_tuples.append((min_rel_vel + i * del_vel, min_rel_vel + (i + 1) * del_vel))

		neg_large_val = -100
		pos_large_val = 100

		self.rel_vel_tuples = [(neg_large_val, min_rel_vel)] + rel_vel_tuples + [(max_rel_vel, pos_large_val)]
		#print(rel_vel_tuples)

		self.ego_acc_values = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
		self.num_discrete_actions = self.ego_acc_values.shape[0]

		self.delta = delta	# minimum probability that a state-action pair must have to satisfy the specification

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
		ustate = self.InitializeUstate()
		self.mdp_state = np.concatenate((self.mdp_state, ustate))

	def InitializeRelDist(self):
		if self.train:
			return 100
		else:
			return 100

	def InitializeRelVel(self):
		if self.train:
			return 0
		else:
			return 0

	def InitializeUstate(self):
		return np.zeros((self.td,))
 
	def GetShield(self, mdp_discrete_state):
		safe_actions_indices = np.where(self.shield[mdp_discrete_state])[0]
		safe_actions = self.ego_acc_values[safe_actions_indices]
		
		return safe_actions

	def AccelerationProfile(self, num_pts=4, N=100):
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

	def ConvertCurrentMDPStateToDiscrete(self):
		cont_rel_dist = self.mdp_state[0]
		cont_rel_vel = self.mdp_state[1]

		disc_rel_dist = 0
		for tup_idx in range(len(self.rel_dist_tuples)):
			tup = self.rel_dist_tuples[tup_idx]
			if tup[0] <= cont_rel_dist <= tup[1]:
				disc_rel_dist = tup_idx
				break

		disc_rel_vel = 0
		for tup_idx in range(len(self.rel_vel_tuples)):
			tup = self.rel_vel_tuples[tup_idx]
			if tup[0] <= cont_rel_vel <= tup[1]:
				disc_rel_vel = tup_idx
				break

		disc_physical_state = (disc_rel_dist * len(self.rel_vel_tuples) + disc_rel_vel,)
		cont_ustate = self.mdp_state[2:]
		disc_ustate = cont_ustate.copy()
		for ui in range(self.td):
			cont_val = cont_ustate[ui]
			disc_val = np.where(self.ego_acc_values >= cont_val)[0][0]
			disc_ustate[ui] = disc_val
		disc_ustate = tuple(disc_ustate)

		discrete_state = disc_physical_state + disc_ustate
		return discrete_state

	def step(self, action):
		action = action[0]
		rel_pos = self.mdp_state[0]
		rel_vel = self.mdp_state[1]
		ustate = self.mdp_state[2:]

		if self.train:
			ego_acc = action 
		else:
			mdp_discrete_state = self.ConvertCurrentMDPStateToDiscrete()
			allowed_actions = self.GetShield(mdp_discrete_state) # allowed action indices
			max_allowed_action = max(allowed_actions)
			if action >= max_allowed_action:
				action = max_allowed_action 

			ego_acc = action 

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
		self.mdp_state = np.array([rel_pos, rel_vel], dtype=np.float32)
		self.mdp_state = np.concatenate((self.mdp_state, ustate[1:]))


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

		return obs 