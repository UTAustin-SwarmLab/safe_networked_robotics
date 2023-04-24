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

		self.max_rel_vel = 5 
		self.min_rel_vel = -5

		self.td = time_delay
		state_action_values_path = '/home/sai-admin/rss_submission_code/cruise_control/constant_generated/state_action_values_%d_td.npy' % self.td  
		self.state_action_values = np.load(state_action_values_path, allow_pickle=True).item()

		self.min_rel_dist_disc = 0 							# minimum value of the relative distance discreet state
		self.max_rel_dist_disc = 25							# maximum value of the relative distance discreet state
		self.num_rel_dist_states = 26

		self.min_rel_vel_disc = 0							# minimum value of the relative velocity discreet state
		self.max_rel_vel_disc = 9							# maximum value of the relative distance discreet state
		self.num_rel_vel_states = 10

		self.delta = delta			# minimum probability that a state-action pair must have to satisfy the specification

		### For random seed purposes 
		self.train = train 									# Are we training or validating? For validating, we set the seed to get constant initializations

		self.fv_max_acc = 0.5  # 1m/s^2
		self.discrete_actions = np.array([-1, -0.5, 0, 0.5, 1.0])
		self.num_discrete_actions = len(self.discrete_actions)
		
		self.delt = 1 # 1s time step 

		### for seed purposes 
		self.train = train # are we training or validating? For validating, we set the seed to get constant initializations

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
		safe_actions = []
		for action in range(1,self.num_discrete_actions+1):
			mdp_state_action_pair = (mdp_discrete_state, action)
			mdp_state_action_value = self.state_action_values[mdp_state_action_pair]
			if mdp_state_action_value >= self.delta:
				safe_actions.append(action-1)

		return safe_actions

	def GetMaxSafeAction(self, mdp_discrete_state):
		mdp_state_action_values = []
		for action in range(1,self.num_discrete_actions+1):
			mdp_state_action_pair = (mdp_discrete_state, action)
			mdp_state_action_values.append(self.state_action_values[mdp_state_action_pair])

		return mdp_state_action_values.index(max(mdp_state_action_values))

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

	def ConvertCurrentMDPStateToDiscrete(self):
		# converts the continuous mdp state to discrete mdp state
		#print("here")
		#print(self.mdp_state[1])
		#print(math.floor((self.mdp_state[1]+10)/2))
		#print(max(self.min_rel_vel_disc, min(self.max_rel_vel_disc, math.floor((self.mdp_state[1]+10)/2))))
		discrete_physical_state = (max(self.min_rel_dist_disc, min(self.max_rel_dist_disc, math.floor(self.mdp_state[0]))), \
									max(self.min_rel_vel_disc, min(self.max_rel_vel_disc, math.floor(self.mdp_state[1]+5))))
		abstract_state = (discrete_physical_state[0] * self.num_rel_vel_states + discrete_physical_state[1],)

		ustate = self.mdp_state[2:]
		discrete_ustate = [self.discrete_actions[self.discrete_actions>=ustate[ui]].min() for ui in range(self.td)]
		discrete_ustate = tuple([np.where(self.discrete_actions == discrete_ustate[ui])[0][0] for ui in range(self.td)])
		discrete_ustate = tuple([u+1 for u in discrete_ustate])

		discrete_state = abstract_state + discrete_ustate
		return discrete_state

	def step(self, action):
		#print(self.mdp_state, action)
		action = action[0]
		rel_pos = self.mdp_state[0]
		rel_vel = self.mdp_state[1]
		ustate = self.mdp_state[2:]

		mdp_discrete_state = self.ConvertCurrentMDPStateToDiscrete()
		#print(self.mdp_state, mdp_discrete_state, action)
		allowed_actions = self.GetShield(mdp_discrete_state)
		#print(allowed_actions)
		#print(self.discrete_actions[allowed_actions])

		if self.train:
			ego_acc = action 
		else:
			mdp_discrete_state = self.ConvertCurrentMDPStateToDiscrete()
			#print("here")
			allowed_actions = self.GetShield(mdp_discrete_state) # allowed action indices
			if rel_pos < 250:
				if len(allowed_actions) != 0:
					max_allowed_action = self.discrete_actions[max(allowed_actions)] # max discrete action
					if action >= max_allowed_action:
						action = max_allowed_action
				else:
					# if there are no allowed actions, take the action with highest pmax value
					action_idx = self.GetMaxSafeAction(mdp_discrete_state)
					action = self.discrete_actions[action_idx]
			ustate = np.append(ustate, action)
			ego_acc = ustate[0]



		"""
		### State transition
		"""
		fv_acc = self.fv_acc_list[self.episode_steps]
		fv_acc = fv_acc*self.fv_max_acc
		#if self.train:
		#fv_acc = 0.0

		rel_acc = fv_acc - ego_acc
		
		### Clipping acceleration to keep within velocity limits
		if rel_vel >= self.max_rel_vel:
			if rel_vel + rel_acc*self.delt >= self.max_rel_vel:
				rel_acc = 0.0
		else:
			if rel_vel + rel_acc*self.delt >= self.max_rel_vel:
				rel_acc = (self.max_rel_vel - rel_vel)/self.delt

		if rel_vel <= self.min_rel_vel:
			if rel_vel + rel_acc*self.delt <= self.min_rel_vel:
				rel_acc = 0.0
		else:
			if rel_vel + rel_acc*self.delt <= self.min_rel_vel:
				rel_acc = (self.min_rel_vel - rel_vel)/self.delt	

		### State update
		rel_dist_trav = rel_vel*self.delt + 0.5*rel_acc*self.delt**2
		rel_pos = rel_pos + rel_dist_trav 
		rel_vel = rel_vel + rel_acc*self.delt
		self.mdp_state = np.array([rel_pos, rel_vel], dtype=np.float32)
		self.mdp_state = np.concatenate((self.mdp_state, ustate[1:]))
		#print(self.mdp_state)
		#print("-----")

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