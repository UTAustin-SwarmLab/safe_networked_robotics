import numpy as np
import math
import os
import sys
import csv

class GridworldConstDelay():
    def __init__(self) -> None:
        super().__init__()
    
        # Number of actions for the robot
        self.n_action = 5
        
        # Observation space
        self.time_delay = None # 2 time steps

        self.n = 8         # RAL experiments assume the size of the grid to be 8
        
        # Task parameters
        self.max_episode_steps = 4*self.n # Epsisode length
        self.goal = (self.n-1,self.n-1) # Location of goal
    
        self.shield_active = False
        self.probList = None
        self.prob_threshold = None
        self.filename = None
        self.csvfile = None
        self.csvwriter = None

    def convert_state_to_int(self, state):
        num_xbins=8
        increments = [(num_xbins**3)*2, (num_xbins**2)*2, num_xbins*2, 2, 1]
        return np.sum(np.multiply(list(state), increments))

    def set_pmax_csv_file_name(self, filename):
        self.filename = filename
        self.csvfile = open(self.filename, 'w', newline='')
        self.csvwriter = csv.writer(self.csvfile, delimiter='\t', lineterminator='\n')

    def set_network_delay(self, delay):
        self.time_delay = delay

    def activate_shield(self, val, path, threshold):
        self.shield_active = val
        self.probList = np.load(path, allow_pickle = True).item()
        self.prob_threshold = threshold

    def trans_right(self, state):
        next_state_x = min(state[0] + 1, self.n - 1)
        next_state_y = state[1]
        
        return (next_state_x, next_state_y)
    
    def trans_left(self, state):
        next_state_x = max(state[0] - 1, 0)
        next_state_y = state[1]
        
        return (next_state_x, next_state_y)

    def trans_up(self, state):
        next_state_x = state[0]
        next_state_y = min(state[1] + 1, self.n - 1)
        
        return (next_state_x, next_state_y)

    def trans_down(self, state):
        next_state_x = state[0]
        next_state_y = max(state[1] - 1, 0)
        
        return (next_state_x, next_state_y)

    def trans_stay(self, state):
        return state

    def trans(self, state, action):
        if action < 0 or action >= self.n_action:
            raise ValueError("Action value out of bounds")
        if state[0] < 0 or state[0] >= self.n or state[1] < 0 or state[1] >= self.n:
            raise ValueError("State value out of bounds")
        
        if action == 1:
            return self.trans_up(state)
        if action == 2:
            return self.trans_right(state)
        if action == 3:
            return self.trans_down(state)
        if action == 4:
            return self.trans_left(state)
        if action == 0:
            return self.trans_stay(state) # Only for robot and not for adversary   

    def UpdateStateList(self, state):
        self.state_list = np.roll(self.state_list, -1, axis=0)
        self.state_list[-1] = np.array(state)
    
    def UpdateActionList(self, action):
        if self.action_list is not None:
            self.action_list = np.roll(self.action_list, -1, axis=0)
            self.action_list[-1] = np.array(action)

    def step(self, action):

        self.episode_steps += 1
        prev_state_rob = self.rob

        # The action is taken based on the current state and hence robot position should change first in step
        if self.shield_active:
            action_shield = 0
            prob_max = -1.0
            prob_list = []  # for debugging purposes
            aug_state = tuple(self.state_list[0].tolist()) + (1,)
            # print(aug_state)
            aug_state = self.convert_state_to_int(aug_state)
            if self.time_delay > 0:
                aug_state = tuple([aug_state] + self.action_list.tolist())
            else:
                aug_state = (aug_state,)

            for i in range(5):
                obs_loc = self.state_list[0].tolist()
                
                # prob = self.probList[self.convert_state_to_int(aug_state)][i]
                prob = self.probList[(aug_state, i)]
                if action == i:
                    prob_action = prob

                prob_list.append(prob)
                if prob > prob_max:
                    prob_max = prob
                    action_shield = i

            if prob_action <= self.prob_threshold:
                action = action_shield
            
            # print(prob_max)
            self.csvwriter.writerow([prob_max])
            # print('===========================')
            # print(f'Obs: {obs_loc}; PPrev Action: {self.action_pprev}; Prev Action: {self.action_prev}')
            # print(f'Action to be taken: {action}; Action shield: {action_shield}')
            # print(f'Probabilities- All: {prob_list[0]:0.2f}, {prob_list[1]:0.2f}, {prob_list[2]:0.2f}, {prob_list[3]:0.2f}, {prob_list[4]:0.2f}, Action: {prob_action:0.2f}')
            # print(f'Action to be taken: {action}; Action shield: {action_shield}')

        self.UpdateActionList(action)

        self.rob = self.trans(self.rob, action)
        info   = {'robot': self.rob,
                  'adversary' : self.adv}
        state  = (*self.rob, *self.adv)

        """If robot wins the game
        The adversary does not play
        """
        if self.rob == self.goal:
            info['gamestatus'] = 'won'
            reward = 2
            done   = True
            if self.time_delay > 0:
                obs = tuple(self.state_list[1].tolist()) #One step forward
                # Doing this because you don't want to add state to state_list yet
            else:
                obs = state
            return obs, reward, done, info

        action_ghost = np.random.randint(1, self.n_action) # ghost cannot stay in place
        self.adv = self.trans(self.adv, action_ghost)
        
        info   = {'robot': self.rob,
                  'adversary' : self.adv}
        state  = (*self.rob, *self.adv)
        self.UpdateStateList(state)
        obs = tuple(self.state_list[0].tolist())
        
        if self.adv == self.rob:
            info['gamestatus'] = 'lost'
            reward = -10
            done   = True
            return obs, reward, done, info
        
        if math.dist(self.rob, self.goal) < math.dist(prev_state_rob, self.goal):
            reward = 0
        else: 
            reward = -1/self.max_episode_steps

        done = False
        if self.episode_steps >= self.max_episode_steps:
            done = True
            info['gamestatus'] = 'tie'

        info['gamestatus'] = 'none'
        
        return obs, reward, done, info

    def reset(self):

        # Environment parameters
        self.episode_steps = 0

        # Environment state
        self.rob = (0,0)    # Robot init state
        self.adv  = (np.random.randint(int(self.n/3), int(3*self.n/3)),     # Adversary init state
                     np.random.randint(int(self.n/3), int(3*self.n/3)))
        state = (*self.rob, *self.adv)
        self.state_list = np.repeat(np.array(state, dtype=int).reshape(1,-1), self.time_delay + 1, axis=0)

        if self.time_delay > 0:
            self.action_list = np.zeros(self.time_delay, np.int8)
        else:
            self.action_list = None
        
        obs = tuple(self.state_list[0].tolist())

        return obs