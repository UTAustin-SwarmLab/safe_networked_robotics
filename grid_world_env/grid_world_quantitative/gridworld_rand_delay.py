import numpy as np
import math
import sys
import csv

# Environment for random time-delay

class GridworldRandDelay(): 
    def __init__(self) -> None:
        super().__init__()
    
        # Number of actions for the robot
        self.n_action = 5
        
        # Observation space
        self.max_time_delay = None

        self.n = 8         # Size of grid
        
        # Task parameters
        self.max_episode_steps = 4*self.n # Epsisode length
        self.goal = (self.n-1,self.n-1) # Location of goal
    
        self.shield_active = False
        self.probDict = None
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

    def set_max_time_delay(self, delay):
        self.max_time_delay = delay
        assert self.max_time_delay == 3, f"environment implemented only for time delay of 3" 
    
    def get_next_time_delay(self, time_delay):
        if time_delay == 0:
            return np.random.choice([0, 1], p=[0.9, 0.1])
        if time_delay == 1:
            return np.random.choice([0, 1, 2], p=[0.8, 0.1, 0.1])
        if time_delay == 2:
            return np.random.choice([0, 1, 2, 3], p=[0.7, 0.1, 0.1, 0.1])
        if time_delay == 3:
            return np.random.choice([0, 1, 2, 3], p=[0.7, 0.1, 0.1, 0.1])
        
    def activate_shield(self, val, path, threshold):
        self.shield_active = val
        self.probDict = np.load(path, allow_pickle = True).item()
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
        self.action_list = np.roll(self.action_list, -1, axis=0)
        self.action_list[-1] = np.array(action)

    def step(self, action):

        self.episode_steps += 1
        prev_state_rob = self.rob
        
        """
        MEASURE CURRENT TIME DELAY
        """
        self.time_delay = self.get_next_time_delay(self.time_delay)

        """
        SHIELD THE CONTROLLER ACTION
        """
        # The action is taken based on the current state and hence robot position should change first in step
        if self.shield_active:
            action_shield = 0
            prob_max = -1.0
            prob_list = []  # for debugging purposes
            idx = self.max_time_delay - self.time_delay
            s_td = tuple(self.state_list[idx].tolist()) + (1,)
            enc_state = self.convert_state_to_int(s_td)
            # print(s_td)
            # print(enc_state)
            if self.time_delay == 0:
                aug_state = (enc_state, -1, -1, -1, 0)
            elif self.time_delay == 1:
                aug_state = (enc_state, self.action_list[2], -1, -1, 0)
            elif self.time_delay == 2:
                aug_state = (enc_state, self.action_list[1], self.action_list[2], -1, 0)
            elif self.time_delay == 3:
                aug_state = (enc_state, self.action_list[0], self.action_list[1], self.action_list[2], 0)
            else:
                sys.exit("Unexpected time delay value")


            for i in range(5):
                obs_loc = s_td
                
                prob = self.probDict[(aug_state, i)]
                if action == i:
                    prob_action = prob

                prob_list.append(prob)
                if prob > prob_max:
                    prob_max = prob
                    action_shield = i

            if prob_action <= self.prob_threshold:
                action = action_shield
            
            self.csvwriter.writerow([prob_max])
            # print('===========================')
            # print(f'Encoded augmented state: {aug_state}; Action list: {self.action_list}')
            # print(f'Probabilities- All: {prob_list[0]:0.2f}, {prob_list[1]:0.2f}, {prob_list[2]:0.2f}, {prob_list[3]:0.2f}, {prob_list[4]:0.2f}, Action: {prob_action:0.2f}')
            # print(f'Action to be taken: {action}; Action shield: {action_shield}') 

        self.UpdateActionList(action)

        """
        ROBOT PLAYS
            If robot wins the game, adversary does not play
        """
        self.rob = self.trans(self.rob, action)
        info   = {'robot': self.rob,
                  'adversary' : self.adv}
        state  = (*self.rob, *self.adv)
        
        if self.rob == self.goal:
            info['gamestatus'] = 'won'
            reward = 2
            done   = True
            if self.time_delay > 0:
                idx = self.max_time_delay - self.time_delay
                obs = tuple(self.state_list[idx + 1].tolist())
                # Doing this because you don't want to add state to state_list yet
            else:
                obs = state
            return obs, reward, done, info

        """
        ADVERSARY PLAYS
        """
        action_ghost = np.random.randint(1, self.n_action) # ghost cannot stay in place
        self.adv = self.trans(self.adv, action_ghost)
        
        info   = {'robot': self.rob,
                  'adversary' : self.adv}
        state  = (*self.rob, *self.adv)
        self.UpdateStateList(state)
        idx = self.max_time_delay - self.time_delay
        obs = tuple(self.state_list[idx].tolist())
        
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
        self.time_delay = np.random.choice([0, 1, 2, 3], p=[0.25, 0.25, 0.25, 0.25])

        self.rob = (0,0)    # Robot init state
        self.adv  = (np.random.randint(int(self.n/3), int(3*self.n/3)),     # Adversary init state
                     np.random.randint(int(self.n/3), int(3*self.n/3)))
        state = (*self.rob, *self.adv)
        self.state_list = np.repeat(np.array(state, dtype=int).reshape(1,-1), self.max_time_delay + 1, axis=0)

        self.action_list = np.zeros(self.max_time_delay, np.int8)

        idx = self.max_time_delay - self.time_delay
        obs = tuple(self.state_list[idx].tolist())

        return obs