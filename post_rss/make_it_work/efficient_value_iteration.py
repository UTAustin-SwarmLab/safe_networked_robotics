import numpy as np
import torch
from torch import nn, optim

mdp_loc = 'cruise_control/constant_generated/mdp_0_td.npy'
mdp = np.load(mdp_loc, allow_pickle=True).item()

print(mdp)


num_states = 100  # Number of states in the MDP
num_actions = 10  # Number of actions in the MDP
gamma = 0.9  # Discount factor
reward = torch.randn(num_states, num_actions)  # Reward function
transition = torch.randn(num_states, num_actions, num_states)  # Transition function

mdp = torch.zeros((10000000, 5, 10000000))
print(mdp)