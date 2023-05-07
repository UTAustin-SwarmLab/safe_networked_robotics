import numpy as np 
import torch 
import copy

device = torch.device("cuda:0")

transition = np.load('constant_generated/transition_arr/0_td.npy', allow_pickle=True)
transition = torch.tensor(transition, dtype=torch.float32).to(device)
num_states = transition.shape[0]
num_actions = transition.shape[1]

bad_state_values = torch.zeros((num_states,)).to(device)
bad_state_values[:132] = 1.0

# Let's perform value iteration 
eps = 1e-16

state_values = torch.zeros((num_states,), dtype=torch.float32).to(device) # initialization
old_state_values = copy.deepcopy(state_values)
max_err = 1000
while max_err > eps:
    state_values = torch.min(torch.matmul(transition, state_values), dim=1).values
    state_values[:132] = 1.0
    max_err = torch.max(state_values - old_state_values)
    old_state_values = state_values

print(state_values)

state_values = state_values.detach().cpu().numpy()

import matplotlib.pyplot as plt

num_rel_dist_states = 27
num_rel_vel_states = 22

vis_array = np.zeros((num_rel_dist_states, num_rel_vel_states))

for rel_dist_val in range(num_rel_dist_states):
    for rel_vel_val in range(num_rel_vel_states):
        physical_state = rel_dist_val * num_rel_vel_states + rel_vel_val
        #print(physical_state) 
        vis_array[rel_dist_val, rel_vel_val] = state_values[physical_state]


# stay_control_vector
plt.imshow(vis_array, cmap=plt.cm.Blues, extent=[-10,10,25,0])
plt.title('Pmax values for the stay control vector', size=12)
plt.ylabel('Relative Distance (m)', size=12)
plt.xlabel('Relative velocity (m/s)', size=12)
plt.legend()
plt.savefig('constant_generated/stay_control_vector_%d_td.png' % 0)
plt.clf()
plt.cla()
plt.close()

print(vis_array)
