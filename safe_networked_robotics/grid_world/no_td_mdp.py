import os 
import sys
sys.path.remove('/usr/lib/python3/dist-packages')
import numpy as np

xmin = 0
xmax = 7
ymin = 0 
ymax = 7

actions_dict = {0:'stay', 1:'up', 2:'right', 3:'down', 4:'left'}
actions_list = list(actions_dict.keys())

mdp = {}

def convert_state_to_int(state):
	int_val = 0
	state_len = len(state)
	for i, j in enumerate(reversed(range(state_len))):
		int_val += state[j] * 10 ** i 
	return int_val

def dynamics(state, act):
    rx = state[0]
    ry = state[1]
    ax = state[2]
    ay = state[3]
    flag = state[-1]

    action = actions_dict[act]

    if flag:
        if action == 'stay':
            next_rx = rx 
            next_ry = ry 

        if action == 'up':
            next_rx = rx
            next_ry = min(ry + 1, ymax)

        if action == 'down':
            next_rx = rx
            next_ry = max(ry - 1, ymin)

        if action == 'left':
            next_rx = max(rx - 1, xmin)
            next_ry = ry 

        if action == 'right':
            next_rx = min(rx + 1, xmax)
            next_ry = ry 

        next_ax = ax
        next_ay = ay

        next_states = [(next_rx, next_ry, next_ax, next_ay, 0)]

    else:
        next_rx = rx 
        next_ry = ry

        next_states = []
        for act in actions_list:
            action = actions_dict[act]
            if action == 'stay':
                continue

            if action == 'up':
                next_ax = ax
                next_ay = min(ay + 1, ymax)

            if action == 'down':
                next_ax = ax
                next_ay = max(ay - 1, ymin)

            if action == 'left':
                next_ax = max(ax - 1, xmin)
                next_ay = ay  

            if action == 'right':
                next_ax = min(ax + 1, xmax)
                next_ay = ay

            next_states.append((next_rx, next_ry, next_ax, next_ay, 1))

    return next_states

for rx in range(xmin, xmax+1):

    for ry in range(ymin, ymax+1):

        for ax in range(xmin, xmax+1):

            for ay in range(xmin, xmax+1):

                for flag in [0, 1]:

                    state = (rx, ry, ax, ay, flag)

                    for act in actions_list:

                        state_action_pair = (state, act)
                        next_states = dynamics(state, act)
                        #next_states = [(convert_state_to_int(next_state),) for next_state in next_states]
                        mdp[state_action_pair] = next_states

print(mdp)
os.makedirs("constant_generated", exist_ok=True)
np.save('constant_generated/mdp_0_td', mdp)
