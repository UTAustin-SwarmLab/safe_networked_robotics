import os 
import sys
import numpy as np
import itertools

xmin = 0
xmax = 7 
ymin = 0  
ymax = 7

actions_dict = {0:'up', 1:'right', 2:'down', 3:'left'}
actions_list = list(actions_dict.keys())

td = int(sys.argv[1])
ustates = list(itertools.product(actions_list, repeat=td))

mdp = {}

def dynamics(state, act):
    rx = state[0]
    ry = state[1]
    ax = state[2]
    ay = state[3]
    ustate = state[4:-1]
    flag = state[-1]

    current_act = ustate[0]
    action = actions_dict[current_act]
    next_ustate = ustate[1:] + (act,)

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

        next_states = [(next_rx, next_ry, next_ax, next_ay) + next_ustate + (0,)]

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

            next_states.append((next_rx, next_ry, next_ax, next_ay) + ustate +(1,))

    return next_states

for rx in range(xmin, xmax+1):

    for ry in range(ymin, ymax+1):
        print(rx,ry)

        for ax in range(xmin, xmax+1):

            for ay in range(xmin, xmax+1):

                for ustate in ustates:

                    for flag in [0, 1]:

                        state = (rx, ry, ax, ay) + ustate + (flag,)

                        for act in actions_list:

                            state_action_pair = (state, act)
                            #print(state_action_pair)
                            next_states = dynamics(state, act)
                            mdp[state_action_pair] = next_states

os.makedirs("constant_generated", exist_ok=True)
np.save('constant_generated/mdp_%d_td' % td, mdp)
