'''
CONSTANT TIME DELAY GRID WORLD EXPERIMENTS
'''
from gridworld_const_delay import GridworldConstDelay
import sys
import os
import pickle
import numpy as np
import collections

from algos.q_learning import q_learning, q_learning_test_pm

def main(argv):
    shield_loc = 'shields/Qmax_values_3_td_random.npy'
    controller_name = 'Q_td0_ns_8x8.pkl'
    csv_file_name = 'temp.csv'
    modelsPath = os.path.join(os.path.dirname(sys.argv[0]), 'models')

    # This main file is for the grid-world scenario where the controller has no idea a time-delay exists
    env = GridworldConstDelay()
    active_shield = True

    # SET DELAY OF 0,1,2 OR 3 AND THE DESIRED THRESHOLD EPSILON
    network_delay = int(argv[1])
    assert network_delay >=0 
    assert network_delay <=3

    shield_loc = 'shields/Qmax_values_'+str(network_delay)+'_td.npy'
    shield_loc = os.path.join(os.path.dirname(sys.argv[0]), shield_loc)
    
    threshold = float(argv[2])
    assert threshold <= 1.0
    assert threshold >= 0.0

    num_episodes_test = int(argv[3])
    assert num_episodes_test>0

    env.activate_shield(active_shield, shield_loc, threshold)
    env.set_network_delay(network_delay)
    env.set_pmax_csv_file_name(csv_file_name)

    filehandler = open(os.path.join(modelsPath, controller_name), 'rb') 
    Q = pickle.load(filehandler)
    Q = collections.defaultdict(lambda: np.zeros(env.n_action), Q)


    stats = q_learning_test_pm(env, Q, epsilon = 0.0, num_episodes=num_episodes_test)
    print(f'Wins: {int(stats.game_status[0])}, \
        Losses: {int(stats.game_status[1])}, \
            Ties: {int(stats.game_status[2])}')
    
if __name__ == "__main__":
   main(sys.argv)
