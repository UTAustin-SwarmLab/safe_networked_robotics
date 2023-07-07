import os
import sys
import numpy as np 

def obtain_epsilon_shield(mdp_states, delta, Vmax, Qmax):
    """
    Definition 1.
    """
    ego_acc_list = np.load('actions.npy', allow_pickle=True)
    num_actions = len(ego_acc_list)

    epsilon_shield = {}
    for state in mdp_states:
        per_state_filtered_actions = []
        state_value = Vmax[state]
        for a_idx in range(num_actions):
            state_action_pair = (state, a_idx)
            qvalue = Qmax[state_action_pair]
            if qvalue >= delta or qvalue == state_value:
                per_state_filtered_actions.append(a_idx)

        epsilon_shield[state] = per_state_filtered_actions
    return epsilon_shield

def obtain_modified_policy(mdp_states, policy, epsilon_shield):
    """
    Definition 2.
    """
    ego_acc_list = list(np.load('actions.npy', allow_pickle=True))
    
    modified_policy = {}
    for state in mdp_states:
        physical_state = (state[0],)
        task_efficient_action = policy[physical_state]
        epsilon_shielded_actions = epsilon_shield[state]

        if task_efficient_action <= ego_acc_list[-1]:
            task_efficient_discrete_action = np.where(ego_acc_list >= task_efficient_action)[0][0] # associating the policy's continuous action with the closest and largest discrete action
            if task_efficient_discrete_action in epsilon_shielded_actions:
                modified_policy[state] = task_efficient_discrete_action # unmodified
            else:
                modified_policy[state] = int(max(epsilon_shielded_actions)) # executing the most task efficient action from the epsilon shield
        else:
            modified_policy[state] = int(max(epsilon_shielded_actions)) # executing the most task efficient action from the epsilon shield

    return modified_policy

def ValueIterationForActualSafety(mdp, prob, mdp_states, zero_td_bad_labels, zero_td_init_labels, modified_policy, td, eps=1e-16):
    V = {state:0 for state in mdp_states}

    # Start value iteration
    guarantee = False 
    while not guarantee: 
        # print("iteration : %d" % i) 
        max_diff = 0
    
        for state in mdp_states: 
            old_state_value = V[state]
            action = modified_policy[state]
            state_action_pair = (state, action)
            next_states = mdp[state_action_pair]
            next_states_probs = prob[state_action_pair]
            next_state_values = [V[next_states[ii]] for ii in range(len(next_states))]
            V[state] = sum([nsv*nsp for nsv,nsp in zip(next_state_values, next_states_probs)])

            physical_state = (state[0],)
            if zero_td_bad_labels[physical_state] == 1:
                V[state] = 1.0
            
            diff = abs(old_state_value - V[state])
            max_diff = max(max_diff, diff)

        if max_diff < eps:
            guarantee = True 

        print("max error : ", max_diff)

    V = {state:1-V[state] for state in mdp_states}

    V_init = {}
    init_ustate = (-1,)*td
    for state in mdp_states:
        physical_state = (state[0],)
        ustate = state[1:-1]            
        itm = state[-1]
        if zero_td_init_labels[physical_state] == 1 and ustate == init_ustate and itm == 0:
            V_init[state] = V[state] 

    return V, V_init


if __name__ == "__main__":

    max_td = int(sys.argv[1]) 

    """
    loading the mdp, unsafe and initial state labels
    """

    mdp = np.load('random_generated/mdp_transitions_%d_td.npy' % max_td, allow_pickle=True).item()
    prob = np.load('random_generated/mdp_probabilities_%d_td.npy' % max_td, allow_pickle=True).item()
    mdp_states = list(np.load('random_generated/states_%d_td.npy' % max_td, allow_pickle=True).item().values())

    zero_td_bad_labels = np.load('constant_generated/unsafe_0_td.npy', allow_pickle=True).item()
    zero_td_initial_labels = np.load('constant_generated/initial_0_td.npy', allow_pickle=True).item()

    # # obtaining the maximum safety probability alias minimum reachability
    vmax_loc = 'random_generated/Vmax_values_%d_td.npy' % max_td 
    Vmax = np.load(vmax_loc, allow_pickle=True).item()
    qmax_loc = 'random_generated/Qmax_values_%d_td.npy' % max_td
    Qmax = np.load(qmax_loc, allow_pickle=True).item()

    # obtaining the epsilon shield as in Defn 1. of the paper
    delta = float(sys.argv[2])
    epsilon_shield = obtain_epsilon_shield(mdp_states, delta, Vmax, Qmax)
    # print(epsilon_shield)

    # obtaining the modified policy as in Defn. 2 of the paper
    abstracted_policy_loc = 'abstracted_dnn_policy.npy' 
    policy = np.load(abstracted_policy_loc, allow_pickle=True).item()
    modified_policy = obtain_modified_policy(mdp_states, policy, epsilon_shield) # Qmax not needed as we focus on executing the most optimal action from the epsilon shield
    # print(modified_policy)

    # obtaining the actual safety probability for the modified policy
    Vactual, Vactual_init = ValueIterationForActualSafety(mdp, prob, mdp_states, zero_td_bad_labels, zero_td_initial_labels, modified_policy, max_td)
    print(Vactual_init)
    init_states_safety_values = list(Vactual_init.values())
    actual_safety = sum(init_states_safety_values)/len(init_states_safety_values) 

    print("the actual safety achievable is ", actual_safety)


    save_dir = 'random_generated/%d_td/' % max_td
    os.makedirs(save_dir, exist_ok=True) 
    save_loc = os.path.join(save_dir, 'shield_%s_prob.npy' % sys.argv[3])
    # print(save_loc)
    # print(shield)
    np.save(save_loc, epsilon_shield)
