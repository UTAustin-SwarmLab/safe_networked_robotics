import os
import sys
import numpy as np 

def obtain_epsilon_shield(mdp, prob, mdp_states, delta, Vmax):
    """
    Definition 1.
    """
    actions_list = list(np.load('actions.npy', allow_pickle=True).item().keys())
    num_actions = len(actions_list)

    epsilon_shield = {}
    for state in mdp_states:
        per_state_filtered_actions = []
        state_value = Vmax[state]
        for a_idx in range(num_actions):
            state_action_pair = (state, a_idx)
            next_states = mdp[state_action_pair]
            next_states_prob = prob[state_action_pair]
            next_states_values = [Vmax[next_states[ii]] for ii in range(len(next_states))]
            qvalue = sum([nsv*nsp for nsv,nsp in zip(next_states_values, next_states_prob)])
            if qvalue >= delta or qvalue == state_value:
                per_state_filtered_actions.append(a_idx)

        epsilon_shield[state] = per_state_filtered_actions
    return epsilon_shield

def obtain_modified_policy(mdp, prob, mdp_states, policy, epsilon_shield):
    """
    Definition 2.
    """    
    actions_dict = np.load('actions.npy', allow_pickle=True).item()
    actions_list = list(actions_dict.keys())
    num_actions = len(actions_list)

    modified_policy = {}
    for state in mdp_states:
        basic_mdp_state = (state[0],)
        
        # obtaining the task efficient action
        task_efficiency_values = policy[basic_mdp_state]
        task_efficient_action = task_efficiency_values.index(max(task_efficiency_values))
        assert len(task_efficiency_values) == num_actions
        assert task_efficiency_values[task_efficient_action] == max(task_efficiency_values)

        # obtaining the most safe action
        epsilon_shielded_actions = epsilon_shield[state]
        max_safety_values = []
        for a_idx in range(num_actions):
            state_action_pair = (state, a_idx)
            next_states = mdp[state_action_pair]
            next_states_prob = prob[state_action_pair]
            next_states_values = [Vmax[next_states[ii]] for ii in range(len(next_states))] 
            qvalue = sum([nsv*nsp for nsv,nsp in zip(next_states_values, next_states_prob)])
            max_safety_values.append(qvalue)
        most_safe_action = max_safety_values.index(max(max_safety_values))
        assert len(max_safety_values) == num_actions
        assert most_safe_action in epsilon_shielded_actions        
        assert max_safety_values[most_safe_action] == max(max_safety_values)

        if task_efficient_action in epsilon_shielded_actions:
            modified_policy[state] = task_efficient_action
        else:
            modified_policy[state] = most_safe_action

    return modified_policy

def ValueIterationForActualSafety(mdp, prob, mdp_states, zero_td_bad_labels, zero_td_goal_labels, zero_td_init_labels, policy, td, eps=1e-5):   
    V = {state:0 for state in mdp_states}

     # Start value iteration
    guarantee = False
    while not guarantee:
        # print("iteration : %d" % i) 
        max_diff = 0
    
        for state in mdp_states:
            old_state_value = V[state]

            action = policy[state]   
            state_action_pair = (state, action)
            next_states = mdp[state_action_pair]
            next_state_values = [V[next_states[ii]] for ii in range(len(next_states))]
            next_state_probs = prob[state_action_pair]
            state_action_value = sum([nsv*nsp for nsv,nsp in zip(next_state_values, next_state_probs)])
            V[state] = state_action_value

            physical_state = (state[0],)
            if zero_td_bad_labels[physical_state] == 1: 
                V[state] = 0.0 
            
            if zero_td_goal_labels[physical_state] == 1: 
                V[state] = 1.0
            
            diff = abs(old_state_value - V[state])
            max_diff = max(max_diff, diff)

        if max_diff < eps:
            guarantee = True 

        print("max error : ", max_diff)

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
    # print(mdp_states)

    zero_td_bad_labels = np.load('constant_generated/unsafe_0_td.npy', allow_pickle=True).item()
    zero_td_goal_labels = np.load('constant_generated/goal_0_td.npy', allow_pickle=True).item()
    zero_td_initial_labels = np.load('constant_generated/initial_0_td.npy', allow_pickle=True).item()

    # # obtaining the maximum safety probability alias minimum reachability
    vmax_loc = 'random_generated/Vmax_values_%d_td.npy' % max_td 
    Vmax = np.load(vmax_loc, allow_pickle=True).item()

    # obtaining the epsilon shield as in Defn 1. of the paper
    delta = float(sys.argv[2])
    epsilon_shield = obtain_epsilon_shield(mdp, prob, mdp_states, delta, Vmax)

    # obtaining the modified policy as in Defn. 2 of the paper
    abstracted_policy_loc = 'abstracted_dnn_policy.npy' 
    policy = np.load(abstracted_policy_loc, allow_pickle=True).item()
    modified_policy = obtain_modified_policy(mdp, prob, mdp_states, policy, epsilon_shield) # Qmax not needed as we focus on executing the most optimal action from the epsilon shield
    # print(modified_policy)

    # obtaining the actual safety probability for the modified policy
    Vactual, Vactual_init = ValueIterationForActualSafety(mdp, prob, mdp_states, zero_td_bad_labels, zero_td_goal_labels, zero_td_initial_labels, modified_policy, max_td)
    print(Vactual_init)
    init_states_safety_values = list(Vactual_init.values())
    actual_safety = sum(init_states_safety_values)/len(init_states_safety_values) 

    print("the actual safety achievable is ", actual_safety)