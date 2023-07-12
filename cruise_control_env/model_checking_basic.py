import sys
import numpy as np 

def ValueIteration(mdp, mdp_states, zero_td_bad_labels, zero_td_init_labels, td, eps=1e-16):
    ego_acc_list = list(np.load('actions.npy', allow_pickle=True))
    num_actions = len(ego_acc_list)

    mdp_state_action_pairs = list(mdp.keys())
    V = {state:0 for state in mdp_states}
    Q = {sa_pair:0 for sa_pair in mdp_state_action_pairs} 

    # Start value iteration
    guarantee = False
    while not guarantee: 
        # print("iteration : %d" % i)  
        max_diff = 0
    
        for state in mdp_states:
            old_state_value = V[state]

            state_action_values_list = []     
            for a in range(num_actions): 
                state_action_pair = (state, a)

                next_states = mdp[state_action_pair][0]
                next_states_probs = mdp[state_action_pair][1]
                next_state_values = [V[next_states[ii]] for ii in range(len(next_states))]
                assert next_state_values == [V[next_st] for next_st in next_states]
                
                state_action_value = sum([nsv*nsp for nsv,nsp in zip(next_state_values, next_states_probs)])

                state_action_values_list.append(state_action_value)
                Q[state_action_pair] = state_action_value

            state_value = min(state_action_values_list) 
            V[state] = state_value

            physical_state = (state[0],)
            if zero_td_bad_labels[physical_state] == 1:
                # print(physical_state)
                V[state] = 1.0
                for aii in range(num_actions): 
                    Q[(state,aii)] = 1.0
            
            diff = abs(old_state_value - V[state])
            max_diff = max(max_diff, diff)

        if max_diff < eps:
            guarantee = True 

        print("max error : ", max_diff)

    V = {state:1-V[state] for state in mdp_states}
    Q = {sa_pair:1-Q[sa_pair] for sa_pair in mdp_state_action_pairs}

    V_init = {}
    stay_action = np.where(np.array(ego_acc_list) == 0.0)[0][0]
    init_ustate = (stay_action,)*td
    for state in mdp_states:
        physical_state = (state[0],)
        ustate = state[1:]            
        if zero_td_init_labels[physical_state] == 1 and ustate == init_ustate:
            V_init[state] = V[state] 

    return V, Q, V_init

if __name__ == "__main__":
    td = int(sys.argv[1]) 
  
    """
    loading the mdp, unsafe and initial state labels
    """

    mdp = np.load('constant_generated/mdp_%d_td.npy' % td, allow_pickle=True).item()
    mdp_states = list(np.load('constant_generated/states_%d_td.npy' % td, allow_pickle=True).item().values())

    zero_td_bad_labels = np.load('constant_generated/unsafe_0_td.npy', allow_pickle=True).item()
    zero_td_initial_labels = np.load('constant_generated/initial_0_td.npy', allow_pickle=True).item()
    # print(zero_td_bad_labels)

    # # obtaining the maximum safety probability alias minimum reachability
    Vmax, Qmax, Vmax_init = ValueIteration(mdp, mdp_states, zero_td_bad_labels, zero_td_initial_labels, td)
    print(Vmax_init)
    init_states_safety_values = list(Vmax_init.values())
    max_safety = sum(init_states_safety_values)/len(init_states_safety_values) 

    print("the maximum safety achievable is ", max_safety)
    # print(Vmax)

    save_loc = 'constant_generated/Vmax_values_%d_td' % td 
    np.save(save_loc, Vmax) 
    save_loc = 'constant_generated/Qmax_values_%d_td' % td 
    np.save(save_loc, Qmax) 