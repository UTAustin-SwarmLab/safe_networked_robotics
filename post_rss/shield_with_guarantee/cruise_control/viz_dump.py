
# print(state_action_values[-22])
# print(state_values[-22])
# print(safest_actions[-22])
# print(safest_actions[-22].bool())
# print(delta_safe_actions[-22])
# print(allowed_actions[-22])


# state_values = state_values.detach().cpu().numpy()
# import matplotlib.pyplot as plt

# num_rel_dist_states = 27
# num_rel_vel_states = 22

# vis_array = np.zeros((num_rel_dist_states, num_rel_vel_states))

# for rel_dist_val in range(num_rel_dist_states):
#     for rel_vel_val in range(num_rel_vel_states):
#         physical_state = rel_dist_val * num_rel_vel_states + rel_vel_val
#         #print(physical_state) 
#         vis_array[rel_dist_val, rel_vel_val] = state_values[physical_state]


# # stay_control_vector
# plt.imshow(vis_array, cmap=plt.cm.Blues, extent=[-10,10,25,0])
# plt.title('Pmax values for the stay control vector', size=12)
# plt.ylabel('Relative Distance (m)', size=12)
# plt.xlabel('Relative velocity (m/s)', size=12)
# plt.legend()
# plt.savefig('constant_generated/stay_control_vector_%d_td.png' % 0)
# plt.clf()
# plt.cla()
# plt.close()

# print(vis_array)
