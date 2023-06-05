import sys
sys.path.remove('/usr/lib/python3/dist-packages')
import itertools
import numpy as np 
import matplotlib.pyplot as plt
 

xmax = 7
ymax = 7

ax_init = 4
ay_init = 4

td = int(sys.argv[1])
per_td_vis_arrays = {}

state_values = np.load('constant_generated/state_values_%d_td.npy' % td, allow_pickle=True).item()
states = list(state_values.keys())

sc_vis_array = np.zeros((xmax+1, ymax+1))


for x in range(xmax+1):
    for y in range(ymax+1):
        physical_state = (x, y) + (ax_init, ay_init)
        stay_ustate = tuple([0 for t in range(td)])
        state = physical_state + stay_ustate + (1,)
        sc_pmax_val = state_values[state]
        print(state, sc_pmax_val)
        #if sc_pmax_val == 1:
        #    sc_vis_array[ymax-y, x] = 1
        #else:
        #    sc_vis_array[ymax-y, x] = 0
        sc_vis_array[ymax-1-y, x] = sc_pmax_val


# stay_control_vector
plt.imshow(sc_vis_array, cmap=plt.cm.Blues, extent=[0,9,0,9])
plt.title('Pmax values for the stay control vector', size=12)
plt.legend()
plt.savefig('generated/stay_control_vector_%d_td_%d_%d.png' % (td, ax_init, ay_init))
plt.clf()
plt.cla()
plt.close()

print(sc_vis_array)
