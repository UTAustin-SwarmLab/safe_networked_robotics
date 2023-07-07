# safe_networked_robotics
The is the repo of **S**afe **N**etworked **R**obotics wia **P**robabilistic **V**erification. 


## TLDR
This paper develops methods to ensure the safety of teleoperated robots with stochastic latency. To do so, we use tools from formal verification to construct a shield (i.e.,run-time monitor) that provides a list of safe actions for any delayed sensory observation, given the expected and worst-case network latency.

## Usage

### Repicate our results for car following
Check the readme file within the cruise_control_env directory 


### Repicate our results for grid world
Check the readme file within the grid_world_env directory 

## To generate mdp without delay 
```
python safe_networked_robotics/cruise_control/no_td_mdp.py
```
## To generate mdp with constant delay 
```
python safe_networked_robotics/cruise_control/constant_td_mdp.py time_delay
```
## To generate mdp with random delay
```
python safe_networked_robotics/cruise_control/cruise_control.py
```
## To generate the plot 
```
python safe_networked_robotics/cruise_control_eval/thres_test.py
```
