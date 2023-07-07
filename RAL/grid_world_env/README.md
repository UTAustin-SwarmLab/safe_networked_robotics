## Usage

Execute the files in the order given below

## 1. Generate mdp without delay 
```
python no_td_mdp.py
```
## 2. Generate mdp with constant delay 
```
python constant_td_mdp.py time_delay
```
## 3. Generate mdp with random delay
```
python random_td_mdp.py
```
## 4. Generate the policy abstraction
```
python policy_abstraction.py
```
## 5. Generate maximum safety values (Vmax and Qmax) for no or constant delay
```
python model_checking_basic.py time_delay
```
## 6. Generate maximum safety values for random delay
```
python model_checking_basic_random.puy max_time_delay
```
## 7. Identify epsilon correspondint to the desired safety (Algorithm 1) for constant delay
```
python model_checking_updated.py time_delay epsilon desired_safety
```
## 8. Identify epsilon correspondint to the desired safety (Algorithm 1) for random delay
```
python model_checking_updated_random.py max_time_delay epsilon desired_safety 
```
## 9. Generate Qmax values for random delay
```
python generate_qvalues_for_random_td.py (assmues max delay of 3)
```
## 10. To generate the plot 3d.
```
python pmax_threshold.py
```
## 11. To generate the plot 3e. 
```
python pmax_threshold_random.py
```
## 12. To generate the plot 3f and to generate the quantitative results. 
Check grid_world_quantitative directory

