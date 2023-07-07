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
## 5. Generate maximum safety values for no or constant delay
```
python model_checking_basic.py time_delay
```
## 6. Generate maximum safety values for random delay
```
python model_checking_basic_random.puy max_time_delay
```
## 7. Generate shield for desired safety by adjusting epsilon (Algorithm 1) for constant delay
```
python model_checking_updated.py time_delay epsilon desired_safety
```
## 8. Generate shield for desired safety by adjusting epsilon (Algorithm 1) for random delay
```
python model_checking_updated_random.py max_time_delay epsilon desired_safety 
```
## 9. Get the quantitative results stored as a pkl file (to generate the box plot)
```
python thres_test_mod.py
```
## 10. To generate the plot 3a.
```
python pmax_threshold.py
```
## 11. To generate the plot 3b. 
```
python pmax_threshold_random.py
```
## 12. To generate the plot 3c. 
```
python quantitative_plot.py
```
