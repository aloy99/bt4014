import numpy as np
from tqdm import tqdm
import multiprocessing
import pandas as pd
from functools import partial
import pickle
from scipy.linalg import solve

# Create class object for a single linear ucb disjoint arm
class cbts_arm():
    
    def __init__(self, arm_index, d, R = 0.01, epsilon = 0.5, delta = 0.5):
        
        # Track arm index
        self.arm_index = arm_index
        
        self.R = R
        self.epsilon = epsilon
        self.delta = delta

        self.d = d
        
        # A: (d x d) matrix = D_a.T * D_a + I_d. 
        # The inverse of A is used in ridge regression 
        self.A = np.identity(d).astype(np.float32)
        
        # b: (d x 1) corresponding response vector. 
        # Equals to D_a.T * c_a in ridge regression formulation
        self.b = np.zeros([d,1]).astype(np.float32)

        self.v_const = R * np.sqrt(24 / epsilon * d)

        self.rng = np.random.default_rng()
        
    def calc_bts(self, x_array, T):
        # Find A inverse for ridge regression
        A_inv = np.linalg.inv(self.A)
        
        mu_hat = A_inv @ self.b

        v = self.v_const * np.sqrt(np.log(T / self.delta))
        mu_tilde = self.rng.multivariate_normal(
        mu_hat.flat, v**2 * A_inv)[..., np.newaxis]

        return x_array @ mu_tilde


    def reward_update(self, reward, x_array):
        # Reshape covariates input into (d x 1) shape vector
        x = x_array.reshape([-1,1])
        
        # Update A which is (d * d) matrix.
        self.A += np.dot(x, x.T)
        
        # Update b which is (d x 1) vector
        # reward is scalar
        self.b += reward * x 


class cbts_policy():
    
    def __init__(self, K_arms, d, R):
        self.K_arms = K_arms
        self.cbts_arms = [cbts_arm(arm_index = i, d = d, R = R) for i in range(K_arms)]

    def topk_by_partition(self, array, k, axis=None, ascending=True):
        if not ascending:
            array *= -1
        ind = np.argpartition(array, k, axis=axis)
        ind = np.take(ind, np.arange(k), axis=axis) # k non-sorted indices
        array = np.take_along_axis(array, ind, axis=axis) # k non-sorted values

        # sort within k elements
        ind_part = np.argsort(array, axis=axis)
        ind = np.take_along_axis(ind, ind_part, axis=axis)
        if not ascending:
            input *= -1
        val = np.take_along_axis(array, ind_part, axis=axis) 
        return ind, val
        
    def select_arm(self, x_array, T):
        # Initiate ucb to be 0
        highest_bts = -1
        
        # Track index of arms to be selected on if they have the max UCB.
        candidate_arms = []
        
        for arm_index in range(self.K_arms):
            # Calculate ucb based on each arm using current covariates at time t
            arm_bts = self.cbts_arms[arm_index].calc_bts(x_array, T)
            
            # If current arm is highest than current highest_ucb
            if arm_bts > highest_ucb:
                
                # Set new max ucb
                highest_ucb = highest_bts
                
                # Reset candidate_arms list with new entry based on current arm
                candidate_arms = [arm_index]

            # If there is a tie, append to candidate_arms
            if arm_bts == highest_bts:
                
                candidate_arms.append(arm_index)
        
        # Choose based on candidate_arms randomly (tie breaker)
        chosen_arm = np.random.choice(candidate_arms)
        
        return chosen_arm

    def select_k_arms(self, x_array, T, k):
        arm_bts = np.array([self.cbts_arms[arm_index].calc_bts(x_array, T) for arm_index in range(self.K_arms)])
        return self.topk_by_partition(arm_bts, k)[1]

def ctr_simulator(df, K_arms, d, R):
    # Initiate policy
    cbts_policy_object = cbts_policy(K_arms = K_arms, d = d, R = R)
    
    # Instantiate trackers
    aligned_time_steps = 0
    cumulative_rewards = 0
    aligned_ctr = []
    unaligned_ctr = [] # for unaligned time steps

    for idx, row in tqdm(df.iterrows()):

        # 1st column: Logged data arm. 
        # Integer data type
        data_arm = row['item_id']

        # 2nd column: Logged data reward for logged chosen arm
        # Float data type
        data_reward = row['click']

        # 3rd columns onwards: 100 covariates. Keep in array of dimensions (100,) with float data type
        #covariate_string_list = row[[f'user-item_affinity_{i}' for i in range(1,80)]]
        covariate_string_list = row.iloc[2:]
        data_x_array = np.array([float(covariate_elem) for covariate_elem in covariate_string_list])

        # Find policy's chosen arm based on input covariates at current time step
        arms = cbts_policy_object.select_k_arms(data_x_array, aligned_time_steps + 1, k = 3)

        # Check if arm_index is the same as data_arm (ie same actions were chosen)
        # Note that data_arms index range from 1 to 10 while policy arms index range from 0 to 9.

        if data_reward == 1:
            arm_index = int(data_arm)
            reward = int(data_arm in arms)
            # Use reward information for the chosen arm to update
            cbts_policy_object.cbts_arms[arm_index].reward_update(reward, data_x_array)

            # For CTR calculation
            aligned_time_steps += 1
            cumulative_rewards += reward
            aligned_ctr.append(cumulative_rewards/aligned_time_steps)
                    
    return (aligned_time_steps, cumulative_rewards, aligned_ctr, cbts_policy_object)


if __name__ == "__main__":
    R = 0.05
    n_simulations = 2
    bootstrap_sample_size = 1000

    df_main = pd.read_csv('full_random_men.csv')
    #optimal size for bootstrap sample: assuming 1/80 chance that each item is actually selected, and a minimum of 1000 steps, 80000 rows required each time

    samples = [df_main.sample(bootstrap_sample_size) for i in range(n_simulations)]
    partial_func = partial(ctr_simulator, K_arms = 34, d = 65, R = R)

    with multiprocessing.Pool(processes = 4) as pool:
        
        results = pool.map(partial_func, samples)

    with open('results.pkl', 'wb') as f:
        pickle.dump(results, f)

    