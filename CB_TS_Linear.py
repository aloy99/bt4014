import numpy as np
import scipy.stats as stats

class ContextualThompsonSampling:
    def __init__(self, n_arms, n_features, delta=0.5,
                 R=0.01, epsilon=0.5, random_state=456):
        self.n_arms = n_arms
        self.n_features = n_features
        self.random_state = random_state
        self.n_features = n_features

        # 0 < delta < 1
        if not isinstance(delta, float):
            raise ValueError("delta should be float")
        elif (delta < 0) or (delta >= 1):
            raise ValueError("delta should be in (0, 1]")
        else:
            self.delta = delta

        # R > 0
        if not isinstance(R, float):
            raise ValueError("R should be float")
        elif R <= 0:
            raise ValueError("R should be positive")
        else:
            self.R = R

        # 0 < epsilon < 1
        if not isinstance(epsilon, float):
            raise ValueError("epsilon should be float")
        elif (epsilon < 0) or (epsilon > 1):
            raise ValueError("epsilon should be in (0, 1)")
        else:
            self.epsilon = epsilon

        self.A = [np.identity(n_features) for _ in range(n_arms)]
        self.b = [np.zeros(n_features) for _ in range(n_arms)]

    def select_arm(self, T, context):
        scores = np.zeros(self.n_arms)
        thetas = [np.zeros(self.n_features) for _ in range(self.n_arms)]
        for arm in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[arm])
            mu_hat = A_inv @ self.b[arm]
            v = self.R * np.sqrt(24 / self.epsilon
                             * self.n_features
                             * np.log(T / self.delta))
            mu_tilde = np.random.multivariate_normal(
            mu_hat.flat, v**2 * A_inv)[..., np.newaxis]

            scores[arm] = context @ mu_tilde
            thetas[arm] = mu_hat
        
        selected_arm = np.argmax(scores)
        return selected_arm, thetas, scores

    def update(self, arm, context, reward):
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context

class Simulator:
    def __init__(self, n_trials, n_arms, n_features, good_arms, good_bias):
        self.n_trials = n_trials
        self.n_arms = n_arms
        self.n_features = n_features
        self.good_arms = good_arms
        self.good_bias = good_bias
        self.X = [np.zeros(n_features) for _ in range(n_trials)]

    def simulate_context_matrix(self, n_trials, n_arms, n_features):
        X = np.array([np.random.uniform(low=0, high = 1, size=n_features) for _ in np.arange(n_trials)])
        self.X = X
        return X

    def simulate_theta(self, n_arms, n_features, good_arms, good_bias = 1) :
        true_theta = np.array([np.random.normal(size = n_features, scale = 1/4) for _ in np.arange(n_arms)])
        if len(good_arms)>0:
            true_theta[good_arms] = true_theta[good_arms] + good_bias
        self.true_theta = true_theta
        return true_theta

    def initialization(self):
        self.simulate_context_matrix(self.n_trials, self.n_arms, self.n_features)
        self.simulate_theta(self.n_arms, self.n_features, self.good_arms,self.good_bias)


def simulate_one_time_reward(x, theta, scale_noise = 0.01):
    signal = theta @ x
    noise  = np.random.normal(scale = scale_noise) 
    return (signal + noise)


# Example usage
## Start running over 1000 trials
# np.random.seed(42)
n_trials = 1000
n_arms = 4
n_features = 5
good_arms = [2]



np.random.seed(4014)


# Let's simulate some interaction with users
my_simulation = Simulator(n_trials, n_arms, n_features, good_arms, 0)
my_simulation.initialization()
cts = ContextualThompsonSampling(n_arms, n_features)

results = dict()
# theta_history is to let me know the estimations for theta values over time
theta_history  = np.empty(shape=(n_trials, n_arms, n_features))
# score_history is to let me know the estimations for the upper bound of each arm
score_history  = np.empty(shape=(n_trials, n_arms))
# arm_selection_history and r_payoff are to let me know the arms I have chosen and the rewards I have received over time
arm_selection_history, r_payoff = [np.empty(n_trials) for _ in range(2)]

play_rewards = 0

for t in np.arange(1, n_trials+1):
    user_context = my_simulation.X[t]
    chosen_arm, theta_history[t],score_history[t]  = cts.select_arm(t, user_context)
    true_reward = np.dot(user_context, my_simulation.true_theta[chosen_arm])
    reward_observed = simulate_one_time_reward(x=user_context, theta=my_simulation.true_theta[chosen_arm])
    play_rewards = play_rewards + reward_observed
    arm_selection_history[t] = chosen_arm

    cts.update(chosen_arm, user_context, reward_observed)
    print("Iteration {}: the reward_observed is {}, and the chosen arm is {}".format(t, reward_observed, chosen_arm))

results = dict(theta_history=theta_history, upper_bound_score_history=score_history, arm_selection_history=arm_selection_history, r_payoff_history=r_payoff)
print("Estimation of average per-step reward:", play_rewards / 100)