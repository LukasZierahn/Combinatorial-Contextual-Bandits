from functools import partial
from typing import Callable
import numpy as np

from algorithms.algorithm import Algorithm
from misc.matrix_geometric_resampling import matrix_geometric_resampling

class SemiBanditExp3(Algorithm):

    def __init__(self, actionset: np.ndarray, seed=0) -> None:
        super().__init__(actionset)

        self.actionset = actionset
        self.exploratory_set = np.arange(len(self.actionset))

        self.mgr_rng = np.random.default_rng(seed)

        self.theta_estimates = None
        self.theta_position = 0

        self.beta = None
        self.gamma = None
        self.eta = None
        self.d = None

    @property
    def K(self):
        return self.actionset.shape[1]

    def set_constants(self, length: int, sigma: float, m: float, K: float, d: float, lambda_min: float, R: float, context_unbiased_estimator: Callable[[np.random.Generator], np.ndarray]):
        self.d = d
        self.theta_estimates: np.ndarray = np.zeros((length, d, K))
        self.theta_position = 0

        self.context_unbiased_estimator = context_unbiased_estimator

        self.beta = 1/(2 * (sigma**2))
    
        max_term = np.max([m * K * d, len(self.exploratory_set) * m / (self.beta * lambda_min)])
        log_term = np.log(np.sqrt(length) * m * sigma * R)
        log_A = np.log(len(self.actionset))

        self.gamma = np.sqrt(max_term * log_A * log_term / length)
        self.eta = np.sqrt(log_A / (length * max_term * log_term))

        self.M = np.int(np.ceil(len(self.exploratory_set) * m / (self.beta * lambda_min) * log_term))


    def get_policy(self, context: np.ndarray) -> np.ndarray:
        exploration_bonus = np.zeros(self.actionset.shape[0])
        exploration_bonus[self.exploratory_set] += 1/len(self.exploratory_set)

        action_scores = np.einsum("a,bac,ec->e", context, self.theta_estimates[:self.theta_position], self.actionset)
        if self.theta_position == 1 and False:
            print(context, self.theta_estimates[:self.theta_position], self.actionset)
            print(action_scores)
            1/0 

        action_scores = np.exp(-self.eta * action_scores)
        action_scores /= np.sum(action_scores)

        probabilities = (1 - self.gamma) * action_scores + self.gamma * exploration_bonus
        return probabilities
    
    def observe_loss_vec(self, loss_vec: np.ndarray, context: np.ndarray):

        def unbiased_estimator(k: int, rng: np.random.Generator) -> np.ndarray:
            context_sample = self.context_unbiased_estimator(rng)
            probabilities = self.get_policy(context_sample)
            action_sample_index = np.random.choice(np.arange(len(self.actionset)), p=probabilities)

            return self.actionset[action_sample_index, k] * context_sample.reshape(-1, 1) @ context_sample.reshape(1, -1)

        for i in range(self.K):
            inverse = matrix_geometric_resampling(self.mgr_rng, self.M, self.beta, partial(unbiased_estimator, i))
            self.theta_estimates[self.theta_position, :, i] = inverse @ context * loss_vec[i]

        self.theta_position += 1 

