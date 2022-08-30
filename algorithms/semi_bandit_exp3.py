from functools import partial
import numpy as np

from algorithms.algorithm import Algorithm
from distributions.sequence import Sequence
from misc.matrix_geometric_resampling import matrix_geometric_resampling

class SemiBanditExp3(Algorithm):

    def __init__(self) -> None:
        super().__init__()

        self.beta: float    = None
        self.gamma: float   = None
        self.eta: float     = None
        self.M: float       = None

    def set_constants(self, rng: np.random.Generator, sequence: Sequence):
        super().set_constants(rng, sequence)
        self.mgr_rng = rng

        self.actionset = sequence.actionset
        self.exploratory_set = self.actionset.get_exploratory_set()

        self.theta_estimates: np.ndarray = np.zeros((sequence.length, sequence.d, sequence.K))
        self.theta_position = 0

        self.context_unbiased_estimator = sequence.context_unbiased_estimator

        self.beta = 1/(2 * (sequence.sigma**2))
    
        m = self.actionset.m
        max_term = np.max([m * sequence.K * sequence.d, len(self.exploratory_set) * m / (self.beta * sequence.lambda_min)])
        log_term = np.log(np.sqrt(sequence.length) * m * sequence.sigma * sequence.R)
        log_A = np.log(self.actionset.number_of_actions)

        self.gamma = np.sqrt(max_term * log_A * log_term / sequence.length)
        if self.gamma > 1: raise Exception(f"gamma should be smaller than 1 but is {self.gamma}, for {sequence.name}")
        self.eta = np.sqrt(log_A / (sequence.length * max_term * log_term))

        self.M = int(np.ceil(len(self.exploratory_set) * m / (self.beta * sequence.lambda_min) * log_term))


    def get_policy(self, context: np.ndarray) -> np.ndarray:
        action_scores = np.einsum("a,bac,ec->e", context, self.theta_estimates[:self.theta_position], self.actionset.actionset)

        min_score = np.min(action_scores)
        action_scores = np.exp(-self.eta * (action_scores - min_score))
        action_scores /= np.sum(action_scores - min_score)

        exploration_bonus = np.zeros(len(action_scores))
        exploration_bonus[self.exploratory_set] += 1/len(self.exploratory_set)

        probabilities = (1 - self.gamma) * action_scores + self.gamma * exploration_bonus
        return probabilities
    
    def observe_loss_vec(self, loss_vec: np.ndarray, context: np.ndarray, action_index: int):

        def unbiased_estimator(k: int, rng: np.random.Generator) -> np.ndarray:
            context_sample = self.context_unbiased_estimator(rng)
            probabilities = self.get_policy(context_sample)
            action_sample_index = rng.choice(np.arange(self.actionset.number_of_actions), p=probabilities)

            return self.actionset[action_sample_index, k] * context_sample.reshape(-1, 1) @ context_sample.reshape(1, -1)

        for i in range(self.actionset.K):
            inverse = matrix_geometric_resampling(self.mgr_rng, self.M, self.beta, partial(unbiased_estimator, i))
            self.theta_estimates[self.theta_position, :, i] = inverse @ context * loss_vec[i]

        self.theta_position += 1 

