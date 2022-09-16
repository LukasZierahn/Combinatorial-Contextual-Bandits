from functools import partial
import numpy as np

from algorithms.algorithm import Algorithm
from distributions.sequence import Sequence
from misc.matrix_geometric_resampling import matrix_geometric_resampling
from misc.tensor_helpers import *

class FullBanditExp3(Algorithm):

    def __init__(self) -> None:
        self.exploration_bonus: np.ndarray = None
        self.full_bandit = True
    
    def set_constants(self, rng: np.random.Generator, sequence: Sequence):
        super().set_constants(rng, sequence)

        self.exploration_bonus = self.actionset.get_johns()

        self.theta_estimate = np.zeros((sequence.d, sequence.K))
        self.context_unbiased_estimator = sequence.context_unbiased_estimator

        m = self.actionset.m
        self.beta = 1/(2 * (sequence.sigma**2) * m)
    
        max_term = np.max([sequence.K * sequence.d, sequence.K * m * sequence.sigma**2 / sequence.lambda_min])
        log_term = np.log(sequence.length * m * sequence.sigma**2 * sequence.R**2)
        log_A = np.log(self.actionset.number_of_actions)

        self.gamma = np.sqrt(log_A * max_term * log_term / sequence.length)
        assert self.gamma < 1, f"FullBanditExp3 gamma should be smaller than 1 but is {self.gamma}, for {sequence.name}"
        self.eta = np.sqrt(log_A) / (m * np.sqrt(sequence.length * max_term * log_term))

        self.M = int(np.ceil(sequence.K / (2 * self.beta * self.gamma * sequence.lambda_min) * log_term))


    def get_policy(self, context: np.ndarray) -> np.ndarray:
        action_scores = np.einsum("a,ac,ec->e", context, self.theta_estimate, self.actionset.actionset)

        min_score = np.min(action_scores)
        action_scores_exp = np.exp(-self.eta * (action_scores - min_score))
        action_scores_exp /= np.sum(action_scores_exp)

        probabilities = (1 - self.gamma) * action_scores_exp + self.gamma * self.exploration_bonus
        return probabilities
    
    def observe_loss(self, loss: float, context: np.ndarray, action_index: int):

        def unbiased_estimator(rng: np.random.Generator) -> np.ndarray:
            context_sample = self.context_unbiased_estimator(rng)

            probabilities = self.get_policy(context_sample)
            action_sample_index = rng.choice(np.arange(self.actionset.number_of_actions), p=probabilities)
            action = np.array(self.actionset[action_sample_index], dtype=float)

            tensor = np.einsum("a,b,c,d->abcd", context_sample, context_sample, action, action)
            return tensor_to_matrix(tensor)

        inverse = matrix_geometric_resampling(self.rng, self.M, self.beta, unbiased_estimator)

        self.theta_estimates += loss * np.einsum("abcd,b,c", matrix_to_tensor(inverse, self.d, self.actionset.K), context, self.actionset[action_index])

