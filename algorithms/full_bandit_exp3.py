from functools import partial
import numpy as np

from algorithms.algorithm import Algorithm
from distributions.sequence import Sequence
from misc.matrix_geometric_resampling import matrix_geometric_resampling

class FullBanditExp3(Algorithm):

    def __init__(self) -> None:
        self.beta = None
        self.gamma = None
        self.eta = None

        self.exploration_bonus: np.ndarray = None

        self.full_bandit = True
    
    def set_constants(self, rng: np.random.Generator, sequence: Sequence):
        super().set_constants(rng, sequence)
        self.mgr_rng = rng

        self.exploration_bonus = self.actionset.get_johns()

        self.theta_estimates = np.zeros((sequence.length, sequence.d, sequence.K))
        self.theta_position = 0

        self.context_unbiased_estimator = sequence.context_unbiased_estimator

        m = self.actionset.m
        self.beta = 1/(2 * (sequence.sigma**2) * m)
    
        max_term = np.max([sequence.K * sequence.d, sequence.K * m * sequence.sigma**2 / sequence.lambda_min])
        log_term = np.log(sequence.length * m * sequence.sigma**2 * sequence.R**2)
        log_A = np.log(self.actionset.number_of_actions)

        self.gamma = np.sqrt(log_A * max_term * log_A * log_term / sequence.length)
        assert self.gamma < 1, f"gamma should be smaller than 1 but is {self.gamma}, for {sequence.name}"
        self.eta = np.sqrt(log_A) / (m * np.sqrt(sequence.length * max_term * log_term))

        self.M = int(np.ceil(sequence.K / (2 * self.beta * self.gamma * sequence.lambda_min) * log_term))


    def get_policy(self, context: np.ndarray) -> np.ndarray:
        if self.theta_position == 0:
            return np.ones(self.actionset.number_of_actions) / self.actionset.number_of_actions

        action_scores = np.einsum("a,bac,ec->e", context, self.theta_estimates[:self.theta_position], self.actionset.actionset)

        #min_score = np.min(action_scores)
        min_score = 0
        action_scores = np.exp(-self.eta * (action_scores - min_score))
        action_scores /= np.sum(action_scores - min_score)

        probabilities = (1 - self.gamma) * action_scores + self.gamma * self.exploration_bonus
        return probabilities
    
    def observe_loss(self, loss: float, context: np.ndarray, action_index: int):

        def unbiased_estimator(rng: np.random.Generator) -> np.ndarray:
            context_sample = self.context_unbiased_estimator(rng)
            probabilities = self.get_policy(context_sample)
            action_sample_index = rng.choice(np.arange(self.actionset.number_of_actions), p=probabilities)

            tensor = np.einsum("a,b,c,d->abcd", context_sample, context_sample,  self.actionset[action_sample_index],  self.actionset[action_sample_index])
            output_matrix_length = len(context_sample) * len(self.actionset[action_sample_index])
            return tensor.reshape(output_matrix_length, output_matrix_length)

        inverse = matrix_geometric_resampling(self.mgr_rng, self.M, self.beta, unbiased_estimator)
        inverse_tensor = inverse.reshape((self.d, self.d, self.K, self.K))

        self.theta_estimates[self.theta_position] = loss * np.einsum("abcd,b,c", inverse_tensor, context, self.actionset[action_index])
        self.theta_position += 1 

