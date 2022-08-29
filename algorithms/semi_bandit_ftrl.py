from functools import partial
import numpy as np

from algorithms.algorithm import Algorithm
from distributions.sequence import Sequence
from misc.matrix_geometric_resampling import matrix_geometric_resampling


class SemiBanditFTRL(Algorithm):

    def __init__(self) -> None:
        super().__init__()

        self.beta: float    = None
        self.gamma: float   = None
        self.eta: float     = None
        self.M: float       = None

    def regulariser(self, action: np.ndarray) -> float:
        return 1/self.eta * np.sum(action * np.log(action + 1e-6) - action)

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
        one_plus_log = 1 + np.log(sequence.K / m)
        max_term = np.max([sequence.K * sequence.d, len(self.exploratory_set) * m * sequence.sigma**2 * one_plus_log / (self.beta * sequence.lambda_min * np.log(2))])
        log_term = np.log(np.sqrt(sequence.length) * m * sequence.sigma * sequence.R)

        self.gamma = np.sqrt(max_term / (sequence.length * m * one_plus_log))
        if self.gamma > 1: raise Exception(f"gamma should be smaller than 1 but is {self.gamma}, for {sequence.name}")
        self.eta = np.sqrt(m * one_plus_log / (sequence.length * max_term))

        self.M = int(np.ceil(len(self.exploratory_set) * log_term / (self.gamma * self.beta * sequence.lambda_min)))


    def get_policy(self, context: np.ndarray) -> np.ndarray:
        if self.theta_position == 0:
            return np.ones(self.K) / self.K

        action_scores = self.actionset.ftrl_routine(context, self.mgr_rng, self)
        exploration_bonus = self.actionset.get_exploratory_set()

        probabilities = (1 - self.gamma) * action_scores + self.gamma * exploration_bonus
        return probabilities

    
    def observe_loss_vec(self, loss_vec: np.ndarray, context: np.ndarray, action_index: int):

        def unbiased_estimator(k: int, rng: np.random.Generator) -> np.ndarray:
            context_sample = self.context_unbiased_estimator(rng)
            probabilities = self.get_policy(context_sample)
            action_sample_index = rng.choice(np.arange(len(self.actionset.actionset)), p=probabilities)

            return self.actionset[action_sample_index, k] * context_sample.reshape(-1, 1) @ context_sample.reshape(1, -1)

        for i in range(self.actionset.K):
            inverse = matrix_geometric_resampling(self.mgr_rng, self.M, self.beta, partial(unbiased_estimator, i))
            self.theta_estimates[self.theta_position, :, i] = inverse @ context * loss_vec[i]

        self.theta_position += 1 

