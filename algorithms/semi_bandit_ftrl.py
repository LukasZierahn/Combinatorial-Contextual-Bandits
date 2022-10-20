from functools import partial
import numpy as np

from algorithms.algorithm import Algorithm
from distributions.sequence import Sequence
from misc.matrix_geometric_resampling import matrix_geometric_resampling


class SemiBanditFTRL(Algorithm):

    def __init__(self) -> None:
        super().__init__()
        
    def regulariser(self, action: np.ndarray) -> float:
        return 1/self.eta * np.sum(action * np.log(action + 1e-6) - action)

    def set_constants(self, rng: np.random.Generator, sequence: Sequence):
        super().set_constants(rng, sequence)

        self.actionset = sequence.actionset
        self.exploratory_set = self.actionset.get_exploratory_set()

        self.theta_estimate: np.ndarray = np.zeros((sequence.d, sequence.K))

        self.context_unbiased_estimator = sequence.context_unbiased_estimator

        self.beta = 1/(sequence.sigma**2)
        E = len(self.exploratory_set)
        one_plus_log = 1 + np.log(sequence.K / self.actionset.m)

        self.gamma = sequence.sigma**2 * one_plus_log * E *  np.log(sequence.length)
        self.gamma /= sequence.length * self.beta * sequence.lambda_min
        self.gamma = np.sqrt(self.gamma)

        self.M = E * np.log(sequence.length) / (self.gamma * self.beta * sequence.lambda_min)

        eta1 = np.log(2) / (sequence.sigma**2 * (self.M + 1))
        eta2 = np.sqrt((self.actionset.m * one_plus_log) / (3 * sequence.K * sequence.d * sequence.length))
        self.eta = np.min([eta1, eta2])

    def get_policy(self, context: np.ndarray) -> np.ndarray:
        optimal_action = np.exp(-1 * self.eta * np.einsum("a,ac->c", context, self.theta_estimate))

        action_scores = self.actionset.ftrl_routine(optimal_action)
        probabilities = (1 - self.gamma) * action_scores + self.gamma * self.exploratory_set
        return probabilities

    
    def observe_loss_vec(self, loss_vec: np.ndarray, context: np.ndarray, action_index: int):

        def unbiased_estimator(k: int, rng: np.random.Generator) -> np.ndarray:
            context_sample = self.context_unbiased_estimator(rng)
            probabilities = self.get_policy(context_sample)
            action_sample_index = rng.choice(np.arange(len(self.actionset.actionset)), p=probabilities)

            return self.actionset[action_sample_index, k] * context_sample.reshape(-1, 1) @ context_sample.reshape(1, -1)

        for i in range(self.actionset.K):
            inverse = matrix_geometric_resampling(self.rng, self.M, self.beta, partial(unbiased_estimator, i))
            self.theta_estimate[:, i] += inverse @ context * loss_vec[i]

