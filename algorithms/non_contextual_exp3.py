from functools import partial
import numpy as np

from algorithms.algorithm import Algorithm
from distributions.sequence import Sequence
from misc.matrix_geometric_resampling import matrix_geometric_resampling

class NonContextualExp3(Algorithm):

    def __init__(self, full_bandit=True) -> None:
        super().__init__()

        self.full_bandit = full_bandit

    def set_constants(self, rng: np.random.Generator, sequence: Sequence):
        super().set_constants(rng, sequence)
        self.mgr_rng = rng

        self.actionset = sequence.actionset
        self.theta_estimates: np.ndarray = np.zeros((sequence.length, sequence.K))
        self.theta_position = 0

        self.exploratory_set = sequence.actionset.get_exploratory_set()
    
        m = self.actionset.m
        denominator = sequence.length * (self.K / m) + 2 * self.K
        log_actionset = np.log(len(self.actionset.actionset))

        self.gamma = self.K * np.sqrt(m * log_actionset / denominator)
        if self.gamma > 1: raise Exception(f"gamma should be smaller than 1 but is {self.gamma}, for {sequence.name}")
        self.eta = np.sqrt(log_actionset / (m * denominator))

    def get_policy(self, context: np.ndarray) -> np.ndarray:
        action_scores = np.einsum("ac,ec->e", self.theta_estimates[:self.theta_position], self.actionset.actionset)

        min_score = np.min(action_scores)
        action_scores_exp = np.exp(-self.eta * (action_scores - min_score))
        action_scores_exp /= np.sum(action_scores_exp)

        exploration_bonus = np.zeros(len(action_scores))
        exploration_bonus[self.exploratory_set] += 1/len(self.exploratory_set)

        probabilities = (1 - self.gamma) * action_scores_exp + self.gamma * exploration_bonus
        return probabilities
    
    def observe_loss_vec(self, loss_vec: np.ndarray, context: np.ndarray, action_index: int):
        self.theta_estimates[self.theta_position] = loss_vec
        self.theta_position += 1 
    
    def observe_loss(self, loss: float, context: np.ndarray, action_index: int):
        probabilities = self.get_policy(None)
        P = np.einsum("e,ef,eg->fg", probabilities, self.actionset.actionset, self.actionset.actionset)

        self.theta_estimates[self.theta_position] = loss * np.linalg.inv(P) @ self.actionset[action_index]
        self.theta_position += 1 
