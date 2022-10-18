from functools import partial
import numpy as np

from algorithms.algorithm import Algorithm
from distributions.sequence import Sequence
from misc.matrix_geometric_resampling import matrix_geometric_resampling

class ShortestPath(Algorithm):

    def __init__(self) -> None:
        super().__init__()

        self.full_bandit = False

    def set_constants(self, rng: np.random.Generator, sequence: Sequence, override_length: int=None):
        super().set_constants(rng, sequence)
        length = override_length
        if length is None:
            length = sequence.length

        self.actionset = sequence.actionset
        self.exploration_bonus = sequence.actionset.get_exploratory_set()
        number_of_exploratory_actions = np.sum(self.exploration_bonus != 0)

        self.theta_estimate: np.ndarray = np.zeros(sequence.K)
    
        m = self.actionset.m
        self.beta = np.sqrt(sequence.K / (length * len(self.actionset.actionset)) * np.log(len(self.actionset.actionset) / 0.95))

        self.eta = np.sqrt(np.log(length) / (4 * length * sequence.K**2 * number_of_exploratory_actions))

        self.gamma = 2 * self.eta * sequence.K * number_of_exploratory_actions

    def get_policy(self, context: np.ndarray) -> np.ndarray:
        action_scores = np.einsum("c,ec->e", self.theta_estimate, self.actionset.actionset)

        min_score = np.min(action_scores)
        action_scores_exp = np.exp(-self.eta * (action_scores - min_score))
        action_scores_exp /= np.sum(action_scores_exp)

        probabilities = (1 - self.gamma) * action_scores_exp + self.gamma * self.exploration_bonus
        return probabilities
        
    def observe_loss_vec(self, loss_vec: np.ndarray, context: np.ndarray, action_index: int):
        probabilities = self.get_policy(None)
        chance_of_selecting = np.einsum("e,ef->f", probabilities, self.actionset.actionset)

        self.theta_estimate += (loss_vec + self.beta) / chance_of_selecting
