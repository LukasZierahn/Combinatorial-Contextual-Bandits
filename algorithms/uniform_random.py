import numpy as np
from algorithms.algorithm import Algorithm
from distributions.sequence import Sequence

class UniformRandom(Algorithm):

    def set_constants(self, rng: np.random.Generator, sequence: Sequence):
        self.actionset = sequence.actionset

    def get_policy(self, context: np.ndarray):
        return np.ones(self.actionset.number_of_actions) / self.actionset.number_of_actions

    def observe_loss_vec(self, loss_vec: np.ndarray, context: np.ndarray, action: np.ndarray):
        pass