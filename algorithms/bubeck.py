import numpy as np

from algorithms.semi_bandit_ftrl_inv import SemiBanditFTRLInv
from distributions.sequence import Sequence
from misc.matrix_geometric_resampling import matrix_geometric_resampling

class Bubeck(SemiBanditFTRLInv):

    def set_constants(self, rng: np.random.Generator, sequence: Sequence, override_length: int=None):
        super().set_constants(rng, sequence)
        length = override_length
        if length is None:
            length = sequence.length

        self.actionset = sequence.actionset
        self.exploratory_set = self.actionset.get_exploratory_set()

        self.theta_estimate: np.ndarray = np.zeros(sequence.K)

        self.context_unbiased_estimator = sequence.context_unbiased_estimator

        self.gamma = np.sqrt(sequence.K / (sequence.m * length))

        self.eta = np.sqrt(sequence.m * np.log(sequence.K / sequence.m)) / np.sqrt(length * sequence.K)

    def get_policy(self, context: np.ndarray) -> np.ndarray:
        optimal_action = np.exp(-1 * self.eta * self.theta_estimate)
        action_scores = self.actionset.ftrl_routine(optimal_action)
        probabilities = (1 - self.gamma) * action_scores + self.gamma * self.exploratory_set

        return probabilities

    def observe_loss_vec(self, loss_vec: np.ndarray, context: np.ndarray, action_index: int):
        probabilities = self.get_policy(None)
        chance_of_selecting = np.einsum("e,ef->f", probabilities, self.actionset.actionset)

        self.theta_estimate += loss_vec / chance_of_selecting
