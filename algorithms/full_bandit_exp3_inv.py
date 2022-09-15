from functools import partial
import numpy as np

from algorithms.algorithm import Algorithm
from algorithms.full_bandit_exp3 import FullBanditExp3
from distributions.sequence import Sequence
from misc.matrix_geometric_resampling import matrix_geometric_resampling

class FullBanditExp3Inv(FullBanditExp3):

    def observe_loss(self, loss: float, context: np.ndarray, action_index: int):

        K = self.actionset.K
        d = len(context)

        action_matrix = np.zeros((K, K))
        for i in range(d):
            context = np.zeros(d)
            context[i] = 1

            probabilities = self.get_policy(context)
            weighted_action = np.einsum("ab,a->b", self.actionset.actionset, probabilities)
            action_matrix += np.outer(weighted_action, weighted_action) / d

        tensor = np.einsum("ab,cd->cadb", np.identity(d)/d,  action_matrix).reshape((d*K, d*K))
        inverse = np.linalg.inv(tensor + np.identity(d*K) * 1e-5)
        inverse_tensor = inverse.reshape((self.d, self.d, self.K, self.K))


        self.theta_estimates[self.theta_position] = loss * np.einsum("abcd,b,c", inverse_tensor, context, self.actionset[action_index])
        self.theta_position += 1 

