from functools import partial
import numpy as np

from algorithms.algorithm import Algorithm
from algorithms.full_bandit_exp3 import FullBanditExp3
from distributions.sequence import Sequence
from misc.matrix_geometric_resampling import matrix_geometric_resampling
from misc.tensor_helpers import *

class FullBanditExp3InvOld(FullBanditExp3):

    def observe_loss(self, loss: float, context: np.ndarray, action_index: int):
        K = self.actionset.K
        d = len(context)

        tensor = np.zeros((d, d, K, K))
        for i in range(d):
            context_curr = np.zeros(d)
            context_curr[i] = 1

            probabilities = self.get_policy(context_curr)
            weighted_action = np.einsum("ab,a->b", self.actionset.actionset, probabilities)
            action_matrix = np.outer(weighted_action, weighted_action)
            tensor += np.einsum("ab,cd->abcd", np.outer(context_curr, context_curr) / d,  action_matrix)

        matrix = tensor_to_matrix(tensor)
        inverse = np.linalg.inv(matrix + np.identity(d*K) * 1e-3)

        self.theta_estimate += loss * np.einsum("abcd,b,c", matrix_to_tensor(inverse, d, K), context, self.actionset[action_index])
