from functools import partial
import numpy as np

from algorithms.algorithm import Algorithm
from algorithms.full_bandit_exp3 import FullBanditExp3
from distributions.sequence import Sequence
from misc.matrix_geometric_resampling import matrix_geometric_resampling
from misc.tensor_helpers import *

class FullBanditExp3Inv(FullBanditExp3):

    def observe_loss(self, loss: float, context: np.ndarray, action_index: int):
        K = self.actionset.K
        d = len(context)

        if self.context_list is None:
            raise Exception

        tensor = np.zeros((d, d, K, K))
        for context_probability, context_curr in self.context_list:

            action_probabilities = self.get_policy(context_curr)
            weighted_action = np.einsum("ab,a->b", self.actionset.actionset, action_probabilities)
            action_matrix = np.outer(weighted_action, weighted_action)
            tensor += np.einsum("ab,cd->abcd", context_probability * np.outer(context_curr, context_curr),  action_matrix)

        matrix = tensor_to_matrix(tensor)
        try:
            inverse = np.linalg.inv(matrix)
        except:
            inverse = np.linalg.inv(matrix + np.identity(d*K) * 1e-5)

        self.theta_estimate += loss * np.einsum("abcd,b,c", matrix_to_tensor(inverse, d, K), context, self.actionset[action_index])

class FullBanditTests(FullBanditExp3Inv):
    pass
