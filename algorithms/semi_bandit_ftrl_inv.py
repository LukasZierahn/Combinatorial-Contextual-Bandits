import numpy as np

from algorithms.semi_bandit_ftrl import SemiBanditFTRL
from distributions.sequence import Sequence
from misc.matrix_geometric_resampling import matrix_geometric_resampling


class SemiBanditFTRLInv(SemiBanditFTRL):

    def observe_loss_vec(self, loss_vec: np.ndarray, context: np.ndarray, action_index: int):

        for i in range(self.actionset.K):
            if loss_vec[i] == 0:
                continue

            d = len(context)
            matrix = np.zeros((d, d))
            for context_probability, context_curr in self.context_list:

                action_probabilities = self.get_policy(context_curr)
                prob_weight = np.sum(action_probabilities * np.array(self.actionset.actionset, dtype=float)[:, i])
                matrix += prob_weight * context_probability * np.outer(context_curr, context_curr)

            inverse = np.linalg.inv(matrix)
            self.theta_estimate[:, i] += inverse @ context * loss_vec[i]

    """def observe_loss_vec(self, loss_vec: np.ndarray, context: np.ndarray, action_index: int):

        for i in range(self.actionset.K):
            print("\n")
            if loss_vec[i] == 0:
                print("skipping", i)
                continue

            d = len(context)
            matrix = np.zeros((d, d))
            for context_probability, context_curr in self.context_list:

                action_probabilities = self.get_policy(context_curr)
                print("current i", i, "action", self.actionset[i], "context_probability", context_probability, "context_curr", context_curr)
                print(action_probabilities, np.array(self.actionset.actionset, dtype=float)[:, i])
                prob_weight = np.sum(action_probabilities * np.array(self.actionset.actionset, dtype=float)[:, i])
                print(prob_weight)
                print("to be added", prob_weight * context_probability * np.outer(context_curr, context_curr))
                print("weighting", prob_weight * context_probability)
                matrix += prob_weight * context_probability * np.outer(context_curr, context_curr)
            print("finished matrix", matrix)

            inverse = np.linalg.inv(matrix + np.identity(d) * 1e-3)
            self.theta_estimate[:, i] += inverse @ context * loss_vec[i]"""

