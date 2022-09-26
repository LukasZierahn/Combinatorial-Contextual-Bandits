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
            for j in range(d):
                context_curr = np.zeros(d)
                context_curr[j] = 1

                probabilities = self.get_policy(context_curr)
                prob_weight = np.sum(probabilities * np.array(self.actionset.actionset, dtype=float)[:, i])
                matrix += prob_weight * np.outer(context_curr, context_curr) / d

            inverse = np.linalg.inv(matrix + np.identity(d) * 1e-3)
            self.theta_estimate[:, i] += inverse @ context * loss_vec[i]

