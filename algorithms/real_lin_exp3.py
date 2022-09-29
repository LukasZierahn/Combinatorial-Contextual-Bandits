from functools import partial
import numpy as np

from algorithms.algorithm import Algorithm
from distributions.sequence import Sequence
from misc.matrix_geometric_resampling import matrix_geometric_resampling

class RealLinExp3(Algorithm):

    def __init__(self) -> None:
        super().__init__()
        self.full_bandit = True

    def set_constants(self, rng: np.random.Generator, sequence: Sequence):
        super().set_constants(rng, sequence)

        self.actionset = sequence.actionset
        self.theta_estimates: np.ndarray = np.zeros((len(self.actionset.actionset), sequence.d))

        self.beta = 1/(2 * (sequence.sigma**2))
    
        number_of_actions = len(self.actionset.actionset)
        log_term = np.log(sequence.length * sequence.sigma**2 * sequence.R**2)

        self.gamma = np.sqrt(log_term / sequence.length)
        if self.gamma > 1: raise Exception(f"gamma should be smaller than 1 but is {self.gamma}, for {sequence.name}")
        self.eta = np.sqrt(np.log(number_of_actions) / (self.sequence.d * number_of_actions * sequence.length * log_term))

        self.M = int(np.ceil(number_of_actions * sequence.sigma**2 * log_term / (self.gamma * sequence.lambda_min)))
        #if self.eta > 2 / (self.M + 1): raise Exception(f"self.eta > 2 / (self.M + 1) violated for eta {self.eta} and M {self.M} ({2 / (self.M + 1)})")


    def get_policy(self, context: np.ndarray) -> np.ndarray:
        action_scores = np.einsum("a,ca->c", context, self.theta_estimates)

        min_score = np.min(action_scores)
        action_scores_exp = np.exp(-self.eta * (action_scores - min_score))
        action_scores_exp /= np.sum(action_scores_exp)

        exploration_bonus = np.ones(len(action_scores)) / len(action_scores)

        probabilities = (1 - self.gamma) * action_scores_exp + self.gamma * exploration_bonus
        return probabilities
    
    def observe_loss(self, loss: float, context: np.ndarray, action_index: int):
        d = len(context)
        matrix = np.zeros((d, d))
        for context_probability, context_curr in self.context_list:
            
            probabilities = self.get_policy(context_curr)
            matrix += probabilities[action_index] * context_probability * np.outer(context_curr, context_curr)


        inverse = np.linalg.inv(matrix + np.identity(d) * 1e-3)
        self.theta_estimates[action_index] += inverse @ context * loss
