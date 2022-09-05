from typing import Dict
import numpy as np

from algorithms.algorithm import Algorithm
from distributions.sequence import Sequence

from algorithms.non_contextual_exp3 import NonContextualExp3

class OnePerContext(Algorithm):

    def __init__(self, full_bandit=True) -> None:
        super().__init__()

        self.full_bandit = full_bandit
        self.context_algos: dict[str, Algorithm] = {}

    def set_constants(self, rng: np.random.Generator, sequence: Sequence):
        super().set_constants(rng, sequence)

    def get_policy(self, context: np.ndarray) -> np.ndarray:
        hash = str(context)
        if hash not in self.context_algos:
            new_algo = NonContextualExp3(self.full_bandit)
            new_algo.set_constants(self.rng, self.sequence)
            self.context_algos[hash] = new_algo
        
        return self.context_algos[hash].get_policy(None)
        
    
    def observe_loss_vec(self, loss_vec: np.ndarray, context: np.ndarray, action_index: int):
        hash = str(context)
        self.context_algos[hash].observe_loss_vec(loss_vec, None, action_index)
    
    def observe_loss(self, loss: float, context: np.ndarray, action_index: int):
        hash = str(context)
        self.context_algos[hash].observe_loss(loss, None, action_index)
