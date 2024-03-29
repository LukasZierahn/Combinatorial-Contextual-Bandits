from typing import Dict
import numpy as np

from algorithms.algorithm import Algorithm
from distributions.sequence import Sequence

from algorithms.shortest_path import ShortestPath

class OnePerContextSP(Algorithm):

    def __init__(self) -> None:
        super().__init__()

        self.full_bandit = False
        self.context_algos: dict[str, Algorithm] = {}
        self.context_count: int = None

    def set_constants(self, rng: np.random.Generator, sequence: Sequence):
        super().set_constants(rng, sequence)

        self.context_count = len(sequence.context_list)

    def get_policy(self, context: np.ndarray) -> np.ndarray:
        hash = str(context)
        if hash not in self.context_algos:
            new_algo = ShortestPath()
            new_algo.set_constants(self.rng, self.sequence, override_length=self.sequence.length/self.context_count)
            
            new_algo.gamma = self.gamma if self.gamma is not None else new_algo.gamma
            new_algo.eta = self.eta if self.eta is not None else new_algo.eta

            self.context_algos[hash] = new_algo
        
        return self.context_algos[hash].get_policy(None)
        
    
    def observe_loss_vec(self, loss_vec: np.ndarray, context: np.ndarray, action_index: int):
        hash = str(context)
        self.context_algos[hash].observe_loss_vec(loss_vec, None, action_index)
    
    def observe_loss(self, loss: float, context: np.ndarray, action_index: int):
        hash = str(context)
        self.context_algos[hash].observe_loss(loss, None, action_index)
