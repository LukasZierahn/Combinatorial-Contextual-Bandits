from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

from distributions.sequence import Sequence


class Algorithm(ABC):
    def __init__(self, actionset: np.ndarray) -> None:
        self.actionset = actionset # This is \mathcal{A} in the paper

    @abstractmethod
    def get_policy(self, context: np.ndarray):
        raise Exception("Function not implemented")   
    
    def observe_loss_vec(self, loss_vec: np.ndarray, context: np.ndarray):
        raise Exception("Function not implemented")   
    
    def observe_loss(self, loss: np.float, context: np.ndarray):
        raise Exception("Function not implemented")   
    
    def run_on_sequence(self, rng: np.random.Generator, sequence: Sequence) -> Tuple[float, np.ndarray]:
        sequence.reset()
        context, _, _, done = sequence.get_next(None)

        losses = []
        while not done:
            probabilities = self.get_policy(context)
            action_index = rng.choice(np.arange(len(self.actionset)), p=probabilities)
            context, loss, loss_vec, done = sequence.get_next(self.actionset[action_index])
            
            if not done:
                self.observe_loss_vec(loss_vec, context)

            losses.append(loss)

        return np.sum(losses), np.array(losses)