from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
from distributions.actionsets.actionset import Actionset

from distributions.sequence import Sequence


class Algorithm(ABC):
    def __init__(self) -> None:
        self.full_bandit = False
        
        self.d: int = None
        self.K: int = None
        self.length: int = None
        self.actionset: Actionset = None

        self.context_list: np.ndarray = None

        self.beta: float    = None
        self.gamma: float   = None
        self.eta: float     = None
        self.M: float       = None


    def set_constants(self, rng: np.random.Generator, sequence: Sequence):
        self.rng = rng
        self.d = sequence.d
        self.K = sequence.K
        self.length = sequence.length
        self.actionset = sequence.actionset
        self.context_list = sequence.context_list
        self.sequence = sequence

    @abstractmethod
    def get_policy(self, context: np.ndarray):
        raise NotImplementedError
    
    def observe_loss_vec(self, loss_vec: np.ndarray, context: np.ndarray, action_index: int):
        raise NotImplementedError
    
    def observe_loss(self, loss: float, context: np.ndarray, action_index: int):
        raise NotImplementedError
    
    def run_on_sequence(self, rng: np.random.Generator, sequence: Sequence) -> Tuple[float, np.ndarray]:
        sequence.reset()
        context, _, _, done = sequence.get_next(None)

        losses = []
        probability_array = []
        action_array = []
        while not done:
            probabilities = self.get_policy(context)
            probability_array.append(probabilities)

            action_index = rng.choice(np.arange(sequence.actionset.number_of_actions), p=probabilities)
            action_array.append(action_index)

            next_context, loss, loss_vec, done = sequence.get_next(sequence.actionset[action_index])
            loss_vec[~sequence.actionset[action_index]] = 0
            if self.full_bandit:
                self.observe_loss(loss, context, action_index)
            else:
                self.observe_loss_vec(loss_vec, context, action_index)

            losses.append(loss)
            context = next_context

        return np.sum(losses), np.array(losses), probability_array, action_array