from abc import ABC, abstractmethod

import numpy as np

from distributions.sequence import Sequence


class Context(ABC):
    def __init__(self, d: int, true_sigma: float) -> None:
        self.true_sigma: float = true_sigma
        self.d = d

    def generate(self, length: int, rng: np.random.Generator) -> np.ndarray:
        result = np.zeros((length, self.d))
        for i in range(length):
            result[i] = self.unbiased_sample(rng)
        return result

    @abstractmethod
    def unbiased_sample(self, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

