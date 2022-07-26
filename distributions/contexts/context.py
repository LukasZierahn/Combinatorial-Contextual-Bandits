from abc import ABC, abstractmethod

import numpy as np

class Context(ABC):
    def __init__(self, d: int, true_sigma: float, lambda_min: float) -> None:
        self.true_sigma: float = true_sigma
        self.d: int = d
        self.lambda_min: float = lambda_min

    def generate(self, length: int, rng: np.random.Generator) -> np.ndarray:
        result = np.zeros((length, self.d))
        for i in range(length):
            result[i] = self.unbiased_sample(rng)
        return result

    @abstractmethod
    def unbiased_sample(self, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

