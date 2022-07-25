
from abc import ABC, abstractmethod
import numpy as np


class Thetas(ABC):
    def __init__(self, K: int, d: int) -> None:
        self.K: int = K
        self.d = d

    def generate(self, length: int, rng: np.random.Generator) -> np.ndarray:
        result = np.zeros((length, self.d))
        for i in range(length):
            result[i] = self.unbiased_sample(rng)
        return result

    @abstractmethod
    def unbiased_sample(self, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

