
from abc import ABC, abstractmethod
import numpy as np


class Thetas(ABC):
    def __init__(self, K: int, d: int, true_R: float) -> None:
        self.K: int = K
        self.d: int = d
        self.true_R = true_R

    @abstractmethod
    def generate(self, length: int, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError
