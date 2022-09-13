
from abc import ABC, abstractmethod
import numpy as np


class Thetas(ABC):
    def __init__(self, d: int, K: int, true_R: float) -> None:
        self.d: int = d
        self.K: int = K
        self.true_R = true_R

        self.name = f"Thetas{self.d}/{self.K}"

    @abstractmethod
    def generate(self, length: int, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError
