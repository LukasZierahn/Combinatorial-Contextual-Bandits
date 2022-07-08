from abc import abstractmethod
import numpy as np

class Distribution:
    def __init__(self, seed=0) -> None:
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def generate(length) -> np.ndarray:
        pass

