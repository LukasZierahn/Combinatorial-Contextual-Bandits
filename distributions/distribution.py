from abc import ABC, abstractmethod
import numpy as np

from distributions.sequence import Sequence

class Distribution(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def generate(length: int, context_rng: np.random.Generator, theta_rng: np.random.Generator) -> Sequence:
        raise NotImplementedError

