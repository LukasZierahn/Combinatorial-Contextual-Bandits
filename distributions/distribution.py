import numpy as np
from distributions.contexts.context import Context

from distributions.sequence import Sequence

class Distribution():
    def __init__(self, Context: Context) -> None:
        pass

    def generate(self, length: int, context_rng: np.random.Generator, theta_rng: np.random.Generator) -> Sequence:
        raise NotImplementedError

