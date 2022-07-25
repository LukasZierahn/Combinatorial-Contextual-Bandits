import numpy as np
from distribution import Distribution
from distributions.sequence import Sequence

class DistributionBySequence(Distribution):
    def __init__(self, sequence: Sequence) -> None:
        super().__init__()

        self.sequence: Sequence = sequence

    def generate(self, length: int, context_rng: np.random.Generator, theta_rng: np.random.Generator) -> Sequence:
        return self.sequence
