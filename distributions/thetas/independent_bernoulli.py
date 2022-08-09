
import numpy as np

from distributions.thetas.thetas import Thetas


class IndependentBernoulli(Thetas):
    def __init__(self, K: int, d: int, p: np.ndarray=None) -> None:
        """
        p at i, j is chance of having a loss of 1 at i, j
        """
        super().__init__(K, d, d)
        self.p = p
        if isinstance(p, list):
            self.p = np.array(p)

        if p is None:
            self.p = np.ones((d, K))/2
        

    def generate(self, length: int, rng: np.random.Generator) -> np.ndarray:
        randoms = rng.random(size=(length, self.d, self.K))

        return (randoms <= self.p)