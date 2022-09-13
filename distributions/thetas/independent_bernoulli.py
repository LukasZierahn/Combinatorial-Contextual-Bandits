
import numpy as np

from distributions.thetas.thetas import Thetas


class IndependentBernoulli(Thetas):
    def __init__(self, d: int, K: int, p: np.ndarray=None) -> None:
        """
        p at i, j is chance of having a loss of 1 at i, j
        """
        super().__init__(d, K, d)
        self.p = p
        if isinstance(p, list):
            self.p = np.array(p)

        if p is None:
            self.p = np.ones((d, K))/2
        
        self.name = f"IndependentBernoulli{self.d};{self.K}"


    def generate(self, length: int, rng: np.random.Generator) -> np.ndarray:
        randoms = rng.random(size=(length, self.d, self.K))

        return (randoms <= self.p)