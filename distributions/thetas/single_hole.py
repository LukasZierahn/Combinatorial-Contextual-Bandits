
import numpy as np

from distributions.thetas.thetas import Thetas


class SingleHole(Thetas):
    def __init__(self, K: int, d: int, p: np.ndarray=None) -> None:
        self.K: int = K
        self.d = d
        self.p = p
        if p == None:
            self.p = np.ones(K)/K

    def generate(self, length: int, rng: np.random.Generator) -> np.ndarray:
        thetas = np.ones((length, self.d, self.K))
        set_to_zero = rng.integers(self.K * self.d, size=(length))
        for i in range(length):
            thetas[i, set_to_zero[i]//self.K, set_to_zero[i]%self.K] = 0

        return thetas
