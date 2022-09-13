
import numpy as np

from distributions.thetas.thetas import Thetas


class SingleHole(Thetas):
    def __init__(self, d: int, K: int, p: np.ndarray=None) -> None:
        super().__init__(d, K, np.sqrt(d - 1))
        self.p = p
        if isinstance(p, list):
            self.p = np.array(p)

        if p is None:
            self.p = np.ones(K)/K
        

        self.name = f"SingleHole{self.d};{self.K}"

    def generate(self, length: int, rng: np.random.Generator) -> np.ndarray:
        thetas = np.ones((length, self.d, self.K))
        set_to_zero = rng.integers(self.K * self.d, size=(length))
        for i in range(length):
            thetas[i, set_to_zero[i]//self.K, set_to_zero[i]%self.K] = 0

        return thetas
