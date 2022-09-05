
import numpy as np

from distributions.thetas.thetas import Thetas


class SingleHole(Thetas):
    def __init__(self, K: int, d: int, p: np.ndarray=None) -> None:
        super().__init__(K, d, np.sqrt(d - 1))
        self.p = p
        if isinstance(p, list):
            self.p = np.array(p)

        if p is None:
            self.p = np.ones(K)/K

    @property
    def name(self) -> str:
        return f"SingleHole{self.K/self.d}"

    def generate(self, length: int, rng: np.random.Generator) -> np.ndarray:
        thetas = np.ones((length, self.d, self.K))
        set_to_zero = rng.integers(self.K * self.d, size=(length))
        for i in range(length):
            thetas[i, set_to_zero[i]//self.K, set_to_zero[i]%self.K] = 0

        return thetas
