import numpy as np
from algorithms.semi_bandit_ftrl import SemiBanditFTRL
from distributions.actionsets.actionset import Actionset
from math import comb

def generate_mset(K: int, m: int) -> np.ndarray:
    if m == K:
        return np.ones((1, K), dtype=bool)

    if m == 0:
        return np.zeros((1, K), dtype=bool)


    leading_false = np.append(generate_mset(K - 1, m), np.zeros((comb(K-1, m), 1), dtype=bool), axis=1)
    leading_true = np.append(generate_mset(K - 1, m - 1), np.ones((comb(K-1, m - 1), 1), dtype=bool), axis=1)
    return np.append(leading_false, leading_true, axis=0)


class MSets(Actionset):

    def __init__(self, K, m) -> None:
        super().__init__(generate_mset(K, m))
        self.m = m

    def get_johns(self):
        if self.m != 1:
            raise Exception(f"tried to call not get_johns on MSets when m = {self.m} which is not 1")
        
        return np.ones(self.K) / self.K