import numpy as np
from distributions.actionsets.actionset import Actionset
from math import comb

from scipy.optimize import minimize

def generate_mset(K: int, number_of_ones: int) -> np.ndarray:
    if number_of_ones == K:
        return np.ones((1, K), dtype=bool)

    if number_of_ones == 0:
        return np.zeros((1, K), dtype=bool)


    leading_false = np.append(generate_mset(K - 1, number_of_ones), np.zeros((comb(K-1, number_of_ones), 1), dtype=bool), axis=1)
    leading_true = np.append(generate_mset(K - 1, number_of_ones - 1), np.ones((comb(K-1, number_of_ones - 1), 1), dtype=bool), axis=1)
    return np.append(leading_false, leading_true, axis=0)


class MSets(Actionset):

    def __init__(self, K, number_of_ones) -> None:
        super().__init__(generate_mset(K, number_of_ones))
        self.number_of_ones = number_of_ones

    def get_johns(self):
        return np.ones(len(self.actionset)) / len(self.actionset)

    def get_exploratory_set(self):
        buffer = []
        for i in range(1 + self.K // self.number_of_ones):
            if i == self.K // self.number_of_ones and self.K / self.number_of_ones == self.K // self.number_of_ones:
                continue

            next_vec = np.zeros(self.K, dtype=bool)
            next_vec[i * self.number_of_ones : np.min([(i + 1) * self.number_of_ones, self.K])] = True
            next_vec[ : np.max([(i + 1) * self.number_of_ones - self.K, 0])] = True

            index = (np.all(self.actionset == next_vec, axis=1)).nonzero()[0][0]
            buffer.append(index)

        return np.array(buffer)

    def ftrl_routine(self, context: np.ndarray, rng: np.random.Generator, ftrl_algorithm):
        actions = np.exp(-1 * ftrl_algorithm.eta * np.einsum("a,bac->c", context, ftrl_algorithm.theta_estimate))
        return actions / np.sum(actions)
        

    def ftrl_routine_slow(self, context: np.ndarray, rng: np.random.Generator, ftrl_algorithm):
        if self.number_of_ones != 1:
            raise Exception(f"tried to call not ftrl_routine on MSets when number_of_ones = {self.number_of_ones} which is not 1")

        def fun(mu):
            action_score = np.einsum("a,bac,c->", context, ftrl_algorithm.theta_estimates[:ftrl_algorithm.theta_position], mu)
            return action_score + ftrl_algorithm.regulariser(mu)

        bnds = []
        for _ in range(self.K):
            bnds.append((0, 1))

        cons = {'type':'eq', 'fun': lambda x: np.sum(x) - 1}

        x0 = rng.random(self.K)
        x0 /= np.sum(x0)
        sol = minimize(fun, x0=x0, bounds=bnds, constraints=cons)

        return sol.x
