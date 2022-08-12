import numpy as np

from algorithms.semi_bandit_ftrl import SemiBanditFTRL


class Actionset():
    def __init__(self, actionset: np.ndarray) -> None:
        self.actionset = actionset
        self.m: int = None

    @property
    def K(self) -> int:
        return self.actionset.shape[1]

    def get_exploratory_set(self):
        return np.arange(len(self.actionset))

    def get_johns(self):
        raise Exception("tried to call not implemented function get_johns")
    
    def ftrl_routine(self, context: np.ndarray, ftrl_algorithm: SemiBanditFTRL):
        def fun(mu):
            action = np.sum(mu * self.actionset, axis=1)
            action_score = np.einsum("a,bac,c", context, ftrl_algorithm.theta_estimates[:ftrl_algorithm.theta_position], self.actionset)
            return action_score + ftrl_algorithm.regulariser(action)


    def __getitem__(self, key):
        return self.actionset[key]
