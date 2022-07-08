import numpy as np


class Algorithm():
    def __init__(self, actionset: np.ndarray) -> None:
        self.actionset = actionset # This is \mathcal{A} in the paper

    def get_policy(self, context: np.ndarray):
        raise Exception("Function not implemented")   
    
    def observe_loss_vec(self, loss_vec: np.ndarray, context: np.ndarray):
        raise Exception("Function not implemented")   
    
    def observe_loss(self, loss: np.float, context: np.ndarray):
        raise Exception("Function not implemented")   
