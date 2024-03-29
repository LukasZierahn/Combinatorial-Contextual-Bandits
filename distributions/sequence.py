from typing import Callable, Dict, Tuple
import numpy as np
from distributions.actionsets.actionset import Actionset

from distributions.contexts.context import Context
from distributions.thetas.thetas import Thetas

class Sequence:
    def __init__(self, actionset: Actionset, length=1, d=1) -> None:
        self.actionset = actionset
        self.m = actionset.m

        self.thetas = np.zeros((length, d, self.K))
        self.contexts = np.zeros((length, d))

        self.loss_sequence_cache: np.ndarray    = None
        self.current_index: int                 = 0
        self.rng: np.random.Generator           = None

        self.context_unbiased_estimator: Callable[[np.random.Generator], np.ndarray] = None
        self.context_list: np.ndarray = None

        self.sigma: float       = None
        self.lambda_min: float  = None
        self.R: float           = None

        self.name_prefix = ""

    @property
    def name(self) -> str:
        name_end = f"Length: {self.length}"
        if self.name_prefix == "":
            return name_end
        return f"{self.name_prefix} {name_end}"

    @property
    def d(self) -> int:
        return self.thetas.shape[1]

    @property
    def K(self) -> int:
        return self.actionset.K

    @property
    def length(self) -> int:
        return self.thetas.shape[0]

    @property
    def finished_generating(self) -> bool:
        return self.length == self.current_index        

    @property
    def loss_sequence(self) -> np.ndarray:
        if not self.finished_generating:
            raise Exception("Tried to access loss sequence before sequence finished generating")

        if isinstance(self.loss_sequence_cache, np.ndarray):
            return self.loss_sequence_cache
        
        self.loss_sequence_cache = np.zeros((self.length, self.K))
        for i in range(self.length):
            self.loss_sequence_cache[i] = self.contexts[i].T @ self.thetas[i]

        return self.loss_sequence_cache
        
    def set_contexts(self, contexts: Context, rng: np.random.Generator):
        self.contexts = contexts.generate(self.length, rng)

        self.sigma = contexts.true_sigma
        self.context_unbiased_estimator = contexts.unbiased_sample
        self.context_list = contexts.get_context_probabilities()
        self.lambda_min = contexts.lambda_min

    def set_theta(self, theta: Thetas, rng: np.random.Generator):
        self.thetas = theta.generate(self.length, rng)
        self.R = theta.true_R

    def get_next(self, action: np.ndarray) -> Tuple[np.ndarray, np.float, np.bool]:
        """
        Ai-Gym like function call to get contexts and losses

        :param action: An action usually from the actionset
        :return: The next context, loss for the played action (and given the last context), done
        """

        if self.current_index == 0:
            self.current_index += 1
            return self.contexts[0], None, None, False

        loss_vec = self.contexts[self.current_index].T @ self.thetas[self.current_index]
        loss = loss_vec @ action

        self.current_index += 1
        if self.current_index == self.length:
            return None, loss, loss_vec, True
        return self.contexts[self.current_index], loss, loss_vec, False

    def reset(self, rng: np.random.Generator=None):
        self.rng = rng
        self.loss_sequence_cache = None
        self.current_index = 0

    def evaluate_policy(self, policy: Dict[str, np.ndarray]) -> np.ndarray:
        loss = np.zeros(self.length)
        for i in range(self.length):
            loss[i] = self.loss_sequence[i] @ policy[str(self.contexts[i])]
        return loss


    def find_optimal_policy(self) -> Tuple[float, np.ndarray, Dict[str, np.ndarray]]:
        optimal_policy = {}

        unique_contexts, unique_contexts_indices = np.unique(self.contexts, axis=0, return_inverse=True)
        for context_id in range(len(unique_contexts)):
            sub_sequence = self.loss_sequence[unique_contexts_indices == context_id]

            loss_of_best_action = np.inf
            for action in self.actionset.actionset:
                loss_of_action = np.sum(sub_sequence @ action)
                if loss_of_action < loss_of_best_action:
                    loss_of_best_action = loss_of_action
                    optimal_policy[str(unique_contexts[context_id])] = action

        losses = self.evaluate_policy(optimal_policy)
        return np.sum(losses), losses, optimal_policy
