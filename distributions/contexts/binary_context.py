import numpy as np
from distributions.contexts.context import Context

from distributions.actionsets.msets import generate_mset

class BinaryContext(Context):
    def __init__(self, d: int, number_of_ones: int=1) -> None:
        super().__init__(d, np.sqrt(number_of_ones), number_of_ones/d)
        self.number_of_ones = number_of_ones

    @property
    def name(self) -> str:
        return f"BinaryContext{self.number_of_ones};{self.d}"

    def unbiased_sample(self, rng: np.random.Generator) -> np.ndarray:
        result = np.zeros(self.d)
        result[rng.choice(self.d, size=self.number_of_ones, replace=False)] = 1
        return result

    def get_context_probabilities(self):
        all_contexts = generate_mset(self.d, self.number_of_ones)
        probabilities = np.ones(len(all_contexts)) / len(all_contexts)

        buffer = []
        for index in range(len(all_contexts)):
            buffer.append([probabilities[index], all_contexts[index]])

        return buffer