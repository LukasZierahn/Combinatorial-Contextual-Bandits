import numpy as np

from distributions.sequence import Sequence

class Distribution:
    def __init__(self) -> None:
        pass

    def generate(length: int, rng: np.random.Generator) -> Sequence:
        raise Exception("Function not implemented")   

