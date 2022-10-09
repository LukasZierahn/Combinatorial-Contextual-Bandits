from typing import Dict
import numpy as np

from algorithms.algorithm import Algorithm
from algorithms.one_per_context import OnePerContext
from distributions.sequence import Sequence

from algorithms.non_contextual_exp3 import NonContextualExp3

class OnePerContextSB(OnePerContext):

    def __init__(self) -> None:
        super().__init__(False)