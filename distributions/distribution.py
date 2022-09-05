import numpy as np
import distributions.actionsets.actionset
from distributions.contexts.context import Context
from distributions.thetas.thetas import Thetas

from distributions.sequence import Sequence

class Distribution():
    def __init__(self, context: Context, thetas: Thetas, actionset: distributions.actionsets.actionset.Actionset) -> None:
        self.context = context
        self.thetas = thetas
        self.actionset = actionset
        
    @property
    def name(self) -> str:
        return f"{self.context.name}_{self.thetas.name}"

    def generate(self, length: int, context_rng: np.random.Generator, theta_rng: np.random.Generator) -> Sequence:
        seq = Sequence(self.actionset, length, self.context.d)
        
        seq.set_contexts(self.context, context_rng)
        seq.set_theta(self.thetas, theta_rng)
        
        return seq
        

