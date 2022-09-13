import numpy as np
import matplotlib.pyplot as plt

from distributions.distribution_by_sequence import DistributionBySequence
from distributions.distribution import Distribution
from distributions.sequence import Sequence
from algorithms.semi_bandit_exp3 import SemiBanditExp3
from algorithms.full_bandit_exp3 import FullBanditExp3
from algorithms.semi_bandit_ftrl import SemiBanditFTRL
from algorithms.uniform_random import UniformRandom
from algorithms.non_contextual_exp3 import NonContextualExp3
from algorithms.one_per_context import OnePerContext

from experiment_manager.experiment_manager import ExperimentManager

from distributions.actionsets.msets import MSets

from distributions.contexts.binary_context import BinaryContext
from distributions.thetas.single_hole import SingleHole
from distributions.thetas.independent_bernoulli import IndependentBernoulli

import multiprocessing as mp

def get_dist(rng, d, K, m):
    p = np.zeros((d, K)) + 0.5
    for i in range(d):
        placed_already = []
        while len(placed_already) < m:

            index = rng.integers(K)
            if index not in placed_already:
                placed_already.append(index)
                p[i, index] = 0.3
    
    return IndependentBernoulli(d, K, p)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    exp_manager = ExperimentManager()
    algos = [UniformRandom(), OnePerContext(), NonContextualExp3(), FullBanditExp3()]

    lenghts = [20000]

    distributions = []
    K = 3
    m = 2
    for d in [3, 5, 12]:
            actionset = MSets(K, m)
            distributions.append(Distribution(BinaryContext(d), get_dist(rng, d, K, m), actionset))

    d = 3
    for K in [5, 8]:
            actionset = MSets(K, m)
            distributions.append(Distribution(BinaryContext(d), get_dist(rng, d, K, m), actionset))

    override_constants = [{
        "M": 10
    },
    {
        "M": 1
    }]
    # data = exp_manager.run(1, lenghts, algos, distributions, override_constants, 1)
    # data = exp_manager.run(16, lenghts, algos, distributions, override_constants, mp.cpu_count())

    # data = exp_manager.run_on_existing(algos, override_constants, 1)
    data = exp_manager.run_on_existing(algos, override_constants, mp.cpu_count())
