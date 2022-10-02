import numpy as np
import matplotlib.pyplot as plt
from algorithms.full_bandit_exp3_inv import FullBanditExp3Inv
from algorithms.real_lin_exp3 import RealLinExp3
from algorithms.semi_bandit_ftrl_inv import SemiBanditFTRLInv

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
                p[i, index] -= 0.1
    
    return IndependentBernoulli(d, K, p)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    exp_manager = ExperimentManager()
    algos = [UniformRandom(), OnePerContext(), NonContextualExp3(), RealLinExp3(), SemiBanditFTRLInv(), FullBanditExp3Inv()]
    algos.reverse()

    lenghts = [100000]

    distributions = []
    for d, number_of_ones in [(3, 1), (5, 2), (12, 3)]:
        for K, m in [(3, 1), (5, 2), (8, 3)]:
            actionset = MSets(K, m)
            distributions.append(Distribution(BinaryContext(d, number_of_ones), get_dist(rng, d, K, m), actionset))

    override_constants = [{
    }]
    
    exp_manager.create_output_dir(25, lenghts, distributions)
    # data = exp_manager.run_on_existing(algos, override_constants, 1)
    data = exp_manager.run_on_existing(algos, override_constants, mp.cpu_count())
