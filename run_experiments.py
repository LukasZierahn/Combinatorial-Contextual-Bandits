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

if __name__ == "__main__":
    exp_manager = ExperimentManager()
    algos = [UniformRandom(), OnePerContext(), NonContextualExp3()]

    lenghts = [10000]

    distributions = []
    for d in [3, 5, 12]:
        for K in [5, 8, 12]:

            actionset = MSets(K, 3)

            # dist_holes = Distribution(BinaryContext(d), SingleHole(K, d, np.array([0.7, 0.3])), actionset)

            epsilon = 0.25 * np.min([np.sqrt(K / lenghts[-1]), 1])
            epsilon = 0.02
            print("epsilon: ", epsilon)
            p = np.zeros((d, K)) + 0.5
            for i in range(d):
                p[i, 0] -= epsilon

            # p = np.zeros((d, K)) + 0.1
            # p[0, 0] = 0.9
            # p[1, 0] = 0.9

            distributions.append(Distribution(BinaryContext(d), IndependentBernoulli(d, K, p), actionset))

    # data = exp_manager.run(10, lenghts, algos, [dist_lower_bound], 1)
    data = exp_manager.run(10, lenghts, algos, distributions, mp.cpu_count())
