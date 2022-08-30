import numpy as np
import matplotlib.pyplot as plt

from distributions.distribution_by_sequence import DistributionBySequence
from distributions.distribution import Distribution
from distributions.sequence import Sequence
from algorithms.semi_bandit_exp3 import SemiBanditExp3
from algorithms.full_bandit_exp3 import FullBanditExp3
from algorithms.semi_bandit_ftrl import SemiBanditFTRL
from algorithms.uniform_random import UniformRandom
from experiment_manager.experiment_manager import ExperimentManager

from distributions.actionsets.msets import MSets

from distributions.contexts.binary_context import BinaryContext
from distributions.thetas.single_hole import SingleHole
from distributions.thetas.independent_bernoulli import IndependentBernoulli

import multiprocessing as mp

exp_manager = ExperimentManager()
algos = [FullBanditExp3(), UniformRandom()]

length = 10000
d = 2
K = 2
actionset = MSets(K, 1)

dist_holes = Distribution(BinaryContext(d), SingleHole(K, d, np.array([0.7, 0.3])), actionset)

epsilon = 0.25 * np.min([np.sqrt(K / length), 1])
print("epsilon: ", epsilon)
p = np.zeros((d, K)) + 0.5
for i in range(d):
    p[i, 0] -= epsilon

dist_lower_bound = Distribution(BinaryContext(d), IndependentBernoulli(d, K, p), actionset)

lenghts = [length]
data = exp_manager.run(10, lenghts, algos, [dist_lower_bound], mp.cpu_count())
