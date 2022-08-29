from typing import List

import numpy as np
from distributions.distribution import Distribution
from algorithms.algorithm import Algorithm
from distributions.sequence import Sequence

def next_rng(seed_sequence: np.random.SeedSequence):
    seeds = seed_sequence.spawn(2)
    return np.random.default_rng(seeds[0]), seeds[1]

def single_run(rng: np.random.Generator, algorithm: Algorithm, sequence: Sequence) -> float:
    algorithm.set_constants(rng, sequence)
    loss, _ = algorithm.run_on_sequence(rng, sequence)
    loss_of_optimal_policy, _, _ = sequence.find_optimal_policy()
    return loss - loss_of_optimal_policy

def single_run_helper(args) -> float:
    return single_run(*args)

class ExperimentManager:
    def __init__(self) -> None:
        self.seed_sequence = None

    def next_rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed_sequence.spawn(1)[0])

    def generate_sequences(self, seed_sequence: np.random.SeedSequence, iterations: int, lengths: List[int], distributions: List[Distribution]) -> np.ndarray:
        sequences = np.zeros((len(distributions), len(lengths), iterations), dtype=object)
        for dist_index, dist in enumerate(distributions):
            for length_index, length in enumerate(lengths):
                for iteration in range(iterations):
                    context_seed, theta_seed, seed_sequence = seed_sequence.spawn(3)
                    sequences[dist_index, length_index, iteration] = dist.generate(length, np.random.default_rng(context_seed), np.random.default_rng(theta_seed))
        return sequences
        
        
    def run(self, iterations: int, lengths: List[int], algorithms: List[Algorithm], distributions: List[Distribution], seed: int=0) -> np.ndarray:
        seed_sequence = np.random.SeedSequence(seed)

        sequences = self.generate_sequences(seed_sequence.spawn(1)[0], iterations, lengths, distributions)
        results = np.zeros((len(algorithms), len(distributions), len(lengths), iterations), dtype=float).reshape(len(algorithms), -1)

        for alg_index, algorithm in enumerate(algorithms):
            for index, seq in enumerate(sequences.flatten()):
                rng, seed_sequence = next_rng(seed_sequence)
                results[alg_index][index] = single_run(rng, algorithm, seq)
        

        return results.reshape((len(algorithms), len(distributions), len(lengths), iterations))



