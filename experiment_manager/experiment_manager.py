from msilib import sequence
from typing import List

import numpy as np
from distributions.distribution import Distribution
from algorithms.algorithm import Algorithm
from distributions.sequence import Sequence

def single_run(algorithm: Algorithm, sequence: Sequence):
    pass

def single_run_helper(args):
    return single_run(*args)

class ExperimentManager:
    def __init__(self) -> None:
        self.seed_sequence = None

    def next_rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed_sequence.spawn(1))

    def generate_sequences(self, seed_sequence: np.random.SeedSequence, iterations: int, lengths: List[int], distributions: List[Distribution]) -> np.ndarray:
        sequences = np.zeros((len(distributions), len(lengths), iterations), dtype=object)
        for dist_index, dist in enumerate(distributions):
            for length_index, length in enumerate(lengths):
                for iteration in range(iterations):
                    sequences[dist_index, length_index, iteration] = dist.generate(length, np.random.default_rng(seed_sequence.spawn(1)))
        return sequences
        
        
    def run(self, iterations: int, lengths: List[int], algorithms: List[Algorithm], distributions: List[Distribution], seed: int=0) -> np.ndarray:
        self.seed_sequence = np.random.SeedSequence(seed)

        sequences = self.generate_sequences(self.next_rng(), iterations, lengths, distributions)
        results = np.zeros((len(algorithms), len(distributions), len(lengths), iterations), dtype=float).reshape(len(algorithms), -1)

        for alg_index, algorithm in enumerate(algorithms):
            for index, seq in enumerate(sequences.flatten()):
                results[alg_index][index] = algorithm.run_on_sequence(self.next_rng(), seq)[0]
            

        self.seed_sequence = None
        return results



