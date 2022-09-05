import os
import shutil
from typing import List

import numpy as np
from distributions.distribution import Distribution
from algorithms.algorithm import Algorithm
from distributions.sequence import Sequence

import time
import pickle
import json
import re

def next_rng(seed_sequence: np.random.SeedSequence):
    seeds = seed_sequence.spawn(2)
    return np.random.default_rng(seeds[0]), seeds[1]

def single_run_helper(args):
    return single_run(*args)

def single_run(rng: np.random.Generator, algorithm: Algorithm, sequence: Sequence, output_dir: str="") -> float:
    print(f"Starting {output_dir}/{algo_name}")
    start = time.time()
    algorithm.set_constants(rng, sequence)
    loss, losses, probability_array, action_array = algorithm.run_on_sequence(rng, sequence)
    end = time.time()
    loss_of_optimal_policy, _, _ = sequence.find_optimal_policy()
    print(f"Finishing {output_dir}/{algo_name} {end - start}")

    if output_dir != "":
        algo_name = re.findall(r"\..*\.(.*)'", str(algorithm.__class__))[0]
        np.savetxt(f"{output_dir}/{algo_name}_losses.csv", losses) 
        np.savetxt(f"{output_dir}/{algo_name}_probability_array.csv", probability_array) 
        np.savetxt(f"{output_dir}/{algo_name}_action_array.csv", action_array)

        general_info = {
            "regret": loss - loss_of_optimal_policy,
            "time_elapsed": end - start,
            "gamma": algorithm.gamma,
            "eta": algorithm.eta,
            "beta": algorithm.beta,
            "M": algorithm.M,
            "is_full_bandit": algorithm.full_bandit
        }
        with open(f"{output_dir}/{algo_name}_general_info.json", "w") as output_file:
            json.dump(general_info, output_file)
        

    return loss - loss_of_optimal_policy

def single_run_helper(args) -> float:
    return single_run(*args)

class ExperimentManager:
    def __init__(self) -> None:
        self.seed_sequence = None

    def generate_sequences(self, seed_sequence: np.random.SeedSequence, iterations: int, lengths: List[int], distributions: List[Distribution]) -> np.ndarray:
        sequences = np.zeros((len(distributions), len(lengths), iterations), dtype=object)
        for dist_index, dist in enumerate(distributions):
            for length_index, length in enumerate(lengths):
                for iteration in range(iterations):
                    context_seed, theta_seed, seed_sequence = seed_sequence.spawn(3)
                    sequences[dist_index, length_index, iteration] = dist.generate(length, np.random.default_rng(context_seed), np.random.default_rng(theta_seed))
        return sequences
        
        
    def run(self, iterations: int, lengths: List[int], algorithms: List[Algorithm], distributions: List[Distribution], number_of_processes: int=1, seed: int=0) -> np.ndarray:
        seed_sequence = np.random.SeedSequence(seed)

        sequences = self.generate_sequences(seed_sequence.spawn(1)[0], iterations, lengths, distributions)
        results = np.zeros((len(algorithms), len(distributions), len(lengths), iterations), dtype=float).reshape(len(algorithms), -1)

        if "output" in os.listdir():
            shutil.rmtree("output")
        os.mkdir("output")
        

        args = []
        for dist_index, dist in enumerate(distributions):
            os.mkdir(f"output/{dist.name}")

            for length_index, length in enumerate(lengths):
                os.mkdir(f"output/{dist.name}/{length}")

                for iteration in range(iterations):
                    os.mkdir(f"output/{dist.name}/{length}/{iteration}")
                    with open(f"output/{dist.name}/{length}/{iteration}/sequence.json", "wb") as output_file:
                        pickle.dump(sequences[dist_index, length_index, iteration], output_file)

                    for alg_index, algorithm in enumerate(algorithms):
                        rng, seed_sequence = next_rng(seed_sequence)

                        output_dir = f"output/{dist.name}/{length}/{iteration}"
                        args.append((rng, algorithm, sequences[dist_index, length_index, iteration], output_dir))
        
        result_list = []
        if number_of_processes == 1:
            for arg in args:
                result_list.append(single_run_helper(arg))
        else:
            import multiprocessing as mp
            with mp.Pool(number_of_processes) as pool:
                result_list = pool.map(single_run_helper, args)

        for alg_index, algorithm in enumerate(algorithms):
            for index, seq in enumerate(sequences.flatten()):
                results[alg_index][index] = result_list[alg_index * len(sequences) + index]
        

        return results.reshape((len(algorithms), len(distributions), len(lengths), iterations))



