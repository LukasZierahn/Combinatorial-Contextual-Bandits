from genericpath import isfile
import os
import shutil
from typing import Dict, List, Union

import numpy as np
from distributions.distribution import Distribution
from algorithms.algorithm import Algorithm
from distributions.sequence import Sequence

import time
import datetime
import pickle
import json
import re

def next_rng(seed_sequence: np.random.SeedSequence):
    seeds = seed_sequence.spawn(2)
    return np.random.default_rng(seeds[0]), seeds[1]

def single_run_helper(args):
    return single_run(*args)

def single_run(rng: np.random.Generator, algorithm: Algorithm, sequence: Sequence, override_constant: Dict[str, float], output_dir: str="") -> float:
    algorithm.set_constants(rng, sequence)
    for key, value in override_constant.items():
        setattr(algorithm, key, value)

    if os.path.isfile(f"{output_dir}_general_info.json"):
        return 0

    print(f"Starting {output_dir} {datetime.datetime.now()}")
    start = time.time()

    loss, losses, probability_array, action_array = algorithm.run_on_sequence(rng, sequence)
    end = time.time()
    print(f"Finishing {output_dir} {end - start}")
    
    loss_of_optimal_policy, _, _ = sequence.find_optimal_policy()

    if output_dir != "":
        # np.savetxt(f"{output_dir}_losses.csv", losses) 
        # np.savetxt(f"{output_dir}_probability_array.csv", probability_array) 
        # np.savetxt(f"{output_dir}_action_array.csv", action_array)

        general_info = {
            "regret": loss - loss_of_optimal_policy,
            "time_elapsed": end - start,
            "gamma": algorithm.gamma,
            "eta": algorithm.eta,
            "beta": algorithm.beta,
            "M": algorithm.M,
            "is_full_bandit": algorithm.full_bandit
        }
        with open(f"{output_dir}_general_info.json", "w") as output_file:
            json.dump(general_info, output_file)
        

def single_run_helper(args) -> float:
    return single_run(*args)

class ExperimentManager:
    def __init__(self) -> None:
        self.seed_sequence = None

    def run_on_existing(self, algorithms: List[Algorithm], override_constants: List[Dict[str, float]] = [{}], number_of_processes: int=1, seed: int=1) -> np.ndarray:
        seed_sequence = np.random.SeedSequence(seed)
        
        args = []
        distributions = os.listdir(f"output/")
        for dist_index, dist in enumerate(distributions):
            lengths = os.listdir(f"output/{dist}")

            for length_index, length in enumerate(lengths):
                iterations = os.listdir(f"output/{dist}/{length}")

                for iteration in iterations:
                    
                    for alg_index, algorithm in enumerate(algorithms):
                        for override_constant in override_constants:

                            algo_name = re.findall(r"\..*\.(.*)'", str(algorithm.__class__))[0]
                            for key, value in override_constant.items():
                                algo_name += f"{key}={value}"
                            
                            # Make sure we generate an rng if we use the algorithm or not to make sure that the seeds stay aligned
                            rng, seed_sequence = next_rng(seed_sequence)

                            if os.path.isfile(f"output/{dist}/{length}/{iteration}/{algo_name}_general_info.json"):
                                continue

                            with open(f"output/{dist}/{length}/{iteration}/sequence.json", "rb") as input_file:
                                sequence = pickle.load(input_file)


                                output_dir = f"output/{dist}/{length}/{iteration}/{algo_name}"

                                if int(iteration) < 10:
                                    args.append((rng, algorithm, sequence, override_constant, output_dir))

        print("Starting", len(args), "runs")
        if number_of_processes == 1:
            for arg in args:
                single_run_helper(arg)
        else:
            import multiprocessing as mp
            with mp.Pool(number_of_processes) as pool:
                pool.map(single_run_helper, args)
                        
    def generate_sequences(self, seed_sequence: np.random.SeedSequence, iterations: int, lengths: List[int], distributions: List[Distribution]) -> np.ndarray:
        sequences = np.zeros((len(distributions), len(lengths), iterations), dtype=object)
        for dist_index, dist in enumerate(distributions):
            for length_index, length in enumerate(lengths):
                for iteration in range(iterations):
                    context_seed, theta_seed, seed_sequence = seed_sequence.spawn(3)
                    sequences[dist_index, length_index, iteration] = dist.generate(length, np.random.default_rng(context_seed), np.random.default_rng(theta_seed))
        return sequences
    
    def create_output_dir(self, iterations: int, lengths: List[int], distributions: List[Distribution], seed: int=0) -> np.ndarray:
        if "output" in os.listdir():
            print("Ouput already exists, not creating any new sequences")
            return

        print(f"Creating sequences with seed {seed}")
        seed_sequence = np.random.SeedSequence(seed)
        sequences = self.generate_sequences(seed_sequence.spawn(1)[0], iterations, lengths, distributions)

        os.mkdir("output")
        for dist_index, dist in enumerate(distributions):
            os.mkdir(f"output/{dist.name}")

            for length_index, length in enumerate(lengths):
                os.mkdir(f"output/{dist.name}/{length}")

                for iteration in range(iterations):
                    os.mkdir(f"output/{dist.name}/{length}/{iteration}")
                    with open(f"output/{dist.name}/{length}/{iteration}/sequence.json", "wb") as output_file:
                        pickle.dump(sequences[dist_index, length_index, iteration], output_file)
