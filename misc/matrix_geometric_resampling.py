from typing import Callable
import numpy as np

def matrix_geometric_resampling(rng: np.random.Generator, M: int, beta: float, unbiased_estimator: Callable[[np.random.Generator], np.ndarray]):
    estimate = unbiased_estimator(rng)
    if estimate.shape[0] != estimate.shape[1]:
        raise Exception("Matrix submitted to matrix_geometric_resampling is not square")
    size = estimate.shape[0]

    A = np.zeros((M, size, size))
    A[0] = np.identity(size) - beta * estimate

    for i in range(1, M):
        A[i] = A[i - 1] @ (np.identity(size) - beta * unbiased_estimator(rng))

    return beta * np.identity(size) + beta * np.sum(A, axis=0)

