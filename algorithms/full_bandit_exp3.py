from functools import partial
import numpy as np
import numpy.linalg as la

from algorithms.algorithm import Algorithm
from distributions.sequence import Sequence
from misc.matrix_geometric_resampling import matrix_geometric_resampling

# Taken from https://gist.github.com/Gabriel-p/4ddd31422a88e7cdf953
def mvee(points, tol=0.0001):
    """
    Finds the ellipse equation in "center form"
    (x-c).T * A * (x-c) = 1
    """
    N, d = points.shape
    Q = np.column_stack((points, np.ones(N))).T
    err = tol+1.0
    u = np.ones(N)/N
    while err > tol:
        # assert u.sum() == 1 # invariant
        X = np.dot(np.dot(Q, np.diag(u)), Q.T)
        M = np.diag(np.dot(np.dot(Q.T, la.inv(X)), Q))
        jdx = np.argmax(M)
        step_size = (M[jdx]-d-1.0)/((d+1)*(M[jdx]-1.0))
        new_u = (1-step_size)*u
        new_u[jdx] += step_size
        err = la.norm(new_u-u)
        u = new_u
    c = np.dot(u, points)
    A = la.inv(np.dot(np.dot(points.T, np.diag(u)), points)
               - np.multiply.outer(c, c))/d
    return A, c

class FullBanditExp3(Algorithm):

    def __init__(self) -> None:
        self.beta = None
        self.gamma = None
        self.eta = None
        self.ellipsoid_H, self.ellipsoid_center = None, None
        self.boundary_points = None
    
    def distance_to_ellipsoid_boundary(self, point: np.ndarray) -> float:
        assert self.ellipsoid_center != None
        assert self.ellipsoid_H != None

        return (self.ellipsoid_center - point) @ self.ellipsoid_H @ (self.ellipsoid_center - point) - 1

    def find_johns(self, tolerance=1e-5):
        self.ellipsoid_H, self.ellipsoid_center = mvee(self.actionset, tolerance)

        self.boundary_points = []
        for index, point in enumerate(self.actionset):
            if self.distance_to_ellipsoid_boundary(point) <= tolerance:
                self.boundary_points.append(index)
        self.boundary_points = np.array(self.boundary_points)

    def set_constants(self, rng: np.random.Generator, sequence: Sequence):
        self.mgr_rng = rng

        self.actionset = sequence.actionset
        self.find_johns()

        self.theta_estimates = np.zeros((sequence.length, sequence.d, sequence.K))
        self.theta_position = 0

        self.context_unbiased_estimator = sequence.context_unbiased_estimator

        m = np.max(np.sum(self.actionset, axis=1))
        self.beta = 1/(2 * (sequence.sigma**2) * m)
    
        max_term = np.max([sequence.K * sequence.d, sequence.K * m * sequence.sigma**2 / sequence.lambda_min])
        log_term = np.log(sequence.length * m * sequence.sigma**2 * sequence.R**2)
        log_A = np.log(len(self.actionset))

        self.gamma = np.sqrt(log_A * max_term * log_A * log_term / sequence.length)
        assert self.gamma < 1, f"gamma should be smaller than 1 but is {self.gamma}, for {sequence.name}"
        self.eta = np.sqrt(log_A) / (m * np.sqrt(sequence.length * max_term * log_term))

        self.M = int(np.ceil(sequence.K / (2 * self.beta * self.gamma * sequence.lambda_min) * log_term))


    def get_policy(self, context: np.ndarray) -> np.ndarray:
        action_scores = np.einsum("a,bac,ec->e", context, self.theta_estimates[:self.theta_position], self.actionset)

        action_scores = np.exp(-self.eta * action_scores)
        action_scores /= np.sum(action_scores)

        exploration_bonus = np.zeros(len(action_scores))
        exploration_bonus[self.boundary_points] += 1/len(self.boundary_points)

        probabilities = (1 - self.gamma) * action_scores + self.gamma * exploration_bonus
        return probabilities
    
    def observe_loss_vec(self, loss_vec: np.ndarray, context: np.ndarray):

        def unbiased_estimator(k: int, rng: np.random.Generator) -> np.ndarray:
            context_sample = self.context_unbiased_estimator(rng)
            probabilities = self.get_policy(context_sample)
            action_sample_index = rng.choice(np.arange(len(self.actionset)), p=probabilities)

            return self.actionset[action_sample_index, k] * context_sample.reshape(-1, 1) @ context_sample.reshape(1, -1)

        for i in range(self.actionset.K):
            inverse = matrix_geometric_resampling(self.mgr_rng, self.M, self.beta, partial(unbiased_estimator, i))
            self.theta_estimates[self.theta_position, :, i] = inverse @ context * loss_vec[i]

        self.theta_position += 1 

