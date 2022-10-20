import numpy as np
import numpy.linalg as la

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

def distance_to_ellipsoid_boundary(ellipsoid_H, ellipsoid_center, point: np.ndarray) -> float:
    return (ellipsoid_center - point) @ ellipsoid_H @ (ellipsoid_center - point) - 1

class Actionset():
    def __init__(self, actionset: np.ndarray) -> None:
        self.actionset = actionset
        self.m: int = np.max(np.sum(self.actionset, axis=1))

    @property
    def K(self) -> int:
        return self.actionset.shape[1]

    @property
    def number_of_actions(self) -> int:
        return self.actionset.shape[0]

    def get_exploratory_set(self):
        return np.ones(len(self.actionset)) / len(self.actionset)

    def get_johns(self):
        raise NotImplementedError()

    def find_boundary_points(self):
        tolerance = 1e-6
        ellipsoid_H, ellipsoid_center = mvee(self.actionset, tolerance)

        boundary_points = []
        for index, point in enumerate(self.actionset):
            if distance_to_ellipsoid_boundary(ellipsoid_H, ellipsoid_center, point) <= tolerance:
                boundary_points.append(index)
        return np.array(boundary_points)
    
    def ftrl_routine(self, optimal_action):
        raise NotImplementedError()


    def __getitem__(self, key):
        return self.actionset[key]
