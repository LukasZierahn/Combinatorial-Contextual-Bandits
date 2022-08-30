from typing import List
import numpy as np
from distributions.actionsets.actionset import Actionset

from scipy.optimize import minimize

def generate_pathfinding(current_position: np.ndarray, edges_visited: np.ndarray) -> np.ndarray:
    print("inp", current_position, edges_visited)
    if np.all(current_position == 0):
        return [edges_visited.flatten()]

    buffer = []
    for i in range(len(current_position)):
        if current_position[i] == 0:
            continue

        current_position[i] -= 1
        edges_visited[current_position] = True
        buffer.extend(generate_pathfinding(current_position.copy(), edges_visited.copy()))
        edges_visited[current_position] = False
        current_position[i] += 1
    
    return buffer

def generate_pathfinding_2d(x: int , y: int, x_edgdes: np.ndarray, y_edges: np.ndarray) -> np.ndarray:
    #print("inp", x, y, edges_visited)
    if x == 0 and y == 0:
        print("found", x_edgdes, y_edges)
        return [np.append(x_edgdes.flatten(), y_edges.flatten())]

    buffer = []
    if x != 0:
        new_x = x - 1
        x_edgdes[new_x, y] = True
        buffer.extend(generate_pathfinding_2d(new_x, y, x_edgdes.copy(), y_edges.copy()))
        x_edgdes[new_x, y] = False
    if y != 0:
        new_y = y - 1
        y_edges[x, new_y] = True
        buffer.extend(generate_pathfinding_2d(x, new_y, x_edgdes.copy(), y_edges.copy()))
        y_edges[x, new_y] = False
    
    return buffer


class Pathfinding(Actionset):

    def __init__(self, x: int, y: int) -> None:
        x_edges = np.zeros((x - 1, y), dtype=bool)
        y_edges = np.zeros((x, y - 1), dtype=bool)
        list_of_actions = generate_pathfinding_2d(x - 1, y - 1, x_edges, y_edges)
        super().__init__(np.array(list_of_actions))

    def get_johns(self):
        if self.m != 1:
            raise Exception(f"tried to call not get_johns on MSets when m = {self.m} which is not 1")
        
        return np.ones(self.K) / self.K

    def ftrl_routine(self, context: np.ndarray, rng: np.random.Generator, ftrl_algorithm):
        actions = np.exp(-1 * ftrl_algorithm.eta * np.einsum("a,bac->c", context, ftrl_algorithm.theta_estimates[:ftrl_algorithm.theta_position]))
        return actions / np.sum(actions)
        

    def ftrl_routine_slow(self, context: np.ndarray, rng: np.random.Generator, ftrl_algorithm):
        if self.m != 1:
            raise Exception(f"tried to call not ftrl_routine on MSets when m = {self.m} which is not 1")

        def fun(mu):
            action_score = np.einsum("a,bac,c->", context, ftrl_algorithm.theta_estimates[:ftrl_algorithm.theta_position], mu)
            return action_score + ftrl_algorithm.regulariser(mu)

        bnds = []
        for _ in range(self.K):
            bnds.append((0, 1))

        cons = {'type':'eq', 'fun': lambda x: np.sum(x) - 1}

        x0 = rng.random(self.K)
        x0 /= np.sum(x0)
        sol = minimize(fun, x0=x0, bounds=bnds, constraints=cons)

        return sol.x
