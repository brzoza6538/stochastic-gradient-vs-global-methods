from .globals import *
import numpy as np 

class CMAES:
    def __init__(
            self,
            f_objective,
            dimension, 
            mean=None, 
            sigma=None,
            lamb=None,
            min_clamp=def_clamps[0],
            max_clamp=def_clamps[1],
            objective_limit=None,
            smallest_val=def_smallest_val,
            max_fes=def_max_fes,
            checkpoints=def_checkpoints,
            ):

        self.f_objective = f_objective
        self.dimension = dimension
        self.min_clamp = min_clamp
        self.max_clamp = max_clamp
        self.max_fes = max_fes
        self.smallest_val = smallest_val
        self.checkpoints = checkpoints

        self.log = {checkpoint: [] for checkpoint in self.checkpoints}
        self.objective_counter = 0
        self.seen_checkpoints = set()

        self.lamb = lamb or int(4 + np.floor(3 * np.log(self.dimension)))
        self.mu = self.lamb // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mueff = np.sum(self.weights)**2 / np.sum(self.weights**2)

        self.cc = (4 + self.mueff / self.dimension) / (self.dimension + 4 + 2 * self.mueff / self.dimension)
        self.cs = (self.mueff + 2) / (self.dimension + self.mueff + 5)
        self.c1 = 2 / ((self.dimension + 1.3)**2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1/self.mueff) / ((self.dimension + 2)**2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff-1)/(self.dimension+1))-1) + self.cs

        if mean is None:
            self.m = mean or np.random.uniform(self.min_clamp, self.max_clamp, size=self.dimension)
        else:
            self.m = mean
        self.objective_limit = objective_limit or self.dimension * self.max_fes

        self.s = sigma or (self.max_clamp - self.min_clamp) / 3
        self.Ps = np.zeros(self.dimension)
        self.Pc = np.zeros(self.dimension)
        self.B = np.eye(self.dimension)
        self.D = np.ones(self.dimension)
        self.C = self.B @ np.diag(self.D ** 2) @ self.B.T
        self.eigeneval = 0
        self.chiN = np.sqrt(self.dimension) * (1 - 1/(4*self.dimension) + 1/(21*self.dimension**2))

        self.error = None

    def start(self):
        while (self.objective_counter < self.objective_limit) and (self.error is None or self.error > self.smallest_val):
            self.step()
            self.collect_data()

    def step(self):
        offspring = []
        fitness = []
        for k in range(self.lamb):
            child = self.m + self.s * (self.B @ (self.D * np.random.randn(self.dimension)))
            child = np.clip(child, self.min_clamp, self.max_clamp)
            offspring.append(child)
            eval, evals_used = self.f_objective(offspring[k])
            fitness.append(eval)
            self.objective_counter += evals_used

        offspring = np.array(offspring)  
        sorted_idx = np.argsort(fitness)  

        offspring = offspring[sorted_idx]
        fitness = np.array(fitness)[sorted_idx]

        m_old = np.copy(self.m)
        self.m = np.sum(self.weights[:, np.newaxis] * offspring[:self.mu], axis=0)

        delta_sigma = (self.m - m_old) / self.s
        self.Ps = (1 - self.cs) * self.Ps + self.cs * np.sqrt(self.mueff) * np.sqrt(1 - (1 - self.cs) ** 2) * delta_sigma
        hsig = (np.linalg.norm(self.Ps) / np.sqrt(1 - (1 - self.cs)**(2 * self.objective_counter / self.lamb)) / self.chiN) < (1.4 + 2 / (self.dimension + 1))
        self.Pc = (1 - self.cc) * self.Pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * (self.m - m_old) / self.s

        artmp = (1 / self.s) * (offspring[:self.mu] - m_old)
        artmp = artmp.T
        self.C = (1 - self.c1 - self.cmu) * self.C \
            + self.c1 * (np.outer(self.Pc, self.Pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C) \
            + self.cmu * (artmp @ np.diag(self.weights) @ artmp.T)

        self.s = self.s * np.exp((self.cs / self.damps) * (np.linalg.norm(self.Ps) / self.chiN - 1))

        if self.objective_counter - self.eigeneval > self.lamb / ((self.c1 + self.cmu) * self.dimension / 10):
            self.eigeneval = self.objective_counter
            self.C = np.triu(self.C) + np.triu(self.C, 1).T
            D_matrix, self.B = np.linalg.eigh(self.C)
            self.D = np.sqrt(np.maximum(D_matrix, self.smallest_val))

        self.error = fitness[0]

    def collect_data(self):
        for checkpoint in self.checkpoints:
            checkpoint_fes = int(checkpoint * self.objective_limit)
            if self.error < self.smallest_val and self.objective_counter <= checkpoint_fes:
                self.log[checkpoint].append(0)
            if checkpoint not in self.seen_checkpoints and self.objective_counter >= checkpoint_fes:
                self.log[checkpoint].append(0 if self.error < self.smallest_val else self.error)
                self.seen_checkpoints.add(checkpoint)

