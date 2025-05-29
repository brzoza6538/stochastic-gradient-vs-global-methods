# https://keras.io/api/optimizers/adagrad/
import numpy as np
from .globals import *

class Adagrad():
    def __init__(
            self,
            f_objective,
            f_gradient,
            dimension,
            x=None, 
            E=1e-8,
            B=0.001,
            initial_accumulator_value=0,
            use_ema=False,
            ema_momentum=0.99,
            max_fes=def_max_fes,
            objective_limit=None,
            min_clamp=def_clamps[0],
            max_clamp=def_clamps[1],
            checkpoints=def_checkpoints,
            smallest_val=def_smallest_val
            ):
        
        self.f_objective = f_objective
        self.f_gradient = f_gradient
        self.dimension = dimension
        self.smallest_val = smallest_val

        self.min_clamp = min_clamp
        self.max_clamp = max_clamp
        self.max_fes = max_fes

        self.initial_accumulator_value = initial_accumulator_value
        self.ema_momentum = ema_momentum
        self.use_ema = use_ema
        
        self.G = np.full(self.dimension, self.initial_accumulator_value, dtype=np.float64)
        self.B = B
        self.E = E

        self.objective_counter = 0
        self.error = None
        self.checkpoints = checkpoints
        self.log = {checkpoint: [] for checkpoint in self.checkpoints}
        self.seen_checkpoints = set()

        if(objective_limit is None):
            self.objective_limit = self.dimension * self.max_fes
        else:
            self.objective_limit = objective_limit

        if(x is None):
            self.x = np.random.uniform(self.min_clamp, self.max_clamp, size=self.dimension)
        else:
            self.x = x 

    def start(self):
        while (self.objective_counter < self.objective_limit) and (self.error is None or self.error > self.smallest_val):
            self.step()
            self.collect_data()
        return self.x, self.error


    def step(self):
        grad, evals_used = self.f_gradient(self.x, E=self.E)
        self.objective_counter += evals_used
        
        if self.use_ema:
            self.G = self.ema_momentum * self.G + (1 - self.ema_momentum) * grad**2
        else:
            self.G += grad**2

        self.x -= (self.B / np.sqrt(self.G + self.E)) * grad

        self.x = np.clip(self.x, self.min_clamp, self.max_clamp)

        self.error, evals_used = self.f_objective(self.x)
        self.objective_counter += evals_used



    
    def collect_data(self):
        for checkpoint in self.checkpoints:
            checkpoint_fes = int(checkpoint * self.objective_limit)
            
            if self.error < self.smallest_val and self.objective_counter <= checkpoint_fes:
                self.log[checkpoint].append(0)

            if checkpoint not in self.seen_checkpoints and self.objective_counter >= checkpoint_fes:
                self.log[checkpoint].append(0 if self.error < self.smallest_val else self.error)
                self.seen_checkpoints.add(checkpoint)