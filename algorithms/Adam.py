from .globals import *
import numpy as np 

# parametry lr=0.001, E=1e-8, B1=0.9, B2=0.999, zostały specjalnie dobrane do zadań sieci neuronowej
# mają działać dla sieci gdzie wartości są w przedziałach typu [-1, 1], lub [0, 1]
# nie ma heurystyki dobierania parametrów 
# można co najwyżej metodą prób i błędów dopasowywać parametry bazując na tych standardowych

# B1 kieruje udziałem kierunku gradientu z przeszłości, B2 długości kroku z przeszłości
# jeśli wykres za mało się zmienia warto zmniejszyć B1
# jeśli wykres przeskakuje "dołki" warto zmniejszyć B2 
# learning rate zmieniać jeśli progres jest za wolny, lub za mało dokładny

# po otrzymaniu punktu x przerzucam go i gradient do przestrzeni [-1, 1], następnie pod koniec wracam do przedziału [-100, 100] 
#https://www.kdnuggets.com/2022/12/tuning-adam-optimizer-parameters-pytorch.html

class Adam():
    def __init__(
            self,
            f_objective,
            f_gradient,
            dimension,
            x=None, 
            lr=0.001, 
            E=1e-8,
            B1=0.9, 
            B2=0.999, 
            max_fes=def_max_fes,
            objective_limit=None,
            min_clamp=def_clamps[0],
            max_clamp=def_clamps[1],
            checkpoints=def_checkpoints,
            smallest_val=def_smallest_val
            ):

        # objective_limit ma priorytet, jak nie ma to max_fes * dimension 
        # smallest val
        # nie ma możliwości countdown-u po epokach?
        self.f_objective = f_objective
        self.lr = lr
        self.E = E
        self.B1 = B1
        self.B2 = B2
        self.dimension = dimension
        self.max_fes = max_fes
        self.f_gradient = f_gradient
        self.min_clamp = min_clamp
        self.max_clamp = max_clamp
        self.checkpoints = checkpoints
        self.smallest_val = smallest_val

        self.log = {checkpoint: [] for checkpoint in self.checkpoints}
        self.objective_counter = 0
        self.error = None
        self.seen_checkpoints = set()

        if(objective_limit is None):
            self.objective_limit = self.dimension * self.max_fes
        else:
            self.objective_limit = objective_limit

        if(x is None):
            self.x = np.random.uniform(self.min_clamp, self.max_clamp, size=self.dimension)
        else:
            self.x = x

        self.m = np.zeros_like(self.x)
        self.v = np.zeros_like(self.x)
        self.t = 0

    def start(self):
        while (self.objective_counter < self.objective_limit) and (self.error is None or self.error > self.smallest_val):
            self.step()
            self.collect_data()

    def step(self):
        grad, evals_used = self.f_gradient(self.x, E=self.E) 
        self.objective_counter += evals_used

        self.t += 1
        self.m = self.B1 * self.m + (1 - self.B1) * grad
        self.v = self.B2 * self.v + (1 - self.B2) * (grad * grad)
        m_hat = self.m / (1 - self.B1 ** self.t)
        v_hat = self.v / (1 - self.B2 ** self.t)

        self.x -= self.lr * m_hat / (np.sqrt(v_hat) + self.E)
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

    def return_epoch_log(self):
        return(self.objective_counter, self.error)




class NormAdam():
    def __init__(
            self,
            f_objective,
            f_gradient,
            dimension,
            x=None, 
            lr=0.001, 
            E=1e-8,
            B1=0.9, 
            B2=0.999, 
            max_fes=def_max_fes,
            objective_limit=None,
            min_clamp=def_clamps[0],
            max_clamp=def_clamps[1],
            checkpoints=def_checkpoints,
            smallest_val=def_smallest_val
            ):

        # objective_limit ma priorytet, jak nie ma to max_fes * dimension 
        # smallest val
        # nie ma możliwości countdown-u po epokach?
        self.f_objective = f_objective
        self.lr = lr
        self.E = E
        self.B1 = B1
        self.B2 = B2
        self.dimension = dimension
        self.max_fes = max_fes
        self.f_gradient = f_gradient
        self.min_clamp = min_clamp
        self.max_clamp = max_clamp
        self.checkpoints = checkpoints
        self.smallest_val = smallest_val

        self.log = {checkpoint: [] for checkpoint in self.checkpoints}
        self.objective_counter = 0
        self.error = None
        self.seen_checkpoints = set()

        if(objective_limit is None):
            self.objective_limit = self.dimension * self.max_fes
        else:
            self.objective_limit = objective_limit

        if(x is None):
            self.real_x = np.random.uniform(self.min_clamp, self.max_clamp, size=self.dimension) #N
        else:
            self.real_x = x #N
        self.x = self.normalize(self.real_x) #N
        self.m = np.zeros_like(self.x)
        self.v = np.zeros_like(self.x)
        self.t = 0

    def normalize(self, x): # N
        return 2 * (x - self.min_clamp) / (self.max_clamp - self.min_clamp) - 1

    def denormalize(self, x): # N
        return 0.5 * (x + 1) * (self.max_clamp - self.min_clamp) + self.min_clamp


    def start(self):
        while (self.objective_counter < self.objective_limit) and (self.error is None or self.error > self.smallest_val):
            self.step()
            self.collect_data()

    def step(self):
        grad, evals_used = self.f_gradient(self.real_x, E=self.E) #N
        self.objective_counter += evals_used
        scale = 2 / (self.max_clamp - self.min_clamp) #N
        grad = grad / scale #N TODO - pamiętaj że wcześniej było  grad = grad * scale

        self.t += 1
        self.m = self.B1 * self.m + (1 - self.B1) * grad
        self.v = self.B2 * self.v + (1 - self.B2) * (grad * grad)
        m_hat = self.m / (1 - self.B1 ** self.t)
        v_hat = self.v / (1 - self.B2 ** self.t)

        self.x -= self.lr * m_hat / (np.sqrt(v_hat) + self.E)

        self.x = np.clip(self.x, -1, 1) #N
        self.real_x = self.denormalize(self.x) #N
        self.error, evals_used = self.f_objective(self.real_x) #N
        self.objective_counter += evals_used

    def collect_data(self):
        for checkpoint in self.checkpoints:
            checkpoint_fes = int(checkpoint * self.objective_limit)
            
            if self.error < self.smallest_val and self.objective_counter <= checkpoint_fes:
                self.log[checkpoint].append(0)

            if checkpoint not in self.seen_checkpoints and self.objective_counter >= checkpoint_fes:
                self.log[checkpoint].append(0 if self.error < self.smallest_val else self.error)
                self.seen_checkpoints.add(checkpoint)

    def return_epoch_log(self):
        return(self.objective_counter, self.error)

