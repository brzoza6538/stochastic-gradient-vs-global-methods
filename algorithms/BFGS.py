# # https://github.com/trsav/bfgs
# # https://github.com/funnydman/BFGS-NelderMead-TrustRegion/blob/master/bfgs_algorithm.py
# # https://habr.com/ru/articles/333356/

# #albo otoczka do scipy

# # !/usr/bin/python
# # -*- coding: utf-8 -*-
# import numpy as np
# import numpy.linalg as ln
# import scipy as sp
# import scipy.optimize


# # Objective function
# def f(x):
#     return x[0] ** 2 - x[0] * x[1] + x[1] ** 2 + 9 * x[0] - 6 * x[1] + 20


# # Derivative
# def f1(x):
#     return np.array([2 * x[0] - x[1] + 9, -x[0] + 2 * x[1] - 6])


# def bfgs_method(f, fprime, x0, maxiter=None, epsi=10e-3):
#     """
#     Minimize a function func using the BFGS algorithm.
    
#     Parameters
#     ----------
#     func : f(x)
#         Function to minimise.
#     x0 : ndarray
#         Initial guess.
#     fprime : fprime(x)
#         The gradient of `func`.
#     """

#     if maxiter is None:
#         maxiter = len(x0) * 200

#     # initial values
#     count = 0
#     gfk = fprime(x0)
#     N = len(x0)
#     # Set the Identity matrix I.
#     I = np.eye(N, dtype=int)
#     Hk = I
#     xk = x0

#     while ln.norm(gfk) > epsi and count < maxiter:
#         # pk - direction of search

#         pk = -np.dot(Hk, gfk)

#         # Line search constants for the Wolfe conditions.
#         # Repeating the line search

#         # line_search returns not only alpha
#         # but only this value is interesting for us

#         line_search = sp.optimize.line_search(f, f1, xk, pk)
#         alpha_k = line_search[0]

#         xkp1 = xk + alpha_k * pk
#         sk = xkp1 - xk
#         xk = xkp1

#         gfkp1 = fprime(xkp1)
#         yk = gfkp1 - gfk
#         gfk = gfkp1

#         count += 1

#         ro = 1.0 / (np.dot(yk, sk))
#         A1 = I - ro * sk[:, np.newaxis] * yk[np.newaxis, :]
#         A2 = I - ro * yk[:, np.newaxis] * sk[np.newaxis, :]
#         Hk = np.dot(A1, np.dot(Hk, A2)) + (ro * sk[:, np.newaxis] *
#                                            sk[np.newaxis, :])

#     return xk, count


# result, k = bfgs_method(f, f1, np.array([1, 1]))

# print('Result of BFGS method:')
# print('Final Result (best point): {}'.format(result))
# print('Iteration Count: {}'.format(k))

from .globals import *
import numpy as np 
from scipy.optimize import minimize

class BFGS():
    def __init__(
            self,
            f_objective,
            f_gradient,
            dimension,
            x=None,
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
        self.max_fes = max_fes
        self.min_clamp = min_clamp
        self.max_clamp = max_clamp
        self.checkpoints = checkpoints
        self.smallest_val = smallest_val

        self.log = {checkpoint: [] for checkpoint in self.checkpoints}
        self.seen_checkpoints = set()

        self.objective_counter = 0
        self.error = None

        if objective_limit is None:
            self.objective_limit = self.dimension * self.max_fes
        else:
            self.objective_limit = objective_limit

        if x is None:
            self.x = np.random.uniform(self.min_clamp, self.max_clamp, size=self.dimension)
        else:
            self.x = x

    def wrapped_f_objective(self, x):
        if self.objective_counter >= self.objective_limit:
            raise StopIteration("Objective limit reached.")

        error, evals = self.f_objective(x)
        self.objective_counter += evals
        self.error = error
        self.collect_data()

        print(self.objective_counter, " / ", self.objective_limit, " -:- ", self.error)
        return error

    def wrapped_grad(self, x):
        grad, evals = self.f_gradient(x)
        self.objective_counter += evals
        return grad


    def start(self):
        try:
            result = minimize(
                self.wrapped_f_objective,
                self.x,
                method='BFGS',
                jac=self.wrapped_grad,
                options={
                    'maxiter': self.objective_limit,
                    'disp': False
                },
                bounds=[(self.min_clamp, self.max_clamp)] * self.dimension
            )
            self.x = result.x
            self.error = result.fun

        except StopIteration:
            # Optimization interrupted by reaching objective_limit
            pass

        # Zapewnij wpisy dla każdego checkpointa
        for checkpoint in self.checkpoints:
            if len(self.log[checkpoint]) == 0:
                self.log[checkpoint].append(0 if self.error < self.smallest_val else self.error)


    def collect_data(self):
        for checkpoint in self.checkpoints:
            checkpoint_fes = int(checkpoint * self.objective_limit)

            if checkpoint not in self.seen_checkpoints and self.objective_counter >= checkpoint_fes:
                self.log[checkpoint].append(0 if self.error < self.smallest_val else self.error)
                self.seen_checkpoints.add(checkpoint)


    def return_epoch_log(self):
        return self.objective_counter, self.error
