# from __future__ import annotations
from .globals import *

from enum import Enum
from typing import Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Dict, Any

import numpy as np
from cma import CMAEvolutionStrategy
from opfunu.cec_based.cec import CecBenchmark
from scipy.optimize import bracket, golden
from typing import Tuple, Union, Optional

rng = np.random.default_rng(0)

from typing import Union

DEFAULT_CMA_OPTIONS: Dict[str, Any] = {
    "tolfun": 1e-9,
    "tolfunhist": 1e-9,
    "tolflatfitness": 3,
}

@dataclass
class OptFun:
    fun: Callable
    grad: Callable
    name: str
    optimum: int

    def optimum_for_dim(self, dim: int):
        return np.ones(dim) * self.optimum

@dataclass
class BaseResult:
    fun: Union[OptFun, CecBenchmark]
    dim: int
    k: Union[int, None]
    grad_variation: "CMAVariation"

def gradient_central(func: Callable, x: np.ndarray, h: float = 1e-6) -> np.ndarray:
    n = len(x)
    grad = np.zeros_like(x, dtype=float)

    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()

        x_plus[i] += h
        x_minus[i] -= h

        grad[i] = (func(x_plus) - func(x_minus)) / (2 * h)

    return grad


def gradient_forward(func: Callable, x: np.ndarray, h: float = 1e-3) -> np.ndarray:
    n = len(x)
    grad = np.zeros_like(x, dtype=float)
    f0 = func(x)

    for i in range(n):
        x_plus = x.copy()
        x_plus[i] += h
        grad[i] = (func(x_plus) - f0) / h

    return grad

def get_function(f: Union[CecBenchmark, OptFun]):
    """Common interface for CEC and OptFun functions."""
    if isinstance(f, CecBenchmark):
        return f.evaluate
    else:
        return f.fun



@dataclass
class StepSizeResult:
    x: np.ndarray
    golden_step_sizes: np.ndarray
    regular_step_sizes: np.ndarray


@dataclass
class CMAResult(BaseResult):
    midpoint_values: np.ndarray
    best_values: np.ndarray
    nums_evals: np.ndarray

    @staticmethod
    def highest_eval_count(results: "Iterable[list[CMAResult]]"):
        return max(r.nums_evals[-1] for result in results for r in result)


class CMAExperimentCallback(ABC):
    @abstractmethod
    def __call__(self, es: CMAEvolutionStrategy):
        pass

def one_dim(fun: Union[OptFun, CecBenchmark], x, d):
    """Gimmick to make a multdimensional function 1dim
    with a set direction d"""

    def wrapper(alpha):
        return get_function(fun)(x + alpha * d)

    return wrapper


class CMAVariation(Enum):
    VANILLA = "vanilla"
    PC = "pc"
    PC_C = "C * pc"
    ANALYTICAL_GRAD = "analytical gradient"
    ANALYTICAL_GRAD_C = "C * analytical gradient"
    CENTRAL_DIFFERENCE_C = "C * central difference"
    FORWARD_DIFFERENCE_C = "C * forward difference"


def lincmaes(
    x: np.ndarray,
    fun: Union[OptFun, CecBenchmark],
    switch_interval: int,
    popsize: int,
    maxevals: Union[int, None] = None,
    gradient_type: CMAVariation = CMAVariation.ANALYTICAL_GRAD_C,
    gradient_cost: int = 0,
    seed: int = 0,
    get_step_information: bool = False,
    callback: Union[CMAExperimentCallback, None] = None,
) -> Tuple[CMAResult, Optional[StepSizeResult]]:
    midpoint_values = []
    evals_values = []
    best_values = []

    if get_step_information:
        golden_step_x, golden_step_sizes, regular_step_sizes = [], [], []

    inopts = {}
    if popsize:
        inopts["popsize"] = popsize
    if maxevals:
        inopts["maxfevals"] = maxevals
    if seed:
        inopts["seed"] = seed

    x = np.clip(x, def_clamps[0], def_clamps[1])  # TODO - inaczej
    es = CMAEvolutionStrategy(x, 1, inopts=inopts)

    while not es.stop():

        for i in range(switch_interval):
            f = get_function(fun)
            # TODO: figure out why this even happens

            try:
                es.tell(*es.ask_and_eval(f))
                if callback is not None:
                    callback(es)

            except ValueError:
                with open("lincmaes_failed.csv", "a") as file:
                    fun_name = getattr(fun, "name", type(fun).__name__)
                    file.write(
                        f"{fun_name},{es.countevals},{gradient_type},{switch_interval // len(x)}, {f(es.mean)}\n"
                    )
                    continue

            evals_values.append(es.countevals)
            midpoint_values.append(f(es.mean))
            best_values.append(f(es.best.x))
#---------------------------------- match -> if-elif 
        if gradient_type == CMAVariation.PC:
            d = es.pc

        elif gradient_type == CMAVariation.PC_C:
            d = es.C @ es.pc  # pyright: ignore[reportOperatorIssue]

        elif gradient_type == CMAVariation.ANALYTICAL_GRAD_C:
            if isinstance(fun, CecBenchmark):
                raise ValueError("CecBenchmark does not support analytical gradient")
            es.countevals += gradient_cost
            d = es.C @ fun.grad(es.mean)

        elif gradient_type == CMAVariation.ANALYTICAL_GRAD:
            if isinstance(fun, CecBenchmark):
                raise ValueError("CecBenchmark does not support analytical gradient")
            es.countevals += gradient_cost
            d = fun.grad(es.mean)

        elif gradient_type == CMAVariation.CENTRAL_DIFFERENCE_C:
            es.countevals += 2 * len(es.mean)
            d = es.C @ gradient_central(get_function(fun), es.mean)

        elif gradient_type == CMAVariation.FORWARD_DIFFERENCE_C:
            es.countevals += len(es.mean)
            d = es.C @ gradient_forward(get_function(fun), es.mean, 1e-3)  # pyright: ignore[reportOperatorIssue]



        else:
            raise ValueError("Vanilla should not be passed to lincmaes")
#----------------------------------

        try:
            fn = one_dim(fun, es.mean, d)
            xa, xb, xc, fa, fb, fc, funccalls = bracket(fn, maxiter=2000)
            es.countevals += funccalls
            solution, fval, funccalls = golden(fn, brack=(xa, xb, xc), full_output=True)
            es.countevals += funccalls

            if get_step_information:
                golden_step_x.append(funccalls)
                golden_step_sizes.append(np.linalg.norm(solution - es.mean))
                regular_step_sizes.append(np.linalg.norm(es.sigma * es.delta))

            # Shift the mean
            solution = es.mean + solution * d
            es.mean = solution
            es.pc = np.zeros_like(solution)

        except RuntimeError:
            with open("golden_failed.csv", "a") as f:
                fun_name = getattr(fun, "name", type(fun).__name__)
                f.write(
                    f"bracket,{fun_name},{es.countevals},{gradient_type},{switch_interval // len(x)}\n"
                )
            continue

        except ValueError:
            with open("golden_failed.csv", "a") as f:
                fun_name = getattr(fun, "name", type(fun).__name__)
                f.write(
                    f"golden,{fun_name},{es.countevals},{gradient_type},{switch_interval // len(x)}\n"
                )
            continue

    result = CMAResult(
        fun=fun,
        dim=len(x),
        k=int(switch_interval / len(x)),
        grad_variation=gradient_type,
        midpoint_values=np.array(midpoint_values),
        best_values=np.array(best_values),
        nums_evals=np.array(evals_values),
    )

    ss_result = None
    if get_step_information:
        ss_result = StepSizeResult(
            x=np.array(golden_step_x),
            golden_step_sizes=np.array(golden_step_sizes),
            regular_step_sizes=np.array(regular_step_sizes),
        )

    return result, ss_result



#----------------------------------


def eswrapper(
    x: np.ndarray,
    fun: Union[OptFun, CecBenchmark],
    popsize: int,
    maxevals: int,
    variation: CMAVariation = CMAVariation.VANILLA,
    line_search_interval: Union[int, None] = None,
    gradient_cost: int = 0,
    seed: int = 0,
    callback: Union[CMAExperimentCallback, None] = None,
) -> CMAResult:
    """Wraps all variations of the CMA-ES into a single function with a common interface."""

    if variation != CMAVariation.VANILLA:
        assert line_search_interval is not None, "Line search interval must be set."
        return lincmaes(
            x,
            fun,
            line_search_interval,
            popsize,
            maxevals,
            gradient_type=variation,
            gradient_cost=gradient_cost,
            seed=seed,
        )[0]

    midpoint_values = []
    evals_values = []
    best_values = []

    inopts = DEFAULT_CMA_OPTIONS.copy()
    if popsize:
        inopts["popsize"] = popsize
    if maxevals:
        inopts["maxfevals"] = maxevals
    if seed:
        inopts["seed"] = seed

    x = np.clip(x, def_clamps[0], def_clamps[1])  # TODO - inaczej
    es = CMAEvolutionStrategy(x, 1, inopts=inopts)

    while not es.stop():
        f = get_function(fun)
        try:
            X = es.ask()
            fit_vals = [f(x) for x in X]

            if any(np.isnan(fx) or np.isinf(fx) for fx in fit_vals):
                raise ValueError("Function returned NaN or inf")

            es.tell(X, fit_vals)

        except ValueError as e:
            with open("error.csv", "a") as file:
                file.write(f"{fun.name},{es.countevals},{es.mean},{variation},{str(e)}\n")

            if callback is not None:
                callback(es)

        evals_values.append(es.countevals)
        midpoint_values.append(f(es.mean))
        best_values.append(f(es.best.x))

    return CMAResult(
        fun=fun,
        dim=len(x),
        k=None,
        grad_variation=variation,
        midpoint_values=np.array(midpoint_values),
        best_values=np.array(best_values),
        nums_evals=np.array(evals_values),
    )
