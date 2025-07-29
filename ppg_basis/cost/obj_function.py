from cost_metrics import terms
from numba import njit

@njit
def objective_function(model, signal, cost_metrics: list, func = None):
    cost_metrics = terms.copy()
    obj_func = func(model, signal) if func is not None else 0
    for metric in cost_metrics:
        obj_func += cost_metrics[metric(model, signal)]
    return obj_func