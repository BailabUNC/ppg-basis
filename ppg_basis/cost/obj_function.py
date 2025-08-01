from .cost_metrics import terms

def objective_function(model, signal, cost_metrics: list, func = None):
    """
    Returns objective function by combining cost metrics
    :param model: reference signal
    :param signal: reconstructed signal
    :param cost_metrics: list of metrics to ccombine
    :param func: optional cost function
    :return: objective function
    """
    obj_func = func(model, signal) if func is not None else 0
    for metric in cost_metrics:
        obj_func += terms[metric](model, signal)
    return obj_func