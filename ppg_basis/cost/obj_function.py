from .cost_metrics import terms, normalize_costs_only
import warnings

@normalize_costs_only(config={"appg": {"max_nrmse": 5.0}})
def objective_function(model, signal, cost_metrics: list, cost_weights: list=None, func = None, fs=125):
    """
    Returns objective function by combining cost metrics
    :param model: reference signal
    :param signal: reconstructed signal
    :param cost_metrics: list of metrics to combine
    :param cost_weights: list of weights per metric
    :param func: optional cost function
    :param fs: sampling rate (Hz)
    :return: objective function
    """
    obj_func = func(model, signal) if func is not None else 0
    if cost_weights is None or len(cost_weights) != len(cost_metrics):
        for metric in cost_metrics:
            obj_func += terms[metric](model, signal)
        return obj_func
    else:
        for i, metric in enumerate(cost_metrics):
            obj_func += (terms[metric](model, signal))*cost_weights[i]
        return obj_func