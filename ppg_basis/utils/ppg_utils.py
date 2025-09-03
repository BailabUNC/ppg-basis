import warnings
import numpy as np
from scipy.optimize import LinearConstraint
from ppg_constants import param_bnds_dict, param_validators, default_params

def validate_param(name, value):
    validator = param_validators.get(name, lambda x: True)
    if validator(value):
        return value
    warnings.warn(f'{name} parameter is invalid ({value}), defaulting to {default_params[name]}', UserWarning)
    return default_params[name]

def pp_interval_generator(time: float, mu: float = 0, sigma: float = 1):
    """
    Generate pulse-to-pulse interval array
    :param time: total time of PPG window
    :param mu: mean pulse-to-pulse interval variation
    :param sigma: standard deviation of pulse-to-pulse interval variation
    :return: pulse-to-pulse interval array
    """
    if sigma == 0:
        if mu <= 0:
            raise ValueError("pp_interval_generator: mu must be >0 when sigma=0")
        n = int(np.ceil(time/mu))
        return [round(mu,3)] * n
    pp_interval = []
    total = 0.0
    while total < time:
        draw = np.random.normal(mu, sigma)
        if draw <=0:
            continue
        val = round(draw, 3)
        pp_interval.append(val)
        total += val
    return pp_interval

def generate_basis_parameters(L, basis_type, random_state=None):
    """
    Randomly generate parameters given a basis function
    :param L: number of basis functions
    :param basis_type: basis function
    :param random_state: initialization protocol for random state
    :return: basis parameter list
    """
    rng = np.random.default_rng(random_state)
    thetai = np.sort(rng.uniform(-np.pi, np.pi, L))
    params = []

    for _ in range(L):
        if basis_type == 'gaussian':
            a = rng.uniform(0.1, 1.0)
            b = rng.uniform(0.1, 3.0)
            params.append([a, b])
        elif basis_type == 'gamma':
            a = rng.uniform(0.1, 1.0)
            alpha = rng.uniform(1.0, 5.0)
            scale = rng.uniform(0.1, 1.0)
            params.append([a, alpha, scale])
        elif basis_type == 'skewed-gaussian':
            a = rng.uniform(0.1, 1.0)
            b = rng.uniform(0.1, 1.0)
            skew = rng.uniform(-5, 5)
            params.append([a, b, skew])
        else:
            raise ValueError(f"Unsupported basis type: {basis_type}")

    return thetai, np.array(params)

def get_bounds_and_constraints(L, basis_type):
    """
    Generate bounds and constraints for optimization
    :param L: number of basis functions
    :param basis_type: basis function
    :return: bounds and constraints
    """
    theta_bounds = [(-np.pi, np.pi)] * L
    
    param_bnds = param_bnds_dict.get(basis_type)
    if param_bnds is None:
        raise ValueError(f"Unsupported basis type: {basis_type}")

    # replicate for L bases
    param_bounds = param_bnds * L
    bounds = theta_bounds + param_bounds

    if L > 1:
        n = len(bounds)
        A = np.zeros((L - 1, n))
        for i in range(L - 1):
            A[i, i] = 1  # θ_i
            A[i, i + 1] = -1  # θ_{i+1}
        lb = -np.inf * np.ones(L - 1)
        ub = 0 * np.ones(L - 1)
        constraint = LinearConstraint(A, lb, ub)
    else:
        constraint = None

    return bounds, constraint