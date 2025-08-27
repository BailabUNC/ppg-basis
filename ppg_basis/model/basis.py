import numpy as np
from numba import njit
from ppg_basis.utils.math_utils import *
from scipy.optimize import LinearConstraint
from scipy.signal import detrend
from ppg_basis.utils.math_utils import _interp1d_lut_scalar, _wrap_pi, _interp_uniform_table

@njit
def basis_function(theta_diff: float, basis_type: str, params): # FIXME: Is this func ever called? Can we go ahead and delete it?
    """
    define basis function for PPG
    :param theta_diff: phase location in PPG period
    :param basis_type: dictates basis function - gaussian, gamma, or skewed gaussian
    :param params: parameter list relevant for each basis
    :return: basis function
    """
    if basis_type == 'gaussian':
        a, b = params
        return a * theta_diff * np.exp(-(theta_diff ** 2) / (2 * b ** 2))
    elif basis_type == 'gamma':
        a, alpha, scale = params
        x = theta_diff + np.pi  # shift domain to positive for gamma
        return a * gamma_pdf(x=x, alpha=alpha, scale=scale)
    elif basis_type == 'skewed-gaussian':
        a, b, skew = params
        x = theta_diff
        norm_part = norm_pdf(x, b)
        cdf_part = norm_cdf(skew * x / b)
        return 2 * a * x * norm_part * cdf_part
    else:
        raise ValueError(f"Unsupported basis type: {basis_type}")

def generate_basis_parameters(L, basis_type, random_state=None): # FIXME: This func only gets called in ppg_generator and ppg_extractor; should it be moved to ppg_utils?
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

def get_bounds_and_constraints(L, basis_type): # FIXME: This func only gets called in ppg_extractor; should it be moved to ppg_utils?
    """
    Generate bounds and constraints for optimization
    :param L: number of basis functions
    :param basis_type: basis function
    :return: bounds and constraints
    """
    theta_bounds = [(-np.pi, np.pi)] * L

    if basis_type == 'gaussian':
        param_bnds = [(0.0, 1.0), (0.05, 3.0)]
    elif basis_type == 'gamma':
        param_bnds = [(0.0, 1.0), (1.0, 6.0), (0.05, 3.0)]
    elif basis_type == 'skewed-gaussian':
        param_bnds = [(0.0, 1.0), (0.05, 3.0), (-10.0, 10.0)]
    else:
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