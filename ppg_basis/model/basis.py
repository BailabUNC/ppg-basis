import numpy as np
from numba import njit
from ppg_basis.utils.math_utils import *
from scipy.optimize import LinearConstraint
from scipy.signal import detrend

def basis_function(theta_diff: float, basis_type: str, params):
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


def precompute_mean_basis_values(basis_params, basis_type_code, M):
    L = basis_params.shape[0]
    mean_vals = np.zeros(L)
    for i in range(L):
        if basis_type_code == 1:
            mean_vals[i] = gamma_mean(basis_params[i, 1], basis_params[i, 2], M)
        elif basis_type_code == 2:
            mean_vals[i] = skewed_gaussian_mean(basis_params[i, 1], basis_params[i, 2], M)
    return mean_vals


def generator_equations(t, point, rr, fs, thetai, basis_params, basis_type_code, mean_vals):
    """
    Generate value for given point in PPG using specified basis function
    :param t: time
    :param point: coordinate
    :param rr: peak-to-peak array
    :param fs: sampling rate
    :param thetai: vector of phase locations for each basis in PPG
    :param basis_params: parameters for basis function
    :param basis_type_code: target basis
    :return: coordinate values calculated using basis & params
    """
    x, y, z = point
    ip = min(int(np.floor(t * fs)), len(rr) - 1)
    w = 2 * np.pi / rr[ip]
    dxdt = -w * y
    dydt = w * x
    dzdt = 0.0
    theta = np.arctan2(y, x)

    for i in range(len(thetai)):
        diff_theta = ((theta - thetai[i] + np.pi) % (2 * np.pi)) - np.pi
        if basis_type_code == 0:
            a, b = basis_params[i, 0], basis_params[i, 1]
            b_sq = max(b ** 2, 1e-6)
            f = a * diff_theta * np.exp(-(diff_theta ** 2) / (2 * b_sq))
            dzdt -= f * w
        elif basis_type_code == 1:
            a, alpha, scale = basis_params[i, 0], basis_params[i, 1], basis_params[i, 2]
            xval = diff_theta + np.pi
            f = a * (gamma_pdf(xval, alpha, scale) - mean_vals[i])
            dzdt -= f * w
        elif basis_type_code == 2:
            a, b, skew = basis_params[i, 0], basis_params[i, 1], basis_params[i, 2]
            norm_val = norm_pdf(diff_theta, b)
            cdf_val = norm_cdf(skew * diff_theta / b)
            f = 2 * a * diff_theta * norm_val * cdf_val
            f-= mean_vals[i]
            dzdt -= f * w

    return np.array([dxdt, dydt, dzdt])


def rk4_integration(y0, tspan, rr, fs, thetai, basis_params, basis_type_code, mean_vals):
    """
    ODE solver using RK method
    :param y0: initial value
    :param tspan: timepoints
    :param rr: peak-to-peak array
    :param fs: sampling rate
    :param thetai: phase location in PPG period
    :param basis_params: parameters for basis function
    :param basis_type_code: target basis
    :return: vector of coordinate values
    """
    n = len(tspan)
    dt = tspan[1] - tspan[0]
    y = np.zeros((len(tspan), 3))
    y[0] = y0
    for i in range(1, n):
        t = tspan[i - 1]
        k1 = generator_equations(t, y[i - 1], rr, fs, thetai, basis_params, basis_type_code, mean_vals)
        k2 = generator_equations(t + dt / 2, y[i - 1] + dt * k1 / 2, rr, fs, thetai, basis_params, basis_type_code, mean_vals)
        k3 = generator_equations(t + dt / 2, y[i - 1] + dt * k2 / 2, rr, fs, thetai, basis_params, basis_type_code, mean_vals)
        k4 = generator_equations(t + dt, y[i - 1] + dt * k3, rr, fs, thetai, basis_params, basis_type_code, mean_vals)
        y[i] = y[i - 1] + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y

# Main model interface
def unified_model_ode(ppinterval, fs, seconds, basis_type, thetai, basis_params):
    """
    Extract z(t) given input parameters and target basis function
    :param ppinterval: peak-to-peak array
    :param fs: sampling rate
    :param seconds: total time window
    :param basis_type: target basis
    :param thetai: phase location in PPG period
    :param basis_params: parameters for basis function
    :return: z(t)
    """
    dt = 1 / fs
    n_samples = int(np.ceil(seconds * fs))
    tspan = np.arange(n_samples) * dt

    rr = np.empty(n_samples)
    total_samples = 0
    i = 0
    while total_samples < n_samples:
        intv = ppinterval[i % len(ppinterval)]
        num = int(np.ceil(fs * intv))
        end = min(total_samples + num, n_samples)
        rr[total_samples:end] = intv
        total_samples = end
        i += 1

    y0 = np.array([-1.0, 0.0, 0.0])

    # Basis type to code
    basis_type_map = {'gaussian': 0, 'gamma': 1, 'skewed-gaussian': 2}
    basis_type_code = basis_type_map[basis_type]

    mean_vals = precompute_mean_basis_values(np.array(basis_params), basis_type_code, M=200)
    traj = rk4_integration(y0, tspan, rr, fs, thetai, np.array(basis_params), basis_type_code, mean_vals)
    z = traj[:, 2]
    z = detrend(z)
    z -= np.mean(z)
    z = (z - np.min(z)) / (np.max(z) - np.min(z) + 1e-8)
    return z

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
        # 1) Theta bounds
        theta_bounds = [(-np.pi, np.pi)] * L

        # 2) Parameter bounds by basis type
        if basis_type == 'gaussian':
            # each basis has (a, b)
            param_bnds = [(0.0, 1.0), (0.05, 3.0)]
        elif basis_type == 'gamma':
            # each basis has (a, alpha, scale)
            param_bnds = [(0.0, 1.0), (1.0, 6.0), (0.05, 3.0)]
        elif basis_type == 'skewed-gaussian':
            # each basis has (a, b, skew)
            param_bnds = [(0.0, 1.0), (0.05, 3.0), (-10.0, 10.0)]
        else:
            raise ValueError(f"Unsupported basis type: {basis_type}")

        # replicate for L bases
        param_bounds = param_bnds * L

        # combine into one list: [θ-bounds…, param-bounds…]
        bounds = theta_bounds + param_bounds

        # 3) Build θ-ordering constraint
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