import numpy as np
from numba import njit
from scipy.stats import gamma, norm

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
        return a * gamma.pdf(x, a=alpha, scale=scale)
    elif basis_type == 'skewed-gaussian':
        a, b, skew = params
        x = theta_diff
        norm_part = norm.pdf(x / b)
        cdf_part = norm.cdf(skew * x / b)
        return 2 * a * x * norm_part * cdf_part
    else:
        raise ValueError(f"Unsupported basis type: {basis_type}")

@njit
def generator_equations(t, point, rr, fs, thetai, basis_params, basis_type_code):
    """
    Generate value for given point in PPG using specified basis function
    :param t: time
    :param point: coordinate
    :param rr: peak-to-peak array
    :param fs: sampling rate
    :param thetai: phase location in PPG period
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

        if basis_type_code == 0:  # Gaussian
            a, b = basis_params[i, 0], basis_params[i, 1]
            b_sq = max(b ** 2, 1e-6)
            dzdt -= a * diff_theta * np.exp(-(diff_theta ** 2) / (2 * b_sq)) * w

        elif basis_type_code == 1:  # Gamma
            a, alpha, scale = basis_params[i, 0], basis_params[i, 1], basis_params[i, 2]
            xval = diff_theta + np.pi
            dzdt -= a * gamma.pdf(xval, a=alpha, scale=scale) * w

        elif basis_type_code == 2:  # Skewed Gaussian
            a, b, skew = basis_params[i, 0], basis_params[i, 1], basis_params[i, 2]
            norm_val = np.exp(-0.5 * (diff_theta / b)**2) / (np.sqrt(2*np.pi) * b)
            cdf_val = 0.5 * (1 + np.erf(skew * diff_theta / (b * np.sqrt(2))))
            dzdt -= 2 * a * diff_theta * norm_val * cdf_val * w

    return np.array([dxdt, dydt, dzdt])

@njit
def rk4_integration(y0, tspan, rr, fs, thetai, basis_params, basis_type_code):
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
    dt = tspan[1] - tspan[0]
    y = np.zeros((len(tspan), 3))
    y[0] = y0
    for i in range(1, len(tspan)):
        t = tspan[i - 1]
        k1 = generator_equations(t, y[i - 1], rr, fs, thetai, basis_params, basis_type_code)
        k2 = generator_equations(t + dt / 2, y[i - 1] + dt * k1 / 2, rr, fs, thetai, basis_params, basis_type_code)
        k3 = generator_equations(t + dt / 2, y[i - 1] + dt * k2 / 2, rr, fs, thetai, basis_params, basis_type_code)
        k4 = generator_equations(t + dt, y[i - 1] + dt * k3, rr, fs, thetai, basis_params, basis_type_code)
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

    traj = rk4_integration(y0, tspan, rr, fs, thetai, np.array(basis_params), basis_type_code)
    z = traj[:, 2]
    z -= np.mean(z)
    z = (z - np.min(z)) / (np.max(z) - np.min(z) + 1e-8)
    return z
