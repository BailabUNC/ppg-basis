import numpy as np
from scipy.signal import detrend
from ppg_basis.utils.math_utils import gamma_pdf, norm_pdf, norm_cdf, interp1d_lut
from numba import njit
from ppg_constants import default_M

def unified_model_ode(ppinterval, fs, seconds, basis_type, thetai, basis_params, ode_solver):
    """
    Extract z(t) given input parameters and target basis function
    :param ppinterval: peak-to-peak array
    :param fs: sampling rate
    :param seconds: total time window
    :param basis_type: target basis
    :param thetai: phase location in PPG period
    :param basis_params: parameters for basis function
    :param ode_solver: select third or fourth order RK ODE solver
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

    M = default_M
    x_table = np.linspace(0, 2 * np.pi, M) # FIXME: Here, we had M = 500 originally, is it okay to replace this with default_M = 1024?
    mean_vals, lut_vals = precompute_mean_basis_values(np.array(basis_params),
                                                       basis_type, M, x_table)

    if ode_solver == "rk3":
        traj = rk3_integration(y0, tspan, rr, fs, thetai, np.array(basis_params),
                               basis_type, mean_vals, x_table, lut_vals)
    else:
        traj = rk4_integration(y0, tspan, rr, fs, thetai, np.array(basis_params),
                           basis_type, mean_vals, x_table, lut_vals)
    z = traj[:, 2]
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    z = detrend(z)
    z -= np.mean(z)
    z = (z - np.min(z)) / (np.max(z) - np.min(z) + 1e-8)
    return z

@njit(cache=True)
def precompute_mean_basis_values(basis_params, basis_type, M, x_table):
    """
    Compute mean of each basis and look up table vals
    :param basis_params: basis functions (one set of parameters per row)
    :param basis_type: target basis
    :param M: number of sampling points = len(x_table)
    :param x_table: array of sampling points
    :return mean_vals: mean of each basis function
    :return lut_vals: look up table values (discretized basis function)
    """
    L = basis_params.shape[0]
    mean_vals = np.zeros(L)
    lut_vals = np.zeros((L,M))
    for i in range(L):
        if basis_type == 'gamma':
            alpha, scale = basis_params[i, 1], basis_params[i, 2]
            for j in range(M):
                lut_vals[i,j] = gamma_pdf(x_table[j], alpha, scale)
            mean_vals[i] = np.trapezoid(lut_vals[i], x_table)/(2*np.pi)
        elif basis_type == 'skewed-gaussian':
            b, skew = basis_params[i, 1], basis_params[i, 2]
            for j in range(M):
                x = x_table[j] - np.pi
                lut_vals[i,j] = 2 * x * norm_pdf(x, b) * norm_cdf(skew * x / b)
            mean_vals[i] = np.trapezoid(lut_vals[i], x_table)/(2*np.pi)
    return mean_vals, lut_vals

@njit(cache=True)
def rk3_integration(y0, tspan, rr, fs, thetai, basis_params, basis_type, mean_vals, x_table, lut_vals):
    """
    ODE solver using 3rd order RK method
    :param y0: initial value
    :param tspan: timepoints
    :param rr: peak-to-peak array
    :param fs: sampling rate
    :param thetai: phase location in PPG period
    :param basis_params: parameters for basis function
    :param basis_type: target basis
    :param mean_vals: mean of each basis
    :param x_table: x values for interp1d
    :param lut_vals: lookup table values
    :return: vector of coordinate values
    """
    n = len(tspan)
    dt = tspan[1] - tspan[0]
    y = np.zeros((len(tspan), 3))
    y[0] = y0
    for i in range(1, n):
        t = tspan[i - 1]
        k1 = generator_equations(t, y[i - 1], rr, fs, thetai, basis_params,
                                 basis_type, mean_vals, x_table, lut_vals)
        k2 = generator_equations(t + dt / 2, y[i - 1] + dt * k1 / 2, rr, fs,
                                 thetai, basis_params, basis_type, mean_vals, x_table, lut_vals)
        k3 = generator_equations(t + dt, y[i-1] - dt * k1 + 2.0 * dt * k2, rr, fs,
                                 thetai, basis_params, basis_type, mean_vals, x_table, lut_vals)
        y[i] = y[i - 1] + (dt / 6) * (k1 + 4.0 * k2 + k3)
    return y

@njit(cache=True)
def rk4_integration(y0, tspan, rr, fs, thetai, basis_params, basis_type, mean_vals, x_table, lut_vals):
    """
    ODE solver using 4th order RK method
    :param y0: initial value
    :param tspan: timepoints
    :param rr: peak-to-peak array
    :param fs: sampling rate
    :param thetai: phase location in PPG period
    :param basis_params: parameters for basis function
    :param basis_type: target basis
    :param mean_vals: mean of each basis
    :param x_table: x values for interp1d
    :param lut_vals: lookup table values
    :return: vector of coordinate values
    """
    n = len(tspan)
    dt = tspan[1] - tspan[0]
    y = np.zeros((len(tspan), 3))
    y[0] = y0
    for i in range(1, n):
        t = tspan[i - 1]
        k1 = generator_equations(t, y[i - 1], rr, fs, thetai, basis_params,
                                 basis_type, mean_vals, x_table, lut_vals)
        k2 = generator_equations(t + dt / 2, y[i - 1] + dt * k1 / 2, rr, fs,
                                 thetai, basis_params, basis_type, mean_vals, x_table, lut_vals)
        k3 = generator_equations(t + dt / 2, y[i - 1] + dt * k2 / 2, rr, fs,
                                 thetai, basis_params, basis_type, mean_vals, x_table, lut_vals)
        k4 = generator_equations(t + dt, y[i - 1] + dt * k3, rr, fs,
                                 thetai, basis_params, basis_type, mean_vals, x_table, lut_vals)
        y[i] = y[i - 1] + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y

@njit(cache=True)
def generator_equations(t, point, rr, fs, thetai, basis_params, basis_type, mean_vals, x_table, lut_vals):
    """
    Generate value for given point in PPG using specified basis function
    :param t: time
    :param point: coordinate
    :param rr: peak-to-peak array
    :param fs: sampling rate
    :param thetai: vector of phase locations for each basis in PPG
    :param basis_params: parameters for basis function
    :param basis_type: target basis
    :param mean_vals: mean of each basis
    :param x_table: x values for interp1d
    :param lut_vals: lookup table values
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
        if basis_type == 'gaussian':
            a, b = basis_params[i, 0], basis_params[i, 1]
            b_sq = max(b ** 2, 1e-6)
            f = a * diff_theta * np.exp(-(diff_theta ** 2) / (2 * b_sq))
        else:
            xval = diff_theta + np.pi
            f = interp1d_lut(xval, x_table, lut_vals[i]) - mean_vals[i]
            f *= basis_params[i, 0]
        dzdt -= f * w

    return np.array([dxdt, dydt, dzdt])
