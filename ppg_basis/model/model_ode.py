import numpy as np
from scipy.signal import detrend
from ppg_basis.utils.math_utils import gamma_pdf_precomp, skewed_gaussian_val
from numba import njit
import math

def unified_model_ode(ppinterval, fs, seconds, basis_type, thetai, basis_params, ode_solver, M):
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

    x_table = np.linspace(0, 2 * np.pi, M)
    mean_vals, lut_vals = precompute_mean_basis_values(np.array(basis_params),
                                                       basis_type, x_table, M)

    if ode_solver == "rk3":
        traj = rk3_integration(y0, tspan, rr, fs, thetai, np.array(basis_params),
                               basis_type, mean_vals, x_table, lut_vals)
    elif ode_solver == "rk4":
        traj = rk4_integration(y0, tspan, rr, fs, thetai, np.array(basis_params),
                           basis_type, mean_vals, x_table, lut_vals)
    else:
        raise ValueError("Unsupported RK method")

    z = traj[:, 2]
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    z = detrend(z)
    z -= np.mean(z)
    z = (z - np.min(z)) / (np.max(z) - np.min(z) + 1e-8)
    return z

@njit(cache=True)
def precompute_mean_basis_values(basis_params, basis_type, x_table, M):
    """
    Compute mean of each basis and look up table vals.
    """
    L = basis_params.shape[0]
    mean_vals = np.zeros(L)
    lut_vals = np.zeros((L,M))
    dx = x_table[1] - x_table[0]

    for i in range(L):
        if basis_type == 'gamma':
            alpha, scale = basis_params[i, 1], basis_params[i, 2]
            alpha_m1 = alpha - 1.0
            inv_scale = 1.0 / max(scale, 1e-10)
            log_norm = math.lgamma(alpha) + alpha * math.log(max(scale, 1e-10))
            for j in range(M):
                lut_vals[i,j] = gamma_pdf_precomp(x_table[j], alpha_m1, inv_scale, log_norm)
            # uniform trapezoid
            s = 0.5 * (lut_vals[i, 0] + lut_vals[i, M-1])
            for j in range(1, M-1):
                s += lut_vals[i, j]
            mean_vals[i] = s * dx / (2.0 * np.pi)

        elif basis_type == 'skewed-gaussian':
            b, skew = basis_params[i, 1], basis_params[i, 2]
            bb = max(b, 1e-6)
            inv_b = 1.0 / bb
            skew_over_b = skew * inv_b
            for j in range(M):
                x = x_table[j] - np.pi
                lut_vals[i,j] = skewed_gaussian_val(x, inv_b, skew_over_b)
            s = 0.5 * (lut_vals[i, 0] + lut_vals[i, M-1])
            for j in range(1, M-1):
                s += lut_vals[i, j]
            mean_vals[i] = s * dx / (2.0 * np.pi)
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
            M = x_table.shape[0]
            dx_table = x_table[1] - x_table[0]
            if xval < 0.0:
                xval = 0.0
            elif xval > x_table[M - 1]:
                xval = x_table[M - 1]
            idx = int(xval / dx_table)
            if idx >= M - 1:
                idx = M - 2
            frac = (xval - idx * dx_table) / dx_table
            f_interp = (1.0 - frac) * lut_vals[i, idx] + frac * lut_vals[i, idx + 1]
            f = (f_interp - mean_vals[i]) * basis_params[i, 0]
        dzdt -= f * w

    return np.array([dxdt, dydt, dzdt])
