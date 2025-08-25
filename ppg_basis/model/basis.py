import numpy as np
from numba import njit
from ppg_basis.utils.math_utils import *
from scipy.optimize import LinearConstraint
from scipy.signal import detrend
from ppg_basis.utils.math_utils import _interp1d_lut_scalar, _wrap_pi, _interp_uniform_table

@njit
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


@njit(cache=True)
def _phase_from_rr(ppinterval, fs, n_samples):
    """
    Construct per-sample RR, ω=2π/RR, and phase θ[n] that wraps in [-π, π]
    """
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

    omega = 2.0 * np.pi / rr
    theta = np.empty(n_samples)
    theta[0] = -np.pi  # consistent with x0=-1,y0=0 → θ≈-π
    dt = 1.0 / fs
    for k in range(1, n_samples):
        theta[k] = _wrap_pi(theta[k-1] + omega[k-1]*dt)
    return rr, omega, theta

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

@njit(parallel=True, cache=True)
def _precompute_f_and_G(basis_params, basis_type, M):
    """
    Build LUTs for f(θ) with unit amplitude (a=1), and its primitive
    G(θ)=∫_0^θ (f(u)-mean)du on a uniform grid in [0, 2π].
    Subtracting mean enforces periodicity (no drift over a cycle).
    """
    L = basis_params.shape[0]
    x_table = np.linspace(0.0, 2.0*np.pi, M)
    f_lut = np.zeros((L, M))
    mean_vals = np.zeros(L)

    for i in range(L):
        if basis_type == 'gaussian':
            b = basis_params[i, 1]
            bb = max(b, 1e-6)
            inv2b2 = 1.0/(2.0*bb*bb)
            for j in prange(M):
                x = x_table[j] - np.pi
                f_lut[i, j] = x * np.exp(-(x*x) * inv2b2)
            mean_vals[i] = 0.0  # odd over symmetric interval
        elif basis_type == 'gamma':
            alpha, scale = basis_params[i, 1], basis_params[i, 2]
            for j in prange(M):
                f_lut[i, j] = gamma_pdf(x_table[j], alpha, scale)
            mean_vals[i] = np.trapezoid(f_lut[i], x_table) / (2.0*np.pi)
        elif basis_type == 'skewed-gaussian':
            b, skew = basis_params[i, 1], basis_params[i, 2]
            bb = max(b, 1e-6)
            for j in prange(M):
                x = x_table[j] - np.pi
                f_lut[i, j] = 2.0 * x * norm_pdf(x, bb) * norm_cdf(skew * x / bb)
            mean_vals[i] = np.trapezoid(f_lut[i], x_table) / (2.0*np.pi)
        else:
            raise ValueError("Unsupported basis type")

    # zero-mean so G is periodic
    for i in range(L):
        f_lut[i] -= mean_vals[i]

    # primitive via cumulative trapezoid
    G_lut = np.zeros_like(f_lut)
    for i in prange(L):
        acc = 0.0
        G_lut[i, 0] = 0.0
        for j in range(1, M):
            dx = x_table[j] - x_table[j-1]
            acc += 0.5*(f_lut[i, j-1] + f_lut[i, j]) * dx
            G_lut[i, j] = acc
        # remove tiny residual slope to enforce periodicity
        slope = G_lut[i, -1] / (2.0*np.pi)
        for j in prange(M):
            G_lut[i, j] -= slope * x_table[j]

    return x_table, G_lut

@njit
def _synthesize_basis_core(theta, thetai, basis_params, x_table, G_lut):
    n = theta.size
    L = thetai.size
    z = np.zeros(n)
    for i in range(L):
        a = basis_params[i, 0]
        for k in range(n):
            # wrap to [-π, π], then shift to [0, 2π] for LUT
            x = _wrap_pi(theta[k] - thetai[i]) + np.pi
            Gi = _interp1d_lut_scalar(x, x_table, G_lut[i])
            z[k] -= a * Gi
    return z

@njit
def _synthesize_gaussian_core(theta, thetai, basis_params):
    n = theta.size
    L = thetai.size
    z = np.zeros(n)
    for i in range(L):
        a = basis_params[i, 0]
        b = max(basis_params[i, 1], 1e-6)
        inv2b2 = 1.0 / (2.0 * b * b)
        amp = a * b * b
        for k in range(n):
            diff = ((theta[k] - thetai[i] + np.pi) % (2.0*np.pi)) - np.pi
            z[k] += amp * np.exp(- (diff * diff) * inv2b2)
    return z

def _build_phase_template_gaussian(thetai, basis_params, M):
    phi = np.linspace(0.0, 2.0*np.pi, M, endpoint=False)
    z_grid = np.zeros(M, dtype=np.float64)
    for i in range(thetai.size):
        a = basis_params[i, 0]
        b = max(basis_params[i, 1], 1e-6)
        amp = a * b * b
        diff = ((phi - (thetai[i] + np.pi)) % (2.0*np.pi)) - np.pi  # wrap to [-π,π]
        z_grid += amp * np.exp(-0.5 * (diff / b)**2)
    return z_grid

def _build_phase_template_generic(basis_type, thetai, basis_params, M):
    x_table = np.linspace(0.0, 2.0*np.pi, M, endpoint=False)
    _, G_lut = _precompute_f_and_G(np.ascontiguousarray(basis_params), basis_type, M)
    z_grid = np.zeros(M, dtype=np.float64)
    for i in range(thetai.size):
        a = basis_params[i, 0]
        # map θ_i to a circular shift
        shift = int(np.round((thetai[i] + np.pi) * M / (2.0*np.pi))) % M
        # roll primitive and scale
        z_grid -= a * np.roll(G_lut[i], shift)
    return z_grid

def build_phase_template(basis_type, thetai, basis_params, M=1024):
    if basis_type == 'gaussian':
        return _build_phase_template_gaussian(thetai, np.asarray(basis_params), M)
    else:
        return _build_phase_template_generic(basis_type, thetai, np.asarray(basis_params), M)

@njit(cache=True)
def sample_template(theta, z_grid):
    n = theta.size
    out = np.empty(n, dtype=np.float64)
    for k in range(n):
        out[k] = _interp_uniform_table(theta[k], z_grid)
    return out

def _tabulate_zero_mean_derivative(basis_type, basis_params, M):
    # returns g_grid on [0,2π): zero-mean derivative basis (unit amplitude)
    phi = np.linspace(0.0, 2.0*np.pi, M, endpoint=False)
    g = np.zeros(M, dtype=np.float64)

    if basis_type == 'gaussian':
        L = basis_params.shape[0]
        acc = np.zeros(M)
        for i in range(L):
            b = max(basis_params[i,1], 1e-6)
            x = ((phi - np.pi) )
            acc += x * np.exp(-0.5*(x/b)**2)
        g = acc / max(L,1)
        g -= g.mean()
        return g

    elif basis_type in ('gamma','skewed-gaussian'):
        x_table = np.linspace(0.0, 2.0*np.pi, M, endpoint=False)
        f_lut = np.zeros(M)
        L = basis_params.shape[0]
        for i in range(L):
            if basis_type == 'gamma':
                alpha, scale = basis_params[i,1], basis_params[i,2]
                for j in range(M):
                    f_lut[j] += gamma_pdf(x_table[j], alpha, scale)
            else:
                b, skew = basis_params[i,1], basis_params[i,2]
                for j in range(M):
                    x = x_table[j] - np.pi
                    f_lut[j] += 2.0 * x * norm_pdf(x, b) * norm_cdf(skew * x / b)
        g = f_lut / max(L,1)
        g -= g.mean()
        return g
    else:
        raise ValueError("Unsupported basis type")

def _primitive_coeffs_from_derivative_fft(g):
    # FFT-based primitive coefficients: G_k = F_k / (ik), k≠0, G_0=0
    G = np.fft.fft(g)
    M = g.size
    k = np.fft.fftfreq(M, d=1.0) * M
    G_new = np.zeros_like(G, dtype=np.complex128)
    for idx in range(M):
        kk = k[idx]
        if kk != 0:
            G_new[idx] = G[idx] / (1j * kk)
        else:
            G_new[idx] = 0.0
    return G_new

def _impulse_train_coeffs(thetai, basis_params, M):
    k = np.fft.fftfreq(M, d=1.0) * M
    S = np.zeros(M, dtype=np.complex128)
    for i in range(thetai.size):
        a = basis_params[i,0]
        S += a * np.exp(-1j * k * thetai[i])
    return S

def build_phase_template_fft(basis_type, thetai, basis_params, M=1024):
    g = _tabulate_zero_mean_derivative(basis_type, np.asarray(basis_params), M)
    Gk = _primitive_coeffs_from_derivative_fft(g)
    Sk = _impulse_train_coeffs(thetai, np.asarray(basis_params), M)
    Zk = - Gk * Sk
    z_grid = np.fft.ifft(Zk).real
    return z_grid


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

def unified_model_fft(ppinterval, fs, seconds, basis_type, thetai, basis_params, M=1024):
    n_samples = int(np.ceil(seconds * fs))
    _, _, theta = _phase_from_rr(ppinterval, fs, n_samples)

    z_grid = build_phase_template_fft(basis_type, thetai, np.asarray(basis_params), M=M)
    z = sample_template(theta, z_grid)

    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    z = detrend(z)
    z -= np.mean(z)
    z = (z - np.min(z)) / (np.max(z) - np.min(z) + 1e-8)
    return z

def unified_model_template(ppinterval, fs, seconds, basis_type, thetai, basis_params, M=1024):
    n_samples = int(np.ceil(seconds * fs))
    _, _, theta = _phase_from_rr(ppinterval, fs, n_samples)

    z_grid = build_phase_template(basis_type, thetai, np.asarray(basis_params), M=M)
    z = sample_template(theta, z_grid)
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    z = detrend(z)
    z -= np.mean(z)
    z = (z - np.min(z)) / (np.max(z) - np.min(z) + 1e-8)
    return z

def unified_model_basis(ppinterval, fs, seconds, basis_type, thetai, basis_params):
    """
    Generate z(t) directly from basis primitives:
    z(θ) = - Σ_i a_i * G_i(θ - θ_i), where G_i is primitive of f_i with zero-mean.
    """
    n_samples = int(np.ceil(seconds * fs))
    # phase trajectory from RR
    rr, omega, theta = _phase_from_rr(ppinterval, fs, n_samples)

    if basis_type == 'gaussian':
        z = _synthesize_gaussian_core(theta, thetai, np.array(basis_params))
    else:
        # precompute primitive LUTs for each basis (unit amplitude)
        M = 500
        x_table = np.linspace(0.0, 2.0*np.pi, M)  # keep a python copy for numba signature
        _, G_lut = _precompute_f_and_G(np.array(basis_params), basis_type, M)
        z = _synthesize_basis_core(theta, thetai, np.array(basis_params), x_table, G_lut)

    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    z = detrend(z)
    z -= np.mean(z)
    z = (z - np.min(z)) / (np.max(z) - np.min(z) + 1e-8)
    return z

# Main model interface
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

    M = 500
    x_table = np.linspace(0, 2 * np.pi, M)
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

def unified_model(ppinterval, fs, seconds, basis_type, thetai, basis_params, solver="rk4"):
    """
    Unified entry point:
      - mode="basis"  → closed-form basis synthesis (no ODE)
      - mode="ode"    → RK ODE (ode_solver in {"rk3","rk4"})
    """
    if solver == "basis":
        return unified_model_basis(ppinterval, fs, seconds, basis_type, thetai, basis_params)
    elif solver == 'template':
        return unified_model_template(ppinterval, fs, seconds, basis_type, thetai, basis_params, M=512)
    elif solver == "fft":
        return unified_model_fft(ppinterval, fs, seconds, basis_type, thetai, basis_params, M=512)
    else:
        return unified_model_ode(ppinterval, fs, seconds, basis_type, thetai, basis_params, solver)


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