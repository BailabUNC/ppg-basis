import numpy as np
from ppg_basis.utils.math_utils import _wrap_pi, gamma_pdf, norm_pdf, norm_cdf, _interp_uniform_table
from numba import njit, prange

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

@njit(cache=True)
def sample_template(theta, z_grid):
    n = theta.size
    out = np.empty(n, dtype=np.float64)
    for k in range(n):
        out[k] = _interp_uniform_table(theta[k], z_grid)
    return out