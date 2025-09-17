from ppg_basis.utils.solver_utils import _phase_from_rr, _precompute_f_and_G
from ppg_basis.utils.math_utils import _wrap_pi, _interp1d_lut_scalar
import numpy as np
from scipy.signal import detrend
from numba import njit

def unified_model_basis(ppinterval, fs, seconds, basis_type, thetai, basis_params, M):
    """
    Generate z(t) directly from basis primitives:
    z(θ) = - Σ_i a_i * G_i(θ - θ_i), where G_i is primitive of f_i with zero-mean.
    """
    n_samples = int(np.ceil(seconds * fs))
    # phase trajectory from RR
    _, theta = _phase_from_rr(ppinterval, fs, n_samples)

    if basis_type == 'gaussian':
        z = _synthesize_gaussian_core(theta, thetai, np.array(basis_params))
    else:
        # precompute primitive LUTs for each basis (unit amplitude)
        x_table = np.linspace(0.0, 2.0*np.pi, M)  # keep a python copy for numba signature
        _, G_lut = _precompute_f_and_G(np.array(basis_params), basis_type, M)
        z = _synthesize_basis_core(theta, thetai, np.array(basis_params), x_table, G_lut)

    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    z = detrend(z)
    z -= np.mean(z)
    z = (z - np.min(z)) / (np.max(z) - np.min(z) + 1e-8)
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