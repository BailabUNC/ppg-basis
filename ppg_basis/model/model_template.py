from ppg_basis.utils.solver_utils import _phase_from_rr, sample_template, _precompute_f_and_G
import numpy as np
from scipy.signal import detrend

def unified_model_template(ppinterval, fs, seconds, basis_type, thetai, basis_params, M):
    n_samples = int(np.ceil(seconds * fs))
    _, theta = _phase_from_rr(ppinterval, fs, n_samples)

    z_grid = build_phase_template(basis_type, thetai, np.asarray(basis_params), M)
    z = sample_template(theta, z_grid)
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    z = detrend(z)
    z -= np.mean(z)
    z = (z - np.min(z)) / (np.max(z) - np.min(z) + 1e-8)
    return z

def build_phase_template(basis_type, thetai, basis_params, M):
    if basis_type == 'gaussian':
        return _build_phase_template_gaussian(thetai, np.asarray(basis_params), M)
    else:
        return _build_phase_template_generic(basis_type, thetai, np.asarray(basis_params), M)
    
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
    """
    Compute shifted primitive via Fourier shift.
    """
    _, G_lut = _precompute_f_and_G(np.ascontiguousarray(basis_params), basis_type, M)
    z_grid = np.zeros(M, dtype=np.float64)
    k = np.fft.fftfreq(M, d=1.0 / M)

    for i in range(thetai.size):
        a = basis_params[i, 0]
        # fractional shift in grid points (continuous, not rounded)
        d = (thetai[i] + np.pi) * M / (2.0 * np.pi)
        # Fourier shift: exact circular shift
        G_fft = np.fft.fft(G_lut[i])
        G_shifted = np.fft.ifft(G_fft * np.exp(-1j * 2.0 * np.pi * k * d / M)).real
        z_grid -= a * G_shifted
    return z_grid
