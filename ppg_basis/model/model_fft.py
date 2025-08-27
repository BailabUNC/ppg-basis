import numpy as np
from scipy.signal import detrend
from ppg_basis.model.solver_utils import _phase_from_rr, sample_template
from ppg_basis.utils.math_utils import gamma_pdf, norm_pdf, norm_cdf

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

def build_phase_template_fft(basis_type, thetai, basis_params, M=1024):
    g = _tabulate_zero_mean_derivative(basis_type, np.asarray(basis_params), M)
    Gk = _primitive_coeffs_from_derivative_fft(g)
    Sk = _impulse_train_coeffs(thetai, np.asarray(basis_params), M)
    Zk = - Gk * Sk
    z_grid = np.fft.ifft(Zk).real
    return z_grid

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