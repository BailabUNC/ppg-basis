import numpy as np
from ppg_basis.utils.math_utils import *

def mse(model, signal):
    """
    Generate MSE cost metric
    :param model: reference signal
    :param signal: reconstructed signal
    :return: MSE
    """
    return np.sum((model - signal) ** 2) / len(model)

def corr(model, signal):
    """
    Generate Pearson's Correlation Coefficient cost metric
    :param model: reference signal
    :param signal: reconstructed signal
    :return: Corr. Coefficient
    """
    return 1 - corrcoef_numba(model, signal)

def appg(model, signal):
    """
    Generate NRMSE of 2nd derivative of PPG (aPPG)
    :param model: reference signal
    :param signal: reconstructed signal
    :return: NRMSE
    """
    sig_smooth = gaussian_filter1d_numba(signal, sigma=2)
    dsig = gradient_1d(sig_smooth, 1 / 125)
    d2sig = gradient_1d(dsig, 1 / 125)
    mod_smooth = gaussian_filter1d_numba(model, sigma=2)
    dmod = gradient_1d(mod_smooth, 1 / 125)
    d2mod = gradient_1d(dmod, 1 / 125)
    return 1 - np.sqrt(np.sum((d2mod - d2sig) ** 2) / np.sum((d2sig - np.mean(d2sig)) ** 2))

# mapping dictionary
terms = {
    "mse" :  mse,
    "corr" : corr,
    "appg" : appg,
}