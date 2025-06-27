import numpy as np
from ppg_basis.utils.math_utils import *
from numba import njit

@njit
def objective_function(model, signal, mse_flag: bool=True, corr_flag: bool=True, appg_flag: bool=False):
    if mse_flag:
        mse = np.sum((model - signal) ** 2) / len(model)
    else:
        mse = 0

    if corr_flag:
        corr = 1 - corrcoef_numba(model, signal)
    else:
        corr = 0

    if appg_flag:
        sig_smooth = gaussian_filter1d_numba(signal, sigma=2)
        dsig = gradient_1d(sig_smooth, 1 / 125)
        d2sig = gradient_1d(dsig, 1 / 125)
        mod_smooth = gaussian_filter1d_numba(model, sigma=2)
        dmod = gradient_1d(mod_smooth, 1 / 125)
        d2mod = gradient_1d(dmod, 1 / 125)

        appg = 1 - np.sqrt(np.sum((d2mod - d2sig) ** 2) / np.sum((d2sig - np.mean(d2sig)) ** 2))
    else:
        appg = 0
    return mse + corr + appg