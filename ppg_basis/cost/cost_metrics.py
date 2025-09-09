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

@njit
def spearman(model, signal):
    """
        Generate Spearman Correlation Coefficient
        :param model: reference signal
        :param signal: reconstructed signal
        :return: Spearman Correlation
        """
    model_rank = rank_array(model)
    signal_rank = rank_array(signal)
    d = model_rank - signal_rank
    return np.sum(d ** 2)  # lower is better

@njit
def kendall(model, signal):
    """
        Generate Kendall Rank Correlation Coefficient
        :param model: reference signal
        :param signal: reconstructed signal
        :return: Kendall's Tau Coefficient
        """
    n = len(model)
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            m_diff = model[i] - model[j]
            s_diff = signal[i] - signal[j]
            if m_diff * s_diff > 0:
                concordant += 1
            elif m_diff * s_diff < 0:
                discordant += 1
    return discordant / (concordant + discordant + 1e-8)  # minimize discordance

@njit
def gamma(model, signal):
    """
        Generate Goodman & Kruskal's Gamma
        :param model: reference signal
        :param signal: reconstructed signal
        :return: Gamma Coefficient
        """
    n = len(model)
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            m_diff = model[i] - model[j]
            s_diff = signal[i] - signal[j]
            if m_diff != 0 and s_diff != 0:
                if m_diff * s_diff > 0:
                    concordant += 1
                elif m_diff * s_diff < 0:
                    discordant += 1
    return discordant / (concordant + discordant + 1e-8)  # minimize bad ranking

@njit
def somers_d(model, signal):
    """
        Generate Somers' D Statistic
        :param model: reference signal
        :param signal: reconstructed signal
        :return: Somers' Statistic
        """
    n = len(model)
    concordant = 0
    discordant = 0
    ties_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            m_diff = model[i] - model[j]
            s_diff = signal[i] - signal[j]
            if s_diff == 0:
                ties_y += 1
            elif m_diff * s_diff > 0:
                concordant += 1
            elif m_diff * s_diff < 0:
                discordant += 1
    return (discordant + ties_y) / (concordant + discordant + ties_y + 1e-8)


# mapping dictionary
terms = {
    "mse" :  mse,
    "corr" : corr,
    "appg" : appg,
    "spearman": spearman,
    "kendall": kendall,
    "gamma": gamma,
    "somers": somers_d
}