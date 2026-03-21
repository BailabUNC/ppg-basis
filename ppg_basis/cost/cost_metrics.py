import numpy as np
from ppg_basis.utils.math_utils import *
import functools
from contextlib import contextmanager

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

def appg(model, signal, fs=125, **kwargs):
    """
    Generate NRMSE of 2nd derivative of PPG (aPPG)
    :param model: reference signal
    :param signal: reconstructed signal
    :param fs: sampling rate (Hz)
    :return: NRMSE
    """
    dt = 1.0 / fs
    sig_smooth = gaussian_filter1d_numba(signal, sigma=2)
    dsig = gradient_1d(sig_smooth, dt)
    d2sig = gradient_1d(dsig, dt)
    mod_smooth = gaussian_filter1d_numba(model, sigma=2)
    dmod = gradient_1d(mod_smooth, dt)
    d2mod = gradient_1d(dmod, dt)
    denom = np.sum((d2sig - np.mean(d2sig)) ** 2)
    if denom < 1e-12:
        return 0.0
    return np.sqrt(np.sum((d2mod - d2sig) ** 2) / denom)

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

def _identity_bounded(x, *_args, **_kw):
    return float(np.clip(x, 0.0, 1.0))

def _corr_normalizer(x, *_args, **_kw):
    return float(np.clip(x / 2.0, 0.0, 1.0))

def _appg_normalizer(x, _n, cfg):
    appg_cfg = (cfg or {}).get("appg", {})
    max_nrmse = appg_cfg.get("max_nrmse", None)
    if max_nrmse is not None and max_nrmse > 0:
        return float(np.clip(x / max_nrmse, 0.0, 1.0))
    k = appg_cfg.get("logistic_k", 1.0)
    return float(1.0 / (1.0 + np.exp(-k * x)))

def _spearman_normalizer(x, n, *_args, **_kw):
    max_s = n * (n**2 - 1) / 3.0 if n >= 2 else 1.0
    return float(np.clip(x / max_s, 0.0, 1.0))

@contextmanager
def _swap_terms(temp_terms):
    global terms
    _old = terms
    terms = temp_terms
    try:
        yield
    finally:
        terms = _old

def normalize_costs_only(config: dict | None = None):
    """
    Decorator factory that normalizes metrics to [0,1] (0=best, 1=worst)
    config:
      {
        "appg": {
          "max_nrmse": float | None,  # linear cap for NRMSE → [0,1] (recommended)
          "logistic_k": float         # steepness if no cap; default 1.0
        }
      }
    """
    def _decorator(f):
        @functools.wraps(f)
        def _wrapped(model, signal, cost_metrics: list, *args, **kwargs):
            n = len(model)
            normalized_terms = {}
            for name, fn in terms.items():
                normalizer = NORMALIZERS.get(name, _identity_bounded)

                def _make_wrapped(name=name, fn=fn, normalizer=normalizer):
                    def _wrapped_metric(m, s):
                        raw = fn(m, s)
                        try:
                            return normalizer(raw, n, config, model=m, signal=s)
                        except TypeError:
                            return normalizer(raw)
                    return _wrapped_metric

                normalized_terms[name] = _make_wrapped()

            with _swap_terms(normalized_terms):
                return f(model, signal, cost_metrics, *args, **kwargs)

        return _wrapped
    return _decorator

terms = {
    "mse" :  mse,
    "corr" : corr,
    "appg" : appg,
    "spearman": spearman,
    "kendall": kendall,
    "gamma": gamma,
    "somers": somers_d
}

NORMALIZERS = {
    "mse": _identity_bounded,
    "corr": _corr_normalizer,
    "appg": _appg_normalizer,
    "spearman": _spearman_normalizer,
    "kendall": _identity_bounded,
    "gamma": _identity_bounded,
    "somers": _identity_bounded,
}