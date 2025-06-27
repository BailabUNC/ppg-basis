import numpy as np

def pp_interval_generator(time: float, mu: float = 0, sigma: float = 1):
    """
    Generate pulse-to-pulse interval array
    :param time: total time of PPG window
    :param mu: mean pulse-to-pulse interval variation
    :param sigma: standard deviation of pulse-to-pulse interval variation
    :return: pulse-to-pulse interval array
    """

    pp_interval = []
    while sum(pp_interval) <= time:
        pp_interval.append(np.round(np.random.normal(mu, sigma),3))
    return pp_interval