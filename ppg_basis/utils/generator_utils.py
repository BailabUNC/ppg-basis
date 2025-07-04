import numpy as np

def pp_interval_generator(time: float, mu: float = 0, sigma: float = 1):
    """
    Generate pulse-to-pulse interval array
    :param time: total time of PPG window
    :param mu: mean pulse-to-pulse interval variation
    :param sigma: standard deviation of pulse-to-pulse interval variation
    :return: pulse-to-pulse interval array
    """
    if sigma == 0:
        if mu <= 0:
            raise ValueError("pp_interval_generator: mu must be >0 when sigma=0")
        n = int(np.ceil(time/mu))
        return [round(mu,3)] * n
    pp_interval = []
    total = 0.0
    while total < time:
        draw = np.random.normal(mu, sigma)
        if draw <=0:
            continue
        val = round(draw, 3)
        pp_interval.append(val)
        total += val
    return pp_interval